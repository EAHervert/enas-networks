import os
import numpy as np
import torch
import torch.nn as nn
import time
import random

from utilities.functions import SSIM
from ENAS_DHDN.TRAINING_FUNCTIONS import evaluate_model, AverageMeter, train_loop


# Here we train the Shared Network which is sampled from the Controller
def Train_Shared(epoch,
                 passes,
                 controller,
                 shared,
                 shared_optimizer,
                 config,
                 dataloader_sidd_training,
                 arc_bools,
                 sa_logger,
                 device=None,
                 fixed_arc=None):
    """Train Shared_Autoencoder by sampling architectures from the Controller.

    Args:
        epoch: Current epoch.
        passes: Number of passes though the training data.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        shared_optimizer: Optimizer for the Shared Network.
        config: config for the hyperparameters.
        dataloader_sidd_training: Training dataset.
        arc_bools: Booleans for architecture selection
        sa_logger: Logs the Shared network Loss and SSIM
        device: The GPU that we will use.
        fixed_arc: Architecture to train, overrides the controller sample.
        ...

    Returns: Nothing.
    """
    # Here we are using the Controller to give us the networks.
    # We don't modify the Controller, so we have it in evaluation mode rather than training mode.
    if controller is not None:
        controller.eval()
    shared.train()
    t1 = time.time()

    results_train = train_loop(epoch=epoch,
                               controller=controller,
                               shared=shared,
                               shared_optimizer=shared_optimizer,
                               config=config,
                               dataloader_sidd_training=dataloader_sidd_training,
                               fixed_arc=fixed_arc,
                               arc_bools=arc_bools,
                               passes=passes,
                               device=device)

    Display_Loss = ("Loss_Shared: %.6f" % results_train['Loss'] +
                    "\tLoss_Original: %.6f" % results_train['Loss_Original'])
    Display_SSIM = ("SSIM_Shared: %.6f" % results_train['SSIM'] +
                    "\tSSIM_Original: %.6f" % results_train['SSIM_Original'])
    Display_PSNR = ("PSNR_Shared: %.6f" % results_train['PSNR'] +
                    "\tPSNR_Original: %.6f" % results_train['PSNR_Original'])

    t2 = time.time()
    print('\n' + '-' * 120)
    print("Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
    print("Epoch {epoch} Training Time: ".format(epoch=epoch), t2 - t1)
    print('-' * 120 + '\n')

    sa_logger.writerow({'Shared_Loss': results_train['Loss'], 'Shared_Accuracy': results_train['SSIM']})

    dict_meters = {'Loss': results_train['Loss'], 'Loss_Original': results_train['Loss_Original'],
                   'SSIM': results_train['SSIM'], 'SSIM_Original': results_train['SSIM_Original'],
                   'PSNR': results_train['PSNR'], 'PSNR_Original': results_train['PSNR_Original']}

    return dict_meters


# This is for training the controller network, which we do once we have gone through the exploration of the child
# architectures.
def Train_Controller(epoch,
                     controller,
                     shared,
                     controller_optimizer,
                     dataloader_sidd_validation,
                     c_logger,
                     config,
                     baseline=None,
                     device=None
                     ):
    """Train controller to optimizer validation accuracy using REINFORCE.

    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        controller_optimizer: Optimizer for the Controller.
        dataloader_sidd_validation: Validation dataset.
        c_logger: Logger for the Controller
        config: config for the hyperparameters.
        baseline: The baseline score (i.e. average val_acc) from the previous epoch
        device: The GPU that we will use.

    Returns:
        baseline: The baseline score (i.e. average val_acc) for the current epoch

    For more stable training we perform weight updates using the average of
    many gradient estimates. controller_num_aggregate indicates how many samples
    we want to average over (default = 20). By default, PyTorch will sum gradients
    each time .backward() is called (as long as an optimizer step is not taken),
    so each iteration we divide the loss by controller_num_aggregate to get the
    average.

    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L270
    """
    print('Epoch ' + str(epoch) + ': Training Controller')

    shared.eval()  # We are fixing an architecture and using a fixed architecture for training of the controller
    controller.train()

    # Here we will have the following meters for the metrics
    reward_meter = AverageMeter()  # Reward R
    baseline_meter = AverageMeter()  # Baseline b, which controls variance
    val_acc_meter = AverageMeter()  # Validation Accuracy
    loss_meter = AverageMeter()  # Loss
    SSIM_Meter = AverageMeter()
    SSIM_Original_Meter = AverageMeter()

    t1 = time.time()

    controller.zero_grad()
    for i in range(config['Controller']['Controller_Train_Steps'] * config['Controller']['Controller_Num_Aggregate']):
        # Randomly selects "validation_samples" batches to run the validation for each controller_num_aggregate
        if i % config['Controller']['Controller_Num_Aggregate'] == 0:
            choices = random.sample(range(80), k=config['Training']['Validation_Samples'])
        controller()  # perform forward pass to generate a new architecture
        architecture = controller.sample_arc

        for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
            if i_validation in choices:
                x_v = validation_batch['NOISY']
                t_v = validation_batch['GT']

                with torch.no_grad():
                    y_v = shared(x_v.to(device), architecture)
                    SSIM_Meter.update(SSIM(y_v, t_v.to(device)).item())
                    SSIM_Original_Meter.update(SSIM(x_v, t_v).item())

        # Use the Normalized SSIM improvement as the accuracy
        normalized_accuracy = (SSIM_Meter.avg - SSIM_Original_Meter.avg) / (1 - SSIM_Original_Meter.avg)

        # make sure that gradients aren't backpropped through the reward or baseline
        reward = normalized_accuracy
        reward += config['Controller']['Controller_Entropy_Weight'] * controller.sample_entropy.item()
        if baseline is None:
            baseline = normalized_accuracy
        else:
            baseline -= (1 - config['Controller']['Controller_Bl_Dec']) * (baseline - reward)

        loss = - controller.sample_log_prob * (reward - baseline)

        reward_meter.update(reward)
        baseline_meter.update(baseline)
        val_acc_meter.update(normalized_accuracy)
        loss_meter.update(loss.item())

        # Controller Update:
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(controller.parameters(), config['Controller']['Controller_Grad_Bound'])
        controller_optimizer.step()
        controller.zero_grad()

        # Aggregate gradients for controller_num_aggregate iteration, then update weights
        if (i + 1) % config['Controller']['Controller_Num_Aggregate'] == 0:
            display = 'Epoch_Number=' + str(epoch) + '-' + \
                      str(i // config['Controller']['Controller_Num_Aggregate']) + \
                      '\tController_log_probs=%+.6f' % controller.sample_log_prob.item() + \
                      '\tController_loss=%+.6f' % loss_meter.val + \
                      '\tEntropy=%.6f' % controller.sample_entropy.item() + \
                      '\tAccuracy (Normalized SSIM)=%.6f' % val_acc_meter.val + \
                      '\tReward=%.6f' % reward_meter.val + \
                      '\tBaseline=%.6f' % baseline_meter.val
            print(display)
            baseline = None

        del x_v, y_v, t_v
        SSIM_Meter.reset()
        SSIM_Original_Meter.reset()

    print()
    print("Controller Average Loss: ", loss_meter.avg)
    print("Controller Average Accuracy (Normalized SSIM): ", val_acc_meter.avg)
    print("Controller Average Reward: ", reward_meter.avg)

    t2 = time.time()
    print("Controller Training Time: ", t2 - t1)
    print()

    c_logger.writerow({'Controller_Reward': reward_meter.avg, 'Controller_Accuracy': val_acc_meter.avg,
                       'Controller_Loss': loss_meter.avg})

    controller_dict = {'Baseline': baseline, 'Loss': loss_meter.avg, 'Accuracy': val_acc_meter.avg,
                       'Reward': reward_meter.avg}

    return controller_dict


def Train_ENAS(
        start_epoch,
        pre_train_epochs,
        num_epochs,
        passes,
        controller,
        shared,
        shared_optimizer,
        controller_optimizer,
        shared_scheduler,
        dataloader_sidd_training,
        dataloader_sidd_validation,
        logger,
        vis,
        vis_window,
        config,
        arc_bools=None,
        sample_size=-1,
        device=None,
        pre_train_controller=False
):
    """Perform architecture search by training a Controller and Shared_Autoencoder.

    Args:
        start_epoch: Epoch to begin on.
        pre_train_epochs: Number of epochs to pre-train the model randomly (Get better results).
        num_epochs: Number of epochs to loop through.
        passes: Number of passes though the training data.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        shared_optimizer: Optimizer for the Shared_Autoencoder.
        controller_optimizer: Optimizer for the Controller.
        shared_scheduler: Controls the learning rate for the Shared_Autoencoder.
        dataloader_sidd_training: Training dataset.
        dataloader_sidd_validation: Validation dataset.
        config: config for the hyperparameters.
        sample_size: Number of the validation samples we will use for evaluation, -1 for all samples.
        device: The GPU that we will use.
        pre_train_controller: Pre-Training the controller when we have pre-trained shared network (optional).
        ...

    Returns: Nothing.
    """

    if arc_bools is None:
        arc_bools = [True, True, True]
    dir_current = os.getcwd()
    if not os.path.exists(dir_current + '/models/'):
        os.makedirs(dir_current + '/models/')

    baseline = None

    # Pre-Training model randomly to get starting point for convergence.
    if pre_train_epochs > 0:
        print('\n' + '-' * 120)
        print("Begin Pre-training.")
        print('-' * 120 + '\n')

        for i in range(pre_train_epochs):
            train_loop(epoch=i,
                       controller=None,
                       shared=shared,
                       shared_optimizer=shared_optimizer,
                       config=config,
                       dataloader_sidd_training=dataloader_sidd_training,
                       arc_bools=arc_bools,
                       fixed_arc=None,
                       device=device,
                       pre_train=True)

        print('\n' + '-' * 120)
        print("End Pre-training.")
        print('-' * 120 + '\n')

    if pre_train_controller and controller is not None:
        _ = Train_Controller(
            epoch=-1,
            controller=controller,
            shared=shared,
            controller_optimizer=controller_optimizer,
            dataloader_sidd_validation=dataloader_sidd_validation,
            c_logger=logger[1],
            config=config,
            baseline=baseline,
            device=device
        )
        baseline = None

    for epoch in range(start_epoch, num_epochs):
        training_results = Train_Shared(
            epoch=epoch,
            passes=passes,
            controller=controller,
            shared=shared,
            shared_optimizer=shared_optimizer,
            config=config,
            dataloader_sidd_training=dataloader_sidd_training,
            arc_bools=arc_bools,
            sa_logger=logger[0],
            device=device
        )
        if controller is not None:
            controller_results = Train_Controller(
                epoch=epoch,
                controller=controller,
                shared=shared,
                controller_optimizer=controller_optimizer,
                dataloader_sidd_validation=dataloader_sidd_validation,
                c_logger=logger[1],
                config=config,
                baseline=baseline,
                device=device
            )
            baseline = None
        else:
            controller_results = None
        validation_results = evaluate_model(epoch=epoch,
                                            controller=controller,
                                            shared=shared,
                                            dataloader_sidd_validation=dataloader_sidd_validation,
                                            config=config,
                                            arc_bools=arc_bools,
                                            sample_size=sample_size,
                                            device=device)

        Legend = ['Shared_Train', 'Orig_Train', 'Shared_Val', 'Orig_Val']

        vis_window[list(vis_window)[0]] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([training_results['Loss'], training_results['Loss_Original'],
                               validation_results['Validation_Loss'], validation_results['Validation_Loss_Original']]),
            win=vis_window[list(vis_window)[0]],
            opts=dict(title=list(vis_window)[0], xlabel='Epoch', ylabel='Loss', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[1]] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([training_results['SSIM'], training_results['SSIM_Original'],
                               validation_results['Validation_SSIM'], validation_results['Validation_SSIM_Original']]),
            win=vis_window[list(vis_window)[1]],
            opts=dict(title=list(vis_window)[1], xlabel='Epoch', ylabel='SSIM', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[2]] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([training_results['PSNR'], training_results['PSNR_Original'],
                               validation_results['Validation_PSNR'], validation_results['Validation_PSNR_Original']]),
            win=vis_window[list(vis_window)[2]],
            opts=dict(title=list(vis_window)[2], xlabel='Epoch', ylabel='PSNR', legend=Legend),
            update='append' if epoch > 0 else None)

        if controller is not None:
            vis_window[list(vis_window)[3]] = vis.line(
                X=np.column_stack([epoch]),
                Y=np.column_stack([controller_results['Loss']]),
                win=vis_window[list(vis_window)[3]],
                opts=dict(title=list(vis_window)[3], xlabel='Epoch', ylabel='Loss'),
                update='append' if epoch > 0 else None)

            vis_window[list(vis_window)[4]] = vis.line(
                X=np.column_stack([epoch]),
                Y=np.column_stack([controller_results['Accuracy']]),
                win=vis_window[list(vis_window)[4]],
                opts=dict(title=list(vis_window)[4], xlabel='Epoch', ylabel='Accuracy'),
                update='append' if epoch > 0 else None)

            vis_window[list(vis_window)[5]] = vis.line(
                X=np.column_stack([epoch]),
                Y=np.column_stack([controller_results['Reward']]),
                win=vis_window[list(vis_window)[5]],
                opts=dict(title=list(vis_window)[5], xlabel='Epoch', ylabel='Reward'),
                update='append' if epoch > 0 else None)

            logger[2].writerow({'Loss_Batch': training_results['Loss'],
                                'Loss_Val': validation_results['Validation_Loss'],
                                'Loss_Original_Train': training_results['Loss_Original'],
                                'Loss_Original_Val': validation_results['Validation_Loss_Original'],
                                'SSIM_Batch': training_results['SSIM'],
                                'SSIM_Val': validation_results['Validation_SSIM'],
                                'SSIM_Original_Train': training_results['SSIM_Original'],
                                'SSIM_Original_Val': validation_results['Validation_SSIM_Original'],
                                'PSNR_Batch': training_results['PSNR'],
                                'PSNR_Val': validation_results['Validation_PSNR'],
                                'PSNR_Original_Train': training_results['PSNR_Original'],
                                'PSNR_Original_Val': validation_results['Validation_PSNR_Original']})

        shared_scheduler.step()

        print()
