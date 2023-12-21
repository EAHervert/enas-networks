import os
import torch
import time
import random

from ENAS_DHDN.TRAINING_FUNCTIONS import evaluate_model, AverageMeter, SSIM, PSNR, nn, np


# Here we train the Shared Network which is sampled from the Controller
def Train_Shared(epoch,
                 passes,
                 controller,
                 shared,
                 shared_optimizer,
                 config,
                 dataloader_sidd_training,
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
        sa_logger: Logs the Shared network Loss and SSIM
        device: The GPU that we will use.
        fixed_arc: Architecture to train, overrides the controller sample.
        ...

    Returns: Nothing.
    """
    # Here we are using the Controller to give us the networks.
    # We don't modify the Controller, so we have it in evaluation mode rather than training mode.
    controller.eval()
    shared.train()

    # Keep track of the accuracy and loss through the process.
    loss_batch = AverageMeter()
    loss_original_batch = AverageMeter()
    ssim_batch = AverageMeter()  # Doubles as the accuracy.
    ssim_original_batch = AverageMeter()
    psnr_batch = AverageMeter()
    psnr_original_batch = AverageMeter()

    loss = nn.L1Loss()
    mse = nn.MSELoss()

    if device is not None:
        loss = loss.to(device)
        mse = mse.to(device)

    # Start the timer for the epoch.
    t1 = time.time()
    for pass_ in range(passes):
        for i_batch, sample_batch in enumerate(dataloader_sidd_training):

            # Pick an architecture to work with from the Graph Network (Shared)
            if fixed_arc is None:
                # Since we are just training the Autoencoders, we do not need to keep track of gradients for Controller.
                with torch.no_grad():
                    controller()
                architecture = controller.sample_arc
            else:
                architecture = fixed_arc

            x = sample_batch['NOISY']
            y = shared(x.to(device), architecture)  # Net is the output of the network
            t = sample_batch['GT']

            loss_value = loss(y, t.to(device))
            loss_batch.update(loss_value.item())

            # Calculate values not needing to be backpropagated
            with torch.no_grad():
                loss_original_batch.update(loss(x.to(device), t.to(device)).item())

                ssim_batch.update(SSIM(y, t.to(device)).item())
                ssim_original_batch.update(SSIM(x, t).item())

                psnr_batch.update(PSNR(mse(y, t.to(device))).item())
                psnr_original_batch.update(PSNR(mse(x.to(device), t.to(device))).item())

            # Backpropagate to train model
            shared_optimizer.zero_grad()
            loss_value.backward()
            nn.utils.clip_grad_norm_(shared.parameters(), config['Shared']['Child_Grad_Bound'])
            shared_optimizer.step()

            if i_batch % 100 == 0:
                Display_Loss = "Loss_Shared: %.6f" % loss_batch.val + "\tLoss_Original: %.6f" % loss_original_batch.val
                Display_SSIM = "SSIM_Shared: %.6f" % ssim_batch.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
                Display_PSNR = "PSNR_Shared: %.6f" % psnr_batch.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

                print("Training Data for Epoch: ", epoch, "Pass:", pass_, "Image Batch: ", i_batch)
                print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

            # Free up space in GPU
            del x, y, t

    Display_Loss = "Loss_Shared: %.6f" % loss_batch.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_Shared: %.6f" % ssim_batch.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_Shared: %.6f" % psnr_batch.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    t2 = time.time()
    print('\n' + '-' * 160)
    print("Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
    print("Epoch {epoch} Training Time: ".format(epoch=epoch), t2 - t1)
    print('-' * 160 + '\n')

    sa_logger.writerow({'Shared_Loss': loss_batch.avg, 'Shared_Accuracy': ssim_batch.avg})

    dict_meters = {'Loss': loss_batch.avg, 'Loss_Original': loss_original_batch.avg, 'SSIM': ssim_batch.avg,
                   'SSIM_Original': ssim_original_batch.avg, 'PSNR': psnr_batch.avg,
                   'PSNR_Original': psnr_original_batch.avg}

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
        # Randomly selects two batches to run the validation
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

        loss = -1 * controller.sample_log_prob * (reward - baseline)

        reward_meter.update(reward)
        baseline_meter.update(baseline)
        val_acc_meter.update(normalized_accuracy)
        loss_meter.update(loss.item())

        loss.backward(retain_graph=True)

        # Aggregate gradients for controller_num_aggregate iteration, then update weights
        if (i + 1) % config['Controller']['Controller_Num_Aggregate'] == 0:
            nn.utils.clip_grad_norm_(controller.parameters(), config['Controller']['Controller_Grad_Bound'])
            controller_optimizer.step()
            controller.zero_grad()

            display = 'Epoch_Number=' + str(epoch) + '-' + \
                      str(i // config['Controller']['Controller_Num_Aggregate']) + \
                      '\tController_loss=%+.6f' % loss_meter.val + \
                      '\tEntropy=%.6f' % controller.sample_entropy.item() + \
                      '\tAccuracy (Normalized SSIM)=%.6f' % val_acc_meter.val + \
                      '\tBaseline=%.6f' % baseline_meter.val
            print(display)

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
        arc_bools=[True, True, True],
        log_every=10,
        eval_every_epoch=1,
        device=None,
        args=None
):
    """Perform architecture search by training a Controller and Shared_Autoencoder.

    Args:
        start_epoch: Epoch to begin on.
        passes: Number of passes though the training data.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        shared_optimizer: Optimizer for the Shared_Autoencoder.
        controller_optimizer: Optimizer for the Controller.
        config: config for the hyperparameters.
        log_every: how often we output the results at the iteration.
        ...

    Returns: Nothing.
    """

    # Hyperparameters
    dir_current = os.getcwd()
    if not os.path.exists(dir_current + '/models/'):
        os.makedirs(dir_current + '/models/')

    baseline = None
    for epoch in range(start_epoch, num_epochs):
        print("Epoch ", str(epoch), ": Training Shared Network")
        training_results = Train_Shared(
            epoch=epoch,
            passes=passes,
            controller=controller,
            shared=shared,
            shared_optimizer=shared_optimizer,
            config=config,
            dataloader_sidd_training=dataloader_sidd_training,
            sa_logger=logger[0],
            device=device
        )

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

        baseline = controller_results['Baseline']  # Update the baseline for variance control

        validation_results = evaluate_model(epoch=epoch, use_random=False, controller=controller, shared=shared,
                                            dataloader_sidd_validation=dataloader_sidd_validation, config=config,
                                            kernel_bool=arc_bools[0], down_bool=arc_bools[1], up_bool=arc_bools[2],
                                            device=device)

        Legend = ['Shared_Train', 'Orig_Train', 'Shared_Val', 'Orig_Val']

        vis_window[list(vis_window)[0]] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([training_results['Loss'], training_results['Loss_Original'], validation_results['Loss'],
                               validation_results['Loss_Original']]),
            win=vis_window[list(vis_window)[0]],
            opts=dict(title=list(vis_window)[0], xlabel='Epoch', ylabel='Loss', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[1]] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([training_results['SSIM'], training_results['SSIM_Original'], validation_results['SSIM'],
                               validation_results['SSIM_Original']]),
            win=vis_window[list(vis_window)[1]],
            opts=dict(title=list(vis_window)[1], xlabel='Epoch', ylabel='SSIM', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[2]] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([training_results['PSNR'], training_results['PSNR_Original'], validation_results['PSNR'],
                               validation_results['PSNR_Original']]),
            win=vis_window[list(vis_window)[2]],
            opts=dict(title=list(vis_window)[2], xlabel='Epoch', ylabel='PSNR', legend=Legend),
            update='append' if epoch > 0 else None)

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

        '''
        state = {'Epoch': Epoch,
                 'args': args,
                 'Shared_State_Dict': Shared.state_dict(),
                 'Controller_State_Dict': Controller.state_dict(),
                 'Shared_Optimizer': Shared_Optimizer.state_dict(),
                 'Controller_Optimizer': Controller_Optimizer.state_dict()}
        filename = 'Checkpoints/' + Output_File + '.pth.tar'
        torch.save(state, filename)
        print()
        '''

        shared_scheduler.step()

        print()
