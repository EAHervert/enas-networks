import os
import torch
import time

from ENAS_DHDN.TRAINING_FUNCTIONS import evaluate_model, AverageMeter, SSIM, PSNR, nn, np


# Here we train the Shared Network which is sampled from the Controller
def Train_Shared(epoch,
                 controller,
                 shared,
                 shared_optimizer,
                 dataloader_sidd_training,
                 sa_logger,
                 device=None,
                 fixed_arc=None):
    """Train Shared_Autoencoder by sampling architectures from the Controller.

    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        shared_optimizer: Optimizer for the Shared Network.
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
        shared_optimizer.step()

        if i_batch % 100 == 0:
            Display_Loss = "Loss_Shared: %.6f" % loss_batch.val + "\tLoss_Original: %.6f" % loss_original_batch.val
            Display_SSIM = "SSIM_Shared: %.6f" % ssim_batch.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
            Display_PSNR = "PSNR_Shared: %.6f" % psnr_batch.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
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

    controller.train()

    meters = [loss_batch.avg, loss_original_batch.avg, ssim_batch.avg, ssim_original_batch.avg, psnr_batch.avg,
              psnr_original_batch.avg]

    return meters


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

    # Here we will have the following meters for the metrics
    reward_meter = AverageMeter()  # Reward R
    baseline_meter = AverageMeter()  # Baseline b, which controls variance
    val_acc_meter = AverageMeter()  # Validation Accuracy
    loss_meter = AverageMeter()  # Loss

    t1 = time.time()

    controller.zero_grad()
    # Todo: Make this modifiable at start
    choices = [1, 5, 10, 12, 28, 32, 44, 53, 65, 76]  # Randomly select image patches to train controller on
    for i in range(config['Controller']['Controller_Train_Steps'] * config['Controller']['Controller_Num_Aggregate']):
        controller()  # perform forward pass to generate a new architecture
        architecture = controller.sample_arc

        SSIM_Meter = AverageMeter()
        for i_validation, validation_batch in enumerate(dataloader_sidd_validation, start=1):
            if i_validation in choices:
                x_v = validation_batch['NOISY']
                t_v = validation_batch['GT']

                with torch.no_grad():
                    y_v = shared(x_v.to(device), architecture)
                    ssim_val = SSIM(y_v, t_v.to(device)).item()
                    SSIM_Meter.update(ssim_val)  # Now, we will use only SSIM for the accuracy.

        del x_v, y_v, t_v

        # make sure that gradients aren't backpropped through the reward or baseline
        reward = SSIM_Meter.avg
        reward += config['Controller']['Controller_Entropy_Weight'] * controller.sample_entropy
        if baseline is None:
            baseline = SSIM_Meter.avg
        else:
            baseline -= (1 - config['Controller']['Controller_Bl_Dec']) * (baseline - reward)

        loss = -1 * controller.sample_log_prob * (reward - baseline)

        # Average gradient over controller_num_aggregate samples
        loss = loss / config['Controller']['Controller_Num_Aggregate']

        reward_meter.update(reward.item())
        baseline_meter.update(baseline)
        val_acc_meter.update(SSIM_Meter.avg)
        loss_meter.update(loss.item())

        loss.backward(retain_graph=True)

        # Aggregate gradients for controller_num_aggregate iteration, then update weights
        if (i + 1) % config['Controller']['Controller_Num_Aggregate'] == 0:
            nn.utils.clip_grad_norm_(controller.parameters(), config['Shared']['Child_Grad_Bound'])
            controller_optimizer.step()
            controller.zero_grad()

            if (i + 1) % config['Controller']['Controller_Num_Aggregate'] == 0:
                display = 'Epoch_Number=' + str(epoch) + '-' + \
                          str(i // config['Controller']['Controller_Num_Aggregate']) + \
                          '\tController_loss=%+.6f' % loss_meter.val + \
                          '\tEntropy=%.6f' % controller.sample_entropy.item() + \
                          '\tAccuracy (SSIM)=%.6f' % val_acc_meter.val + \
                          '\tBaseline=%.6f' % baseline_meter.val
                print(display)

        SSIM_Meter.reset()

    print()
    print("Controller Average Loss: ", loss_meter.avg)
    print("Controller Average Accuracy (SSIM): ", val_acc_meter.avg)
    print("Controller Average Reward: ", reward_meter.avg)

    t2 = time.time()
    print("Controller Training Time: ", t2 - t1)
    print()

    c_logger.writerow({'Controller_Reward': reward_meter.avg, 'Controller_Accuracy': val_acc_meter.avg,
                       'Controller_Loss': loss_meter.avg})

    shared.train()

    return baseline, loss_meter.avg, val_acc_meter.avg, reward_meter.avg


def Train_ENAS(
        start_epoch,
        num_epochs,
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
        log_every=10,
        eval_every_epoch=1,
        device=None,
        args=None
):
    """Perform architecture search by training a Controller and Shared_Autoencoder.

    Args:
        start_epoch: Epoch to begin on.
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
        results = Train_Shared(
            epoch=epoch,
            controller=controller,
            shared=shared,
            shared_optimizer=shared_optimizer,
            dataloader_sidd_training=dataloader_sidd_training,
            sa_logger=logger[0],
            device=device
        )

        loss, loss_Original, SSIM, SSIM_Original, PSNR, PSNR_Original = results

        vis_window['Shared_Network_Loss'] = vis.line(
            X=np.column_stack([epoch] * 2),
            Y=np.column_stack([loss, loss_Original]),
            win=vis_window['Shared_Network_Loss'],
            opts=dict(title='Shared_Network_Loss', xlabel='Epoch', ylabel='Loss', legend=['Network', 'Original']),
            update='append' if epoch > 0 else None)

        vis_window['Shared_Network_SSIM'] = vis.line(
            X=np.column_stack([epoch] * 2),
            Y=np.column_stack([SSIM, SSIM_Original]),
            win=vis_window['Shared_Network_SSIM'],
            opts=dict(title='Shared_Network_SSIM', xlabel='Epoch', ylabel='SSIM', legend=['Network', 'Original']),
            update='append' if epoch > 0 else None)

        vis_window['Shared_Network_PSNR'] = vis.line(
            X=np.column_stack([epoch] * 2),
            Y=np.column_stack([PSNR, PSNR_Original]),
            win=vis_window['Shared_Network_PSNR'],
            opts=dict(title='Shared_Network_PSNR', xlabel='Epoch', ylabel='PSNR', legend=['Network', 'Original']),
            update='append' if epoch > 0 else None)

        baseline, controller_loss, controller_val, reward = Train_Controller(
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

        vis_window['Controller_Loss'] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([controller_loss]),
            win=vis_window['Controller_Loss'],
            opts=dict(title='Controller_Loss', xlabel='Epoch', ylabel='Loss'),
            update='append' if epoch > 0 else None)

        vis_window['Controller_Validation_Accuracy'] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([controller_val]),
            win=vis_window['Controller_Validation_Accuracy'],
            opts=dict(title='Controller_Validation_Accuracy', xlabel='Epoch', ylabel='Accuracy'),
            update='append' if epoch > 0 else None)

        vis_window['Controller_Reward'] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([reward]),
            win=vis_window['Controller_Reward'],
            opts=dict(title='Controller_Reward', xlabel='Epoch', ylabel='Reward'),
            update='append' if epoch > 0 else None)

        if epoch % eval_every_epoch == 0:
            evaluate_model(epoch=epoch,
                           controller=controller,
                           shared=shared,
                           dataloader_sidd_validation=dataloader_sidd_validation,
                           device=device)
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
