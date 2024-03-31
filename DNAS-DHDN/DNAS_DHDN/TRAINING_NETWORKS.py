import os
import numpy as np
import time

from utilities.functions import SSIM, AverageMeter
from DNAS_DHDN.TRAINING_FUNCTIONS import evaluate_model, train_loop


# Here we train the Shared Network which is sampled from the Controller
def Train_Shared(epoch,
                 passes,
                 weights,
                 shared,
                 shared_optimizer,
                 config,
                 dataloader_sidd_training,
                 da_logger,
                 device=None):
    """Train Shared_Autoencoder by sampling architectures from the Controller.

    Args:
        epoch: Current epoch.
        passes: Number of passes though the training data.
        weights: weights to generate alphas.
        shared: Network that contains all possible architectures, with shared weights.
        shared_optimizer: Optimizer for the Shared Network.
        config: config for the hyperparameters.
        dataloader_sidd_training: Training dataset.
        da_logger: Logs the Shared network Loss and SSIM
        device: The GPU that we will use.
        ...

    Returns: Nothing.
    """
    shared.train()
    t1 = time.time()

    results_train = train_loop(epoch=epoch,
                               weights=weights,
                               shared=shared,
                               shared_optimizer=shared_optimizer,
                               config=config,
                               dataloader_sidd_training=dataloader_sidd_training,
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

    da_logger.writerow({'Differential_Loss': results_train['Loss'], 'Differential_Accuracy': results_train['SSIM']})

    dict_meters = {'Loss': results_train['Loss'], 'Loss_Original': results_train['Loss_Original'],
                   'SSIM': results_train['SSIM'], 'SSIM_Original': results_train['SSIM_Original'],
                   'PSNR': results_train['PSNR'], 'PSNR_Original': results_train['PSNR_Original']}

    return dict_meters


def Train_Alphas(
        epoch: int,
        weights,
        shared,
        eta,
        config,
        dataloader_sidd_training,
        dataloader_sidd_validation,
        da_logger,
        device):
    # return dict_meters
    return 0


def Train_DNAS(
        start_epoch,
        pre_train_epochs,
        num_epochs,
        passes,
        shared,
        shared_optimizer,
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
        shared: Network that contains all possible architectures, with shared weights.
        shared_optimizer: Optimizer for the Shared_Autoencoder.
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
                       alphas=alphas,
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

    for epoch in range(start_epoch, num_epochs):
        training_results = Train_Shared(
            epoch=epoch,
            passes=passes,
            alphas=alphas,
            shared=shared,
            shared_optimizer=shared_optimizer,
            config=config,
            dataloader_sidd_training=dataloader_sidd_training,
            arc_bools=arc_bools,
            sa_logger=logger[0],
            device=device
        )
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
