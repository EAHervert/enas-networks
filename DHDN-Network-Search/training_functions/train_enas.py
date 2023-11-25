import visdom
import numpy as np
import os

from utilities import dataset
from torch.utils.data import DataLoader
from train_networks import Train_Shared, Train_Controller
from train_functions import evaluate_model

def Train_ENAS(
        start_epoch,
        num_epochs,
        controller,
        shared,
        shared_optimizer,
        controller_optimizer,
        shared_scheduler,
        logger,
        config,
        log_every=10,
        eval_every_epoch=1,
        device=None,
        output_file="Out",
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

    Returns: Nothing.
    """

    # This window will show the SSIM and PSNR of the different networks.
    vis = visdom.Visdom(
        server='eng-ml-01.utdallas.edu',
        port=8097,
        use_incoming_socket=False
    )

    vis.env = output_file
    vis_window = {
        'Shared_Network_Loss': None, 'Shared_Network_SSIM': None,
        'Shared_Network_PSNR': None, 'Controller_Loss': None,
        'Controller_Validation_Accuracy': None, 'Controller_Reward': None
    }

    # Hyperparameters
    dir_current = os.getcwd()
    if not os.path.exists(dir_current + '/models/'):
        os.makedirs(dir_current + '/models/')

    # Noise Dataset
    path_training = dir_current + '/instances/' + config['Training_CSV']
    path_validation_noisy = dir_current + config['Locations']['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations']['Validation_GT']

    SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing())
    SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy, mat_gt_file=path_validation_gt)

    dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=config['Training']['Train_Batch_Size'],
                                          shuffle=True, num_workers=16)

    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                            batch_size=config['Training']['Validation_Batch_Size'],
                                            shuffle=False, num_workers=8)

    baseline = None
    for epoch in range(start_epoch, num_epochs):
        print("Epoch ", str(epoch), ": Training Shared Network")
        loss, SSIM, SSIM_Original, PSNR, PSNR_Original = Train_Shared(
            epoch=epoch,
            controller=controller,
            shared=shared,
            shared_optimizer=shared_optimizer,
            dataloader_sidd_training=dataloader_sidd_training,
            sa_logger=logger[0],
            config=config,
            device=device
        )

        vis_window['Shared_Network_Loss'] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([loss]),
            win=vis_window['Shared_Network_Loss'],
            opts=dict(title='Shared_Network_Loss', xlabel='Epoch', ylabel='Loss'),
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
            evaluate_model(epoch, controller, shared, device=device)
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
