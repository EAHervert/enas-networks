import os
import sys
from utilities import dataset
from DNAS_DHDN import TRAINING_NETWORKS
from DNAS_DHDN import DIFFERENTIABLE_DHDN
import datetime
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR  # Changing the learning rate for the Shared
from torch.utils.data import DataLoader
import visdom
import argparse

from utilities.utils import CSVLogger, Logger
from utilities.functions import display_time, list_of_ints, generate_alphas

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='ENAS_SEARCH_DHDN')

parser.add_argument('--output_file', default='Pre_Train_DHDN', type=str)

# Training:
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', default='cuda:0', type=str)  # GPU to use
# Put shared network on two devices instead of one
parser.add_argument('--data_parallel', default=True, type=lambda x: (str(x).lower() == 'true'))
# To do outer sums for models
parser.add_argument('--cutout_images', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--outer_sum', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--fixed_arc', default=[], type=list_of_ints)  # Overrides the controller sample
parser.add_argument('--training_csv', default='sidd_np_instances_064_0032.csv', type=str)  # training samples to use
parser.add_argument('--load_differential', default=False, type=lambda x: (str(x).lower() == 'true'))  # Load diff model
parser.add_argument('--model_differential_path', default='differential_network_sidd_0032.pth', type=str)

args = parser.parse_args()


# Now, let us run all these pieces and have out program train the controller.
def main():
    global args

    current_time = datetime.datetime.now()
    d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

    # Hyperparameters
    dir_current = os.getcwd()
    config_path = dir_current + '/configs/config_differential.json'
    config = json.load(open(config_path))
    model_diff_path = '/models/' + args.model_differential_path

    if not os.path.exists(dir_current + '/models/'):
        os.makedirs(dir_current + '/models/')

    # Define the devices:
    device_0 = torch.device(args.device)

    if not os.path.isdir('Logs_DHDN/'):
        os.mkdir('Logs_DHDN/')

    Output_Path = 'Logs_DHDN/' + args.output_file + '/'
    if not os.path.isdir(Output_Path):
        os.mkdir(Output_Path)

    Result_Path = Output_Path + d1
    if not os.path.isdir(Result_Path):
        os.mkdir(Result_Path)

    Model_Path = 'models/' + d1
    if not os.path.isdir(Model_Path):
        os.mkdir(Model_Path)

    # Let us create the loggers to keep track of the Losses, Accuracies, and Rewards.
    File_Name_DA = Result_Path + '/differential_autoencoder.log'
    Field_Names_DA = ['Differential_Loss', 'Differential_Accuracy']
    DA_Logger = CSVLogger(fieldnames=Field_Names_DA, filename=File_Name_DA)

    # Create the CSV Logger:
    File_Name = Result_Path + '/data.csv'
    Field_Names = ['Loss_Batch', 'Loss_Original_Train', 'SSIM_Batch', 'SSIM_Original_Train', 'PSNR_Batch',
                   'PSNR_Original_Train']
    CSV_Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

    # This window will show the SSIM and PSNR of the different networks.
    vis = visdom.Visdom(
        server='eng-ml-01.utdallas.edu',
        port=8097,
        use_incoming_socket=False
    )

    vis.env = args.output_file
    vis_window = {
        'DN_Loss_{d1}'.format(d1=d1): None, 'DN_SSIM_{d1}'.format(d1=d1): None, 'DN_PSNR_{d1}'.format(d1=d1): None
    }

    t_init = time.time()
    np.random.seed(args.seed)

    if config['CUDA']['Device0']:
        torch.cuda.manual_seed(args.seed)

    sys.stdout = Logger(filename=Result_Path + '/log.log')

    print(args)
    print()
    print(config)
    print()

    Diff_Autoencoder = DIFFERENTIABLE_DHDN.DifferentiableDHDN(
        k_value=config['Differential']['K_Value'],
        channels=config['Differential']['Channels'],
        outer_sum=args.outer_sum
    )

    alphas = generate_alphas()

    if args.load_differential:
        state_dict_diff = torch.load(dir_current + model_diff_path, map_location='cpu')
        Diff_Autoencoder.load_state_dict(state_dict_diff)

    if args.data_parallel:
        Diff_Autoencoder = nn.DataParallel(Diff_Autoencoder, device_ids=[0, 1]).cuda()
    else:
        Diff_Autoencoder = Diff_Autoencoder.to(device_0)

    # We will use ADAM on the child network (Different from Original ENAS paper)
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L213
    # Shared_Autoencoder_Optimizer = torch.optim.Adam(params=Shared_Autoencoder.parameters(),
    #                                                 lr=config['Shared']['Child_lr'],
    #                                                 weight_decay=config['Shared']['Weight_Decay'])

    Diff_Autoencoder_Optimizer = torch.optim.Adam(params=Diff_Autoencoder.parameters(),
                                                  lr=config['Differential']['Child_lr'])

    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L154
    # Use step LR scheduler instead of Cosine Annealing
    Diff_Autoencoder_Scheduler = StepLR(
        optimizer=Diff_Autoencoder_Optimizer,
        step_size=config['Differential']['Step_Size'],
        gamma=config['Differential']['Child_gamma']
    )

    # Noise Dataset
    path_training = dir_current + '/instances/' + args.training_csv

    # Todo: Make function that returns these datasets.
    SIDD_training = dataset.DatasetNoise(csv_file=path_training,
                                         transform=dataset.RandomProcessing(cutout_images=args.cutout_images),
                                         device=device_0)
    dataloader_sidd_training = DataLoader(dataset=SIDD_training,
                                          batch_size=config['Training']['Train_Batch_Size'],
                                          shuffle=True)
    if not args.fixed_arc:
        fixed_arc = None
        print('-' * 120 + '\nUsing randomly generated architectures.' + '\n' + '-' * 120)
    else:
        fixed_arc = args.fixed_arc
        print('-' * 120 + '\nUsing Fixed architecture: ', fixed_arc, + '\n' + '-' * 120)

    for epoch in range(args.epochs):
        # Train the weights of the shared network
        training_results = TRAINING_NETWORKS.Train_Shared(epoch=epoch,
                                                          passes=1,
                                                          alphas=alphas,
                                                          shared=Diff_Autoencoder,
                                                          shared_optimizer=Diff_Autoencoder_Optimizer,
                                                          config=config,
                                                          dataloader_sidd_training=dataloader_sidd_training,
                                                          da_logger=DA_Logger,
                                                          device=device_0,
                                                          )

        # Train the operation weights on validation data
        Legend = ['Shared_Train', 'Orig_Train']

        vis_window[list(vis_window)[0]] = vis.line(
            X=np.column_stack([epoch] * 2),
            Y=np.column_stack([training_results['Loss'], training_results['Loss_Original']]),
            win=vis_window[list(vis_window)[0]],
            opts=dict(title=list(vis_window)[0], xlabel='Epoch', ylabel='Loss', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[1]] = vis.line(
            X=np.column_stack([epoch] * 2),
            Y=np.column_stack([training_results['SSIM'], training_results['SSIM_Original']]),
            win=vis_window[list(vis_window)[1]],
            opts=dict(title=list(vis_window)[1], xlabel='Epoch', ylabel='SSIM', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[2]] = vis.line(
            X=np.column_stack([epoch] * 2),
            Y=np.column_stack([training_results['PSNR'], training_results['PSNR_Original']]),
            win=vis_window[list(vis_window)[2]],
            opts=dict(title=list(vis_window)[2], xlabel='Epoch', ylabel='PSNR', legend=Legend),
            update='append' if epoch > 0 else None)

        CSV_Logger.writerow({'Loss_Batch': training_results['Loss'],
                             'Loss_Original_Train': training_results['Loss_Original'],
                             'SSIM_Batch': training_results['SSIM'],
                             'SSIM_Original_Train': training_results['SSIM_Original'],
                             'PSNR_Batch': training_results['PSNR'],
                             'PSNR_Original_Train': training_results['PSNR_Original']
                             })

        Diff_Autoencoder_Scheduler.step()

    DA_Logger.close()
    CSV_Logger.close()

    t_final = time.time()

    display_time(t_final - t_init)

    Shared_Path = Model_Path + '/balanced_pre_trained_differential_network_parameters.pth'

    if args.data_parallel:
        torch.save(Diff_Autoencoder.module.state_dict(), Shared_Path)
    else:
        torch.save(Diff_Autoencoder.state_dict(), Shared_Path)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
