import os
import sys
from utilities import dataset
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
from utilities.functions import display_time, drop_weights
from DNAS_DHDN import DIFFERENTIABLE_DHDN

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='DNAS_SEARCH_DHDN')

parser.add_argument('--output_file', default='DNAS_DHDN', type=str)

# Training:
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--pre_train_epochs', type=int, default=-1)  # Randomly pre-training model
parser.add_argument('--passes', type=int, default=1)
parser.add_argument('--sample_size', type=int, default=-1)  # How many samples from validation to evaluate
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', default='cuda:0', type=str)  # GPU to use
# Put shared network on two devices instead of one
parser.add_argument('--data_parallel', default=True, type=lambda x: (str(x).lower() == 'true'))
# To do outer sums for models
parser.add_argument('--outer_sum', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--fixed_arc', action='store_true', default=False)
parser.add_argument('--kernel_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--down_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--up_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--training_csv', default='sidd_np_instances_064_0128.csv', type=str)  # training samples to use
parser.add_argument('--load_shared', default=False, type=lambda x: (str(x).lower() == 'true'))  # Load shared model(s)
# Use Controller, False for random generation
parser.add_argument('--use_controller', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--load_controller', default=False, type=lambda x: (str(x).lower() == 'true'))  # Load Controller
parser.add_argument('--pre_train_controller', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--model_controller_path', default='2023_12_15__16_25_17/controller_parameters.pth', type=str)
parser.add_argument('--model_shared_path', default='2023_12_15__16_25_17/shared_network_parameters.pth', type=str)

args = parser.parse_args()


# Now, let us run all these pieces and have out program train the controller.
def main():
    global args

    current_time = datetime.datetime.now()
    d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

    # Hyperparameters
    dir_current = os.getcwd()
    config_path = dir_current + '/configs/config_search.json'
    config = json.load(open(config_path))
    if not os.path.exists(dir_current + '/models/'):
        os.makedirs(dir_current + '/models/')

    # Define the devices:
    device_0 = torch.device(args.device)

    if not os.path.isdir('Logs_DHDN/'):
        os.mkdir('Logs_DHDN/')

    Result_Path = 'Logs_DHDN/' + args.output_file + '/' + d1
    if not os.path.isdir(Result_Path):
        os.mkdir(Result_Path)

    Model_Path = 'models/' + d1
    if not os.path.isdir(Model_Path):
        os.mkdir(Model_Path)

    model_controller_path = '/models/' + args.model_controller_path
    model_shared_path = '/models/' + args.model_shared_path

    # Let us create the loggers to keep track of the Losses, Accuracies, and Rewards.
    File_Name_SA = Result_Path + '/shared_autoencoder.log'
    Field_Names_SA = ['Shared_Loss', 'Shared_Accuracy']
    SA_Logger = CSVLogger(fieldnames=Field_Names_SA, filename=File_Name_SA)

    # Create the CSV Logger:
    File_Name = Result_Path + '/data.csv'
    Field_Names = ['Loss_Batch', 'Loss_Val', 'Loss_Original_Train', 'Loss_Original_Val',
                   'SSIM_Batch', 'SSIM_Val', 'SSIM_Original_Train', 'SSIM_Original_Val',
                   'PSNR_Batch', 'PSNR_Val', 'PSNR_Original_Train', 'PSNR_Original_Val']
    CSV_Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

    Network_Logger = [SA_Logger, CSV_Logger]

    # This window will show the SSIM and PSNR of the different networks.
    vis = visdom.Visdom(
        server='eng-ml-01.utdallas.edu',
        port=8097,
        use_incoming_socket=False
    )

    vis.env = args.output_file
    vis_window = {
        'SN_Loss_{d1}'.format(d1=d1): None, 'SN_SSIM_{d1}'.format(d1=d1): None,
        'SN_PSNR_{d1}'.format(d1=d1): None, 'Ctrl_Loss_{d1}'.format(d1=d1): None,
        'Ctrl_Accuracy_{d1}'.format(d1=d1): None, 'Ctrl_Reward_{d1}'.format(d1=d1): None
    }

    t_init = time.time()
    np.random.seed(args.seed)

    if config['CUDA']['Device0']:
        torch.cuda.manual_seed(args.seed)

    if args.fixed_arc:
        sys.stdout = Logger(filename=Result_Path + '/log_fixed.log')
    else:
        sys.stdout = Logger(filename=Result_Path + '/log.log')

    print(args)
    print()
    print(config)
    print()

    Differentiable_Autoencoder = DIFFERENTIABLE_DHDN.DifferentiableDHDN(
        k_value=config['Shared']['K_Value'],
        channels=config['Shared']['Channels'],
        outer_sum=args.outer_sum
    )

    if args.data_parallel:
        Differentiable_Autoencoder = nn.DataParallel(Differentiable_Autoencoder, device_ids=[0, 1]).cuda()
    else:
        Differentiable_Autoencoder = Differentiable_Autoencoder.to(device_0)

    if args.load_shared:
        state_dict_shared = torch.load(dir_current + model_shared_path, map_location=device_0)
        if args.drop > 0:
            state_dict_shared = drop_weights(state_dict_shared, p=args.drop, device=device_0)
        Differentiable_Autoencoder.load_state_dict(state_dict_shared)

    # We will use ADAM on the child network (Different from Original ENAS paper)
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L213
    # Shared_Autoencoder_Optimizer = torch.optim.Adam(params=Shared_Autoencoder.parameters(),
    #                                                 lr=config['Shared']['Child_lr'],
    #                                                 weight_decay=config['Shared']['Weight_Decay'])

    Shared_Autoencoder_Optimizer = torch.optim.Adam(params=Differentiable_Autoencoder.parameters(),
                                                    lr=config['Shared']['Child_lr'])

    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L154
    # Use step LR scheduler instead of Cosine Annealing
    Shared_Autoencoder_Scheduler = StepLR(
        optimizer=Shared_Autoencoder_Optimizer,
        step_size=config['Shared']['Step_Size'],
        gamma=config['Shared']['Child_gamma']
    )

    # Noise Dataset
    path_training = dir_current + '/instances/' + args.training_csv
    path_validation_noisy = dir_current + config['Locations']['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations']['Validation_GT']

    # Todo: Make function that returns these datasets.
    SIDD_training = dataset.DatasetNoise(csv_file=path_training,
                                         transform=dataset.RandomProcessing(),
                                         device=device_0)
    SIDD_validation = dataset.DatasetMAT(mat_noisy_file=path_validation_noisy,
                                         mat_gt_file=path_validation_gt,
                                         device=device_0)

    dataloader_sidd_training = DataLoader(dataset=SIDD_training,
                                          batch_size=config['Training']['Train_Batch_Size'],
                                          shuffle=True)

    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                            batch_size=config['Training']['Validation_Batch_Size'],
                                            shuffle=False)

    if not args.fixed_arc:
        TRAINING_NETWORKS.Train_ENAS(
            start_epoch=0,
            pre_train_epochs=args.pre_train_epochs,
            num_epochs=args.epochs,
            passes=args.passes,
            controller=Controller,
            shared=Shared_Autoencoder,
            shared_optimizer=Shared_Autoencoder_Optimizer,
            controller_optimizer=Controller_Optimizer,
            shared_scheduler=Shared_Autoencoder_Scheduler,
            dataloader_sidd_training=dataloader_sidd_training,
            dataloader_sidd_validation=dataloader_sidd_validation,
            logger=Network_Logger,
            vis=vis,
            vis_window=vis_window,
            config=config,
            arc_bools=[args.kernel_bool, args.down_bool, args.up_bool],
            sample_size=args.sample_size,
            device=device_0,
            pre_train_controller=args.pre_train_controller
        )
    else:  # Todo: add the fixed_arc training optionality
        print("Exiting:")
        exit()

    SA_Logger.close()
    CSV_Logger.close()

    t_final = time.time()

    display_time(t_final - t_init)

    if not args.fixed_arc:
        Shared_Path = Model_Path + '/random__shared_network_parameters.pth'
    else:  # Todo: fix with above
        Shared_Path = Model_Path + '/{arc}__parameters.pth'.format(arc=args.fixed_arc)

    torch.save(Differentiable_Autoencoder.state_dict(), Shared_Path)


if __name__ == "__main__":
    main()
