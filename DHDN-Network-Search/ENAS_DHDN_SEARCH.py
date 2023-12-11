import os
import sys
from utilities import dataset
from ENAS_DHDN import TRAINING_NETWORKS
from ENAS_DHDN import SHARED_DHDN
from ENAS_DHDN import CONTROLLER
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
from utilities.functions import display_time

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='ENAS_SEARCH_DHDN')

parser.add_argument('--Output_File', default='ENAS_DHDN', type=str)
# parser.add_argument('--Resume', default='', type=str) # If we are resuming progress from a checkpoint,
# we can place the checkpoint here.


# Training:
parser.add_argument('--Epochs', type=int, default=30)
parser.add_argument('--Log_Every', type=int, default=10)
parser.add_argument('--Eval_Every_Epoch', type=int, default=1)
parser.add_argument('--Seed', type=int, default=0)
parser.add_argument('--outer_sum', default=False, type=bool)  # To do outer sums for models
parser.add_argument('--Fixed_Arc', action='store_true', default=False)
parser.add_argument('--Kernel_Bool', type=bool, default=True)
parser.add_argument('--Down_Bool', type=bool, default=True)
parser.add_argument('--Up_Bool', type=bool, default=True)
parser.add_argument('--Controller_Train_Every', type=int, default=1)
parser.add_argument('--training_csv', default='sidd_np_instances_064_128.csv', type=str)  # training samples to use
parser.add_argument('--load_shared', default=False, type=bool)  # Load shared model(s)
parser.add_argument('--load_controller', default=False, type=bool)  # Load Controller
parser.add_argument('--model_controller_path', default='2023_12_02__22_49_09/controller_parameters.pth', type=str)
parser.add_argument('--model_shared_path', default='2023_12_02__22_49_09/shared_network_parameters.pth', type=str)

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
    device_0 = torch.device(config['CUDA']['Device0'])

    if not os.path.isdir('Logs_DHDN/'):
        os.mkdir('Logs_DHDN/')

    Result_Path = 'Logs_DHDN/' + args.Output_File + '/' + d1
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

    File_Name_C = Result_Path + '/controller.log'
    Field_Names_C = ['Controller_Reward', 'Controller_Accuracy', 'Controller_Loss']
    C_Logger = CSVLogger(fieldnames=Field_Names_C, filename=File_Name_C)

    Network_Logger = [SA_Logger, C_Logger]

    # This window will show the SSIM and PSNR of the different networks.
    vis = visdom.Visdom(
        server='eng-ml-01.utdallas.edu',
        port=8097,
        use_incoming_socket=False
    )

    vis.env = config['Locations']['Output_File']
    vis_window = {
        'SN_Loss_{d1}'.format(d1=d1): None, 'SN_SSIM_{d1}'.format(d1=d1): None,
        'SN_PSNR_{d1}'.format(d1=d1): None, 'Ctrl_Loss_{d1}'.format(d1=d1): None,
        'Ctrl_Accuracy_{d1}'.format(d1=d1): None, 'Ctrl_Reward_{d1}'.format(d1=d1): None
    }

    t_init = time.time()

    np.random.seed(args.Seed)

    if config['CUDA']['Device0']:
        torch.cuda.manual_seed(args.Seed)

    if args.Fixed_Arc:
        sys.stdout = Logger(filename=Result_Path + '/log_fixed.log')
    else:
        sys.stdout = Logger(filename=Result_Path + '/log.log')

    print(args)
    print()
    print(config)
    print()

    Controller = CONTROLLER.Controller(
        k_value=config['Shared']['K_Value'],
        kernel_bool=args.Kernel_Bool,
        down_bool=args.Down_Bool,
        up_bool=args.Up_Bool,
        LSTM_size=config['Controller']['Controller_LSTM_Size'],
        LSTM_num_layers=config['Controller']['Controller_LSTM_Num_Layers']
    )

    Controller = Controller.to(device_0)

    if args.load_controller:
        state_dict_controller = torch.load(dir_current + model_controller_path, map_location=device_0)
        Controller.load_state_dict(state_dict_controller)

    Shared_Autoencoder = SHARED_DHDN.SharedDHDN(
        k_value=config['Shared']['K_Value'],
        channels=config['Shared']['Channels'],
        outer_sum=args.outer_sum
    )

    if config['CUDA']['DataParallel']:
        Shared_Autoencoder = nn.DataParallel(Shared_Autoencoder, device_ids=[0, 1]).cuda()
    else:
        Shared_Autoencoder = Shared_Autoencoder.to(device_0)

    if args.load_shared:
        state_dict_shared = torch.load(dir_current + model_shared_path, map_location=device_0)
        Shared_Autoencoder.load_state_dict(state_dict_shared)

    # We will use the ADAM optimizer for the controller.
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
    Controller_Optimizer = torch.optim.Adam(params=Controller.parameters(),
                                            lr=config['Controller']['Controller_lr'],
                                            betas=(0.9, 0.999))

    # We will use ADAM on the child network (Different from Original ENAS paper)
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L213
    Shared_Autoencoder_Optimizer = torch.optim.Adam(params=Shared_Autoencoder.parameters(),
                                                    lr=config['Shared']['Child_lr'],
                                                    weight_decay=config['Shared']['Weight_Decay'])

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

    SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing())
    SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy, mat_gt_file=path_validation_gt)

    dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=config['Training']['Train_Batch_Size'],
                                          shuffle=True, num_workers=16)

    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                            batch_size=config['Training']['Validation_Batch_Size'],
                                            shuffle=False, num_workers=8)

    # TODO: For resuming a run (will modify later)
    '''    
    if args.Resume:
        if os.path.isfile(args.Resume):
            print("Loading Checkpoint '{}'".format(args.Resume))
            Checkpoint = torch.load(args.Resume)
            Start_Epoch = Checkpoint['Epoch']
            # args = checkpoint['args']
            Shared_Autoencoder.load_state_dict(Checkpoint['Shared_Autoencoder_State_Dict'])
            Controller.load_state_dict(Checkpoint['Controller_State_Dict'])
            Shared_Autoencoder_Optimizer.load_state_dict(Checkpoint['Shared_Autoencoder_Optimizer'])
            Controller_Optimizer.load_state_dict(Checkpoint['Controller_Optimizer'])
            # shared_cnn_scheduler.optimizer = shared_cnn_optimizer  # Not sure if this actually works
            print("Loaded Checkpoint '{}' (Epoch {})"
                  .format(args.Resume, Checkpoint['Epoch']))
        else:
            raise ValueError("No checkpoint found at '{}'".format(args.Resume))
    else:
        Start_Epoch = 0
    '''
    start_epoch = 0

    if not args.Fixed_Arc:
        TRAINING_NETWORKS.Train_ENAS(
            start_epoch=start_epoch,
            num_epochs=args.Epochs,
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
            log_every=args.Log_Every,
            eval_every_epoch=args.Eval_Every_Epoch,
            device=device_0,
        )
    else:
        print("Exiting:")
        exit()
        '''
        assert args.resume != '', 'A pretrained model should be used when training a fixed architecture.'
        Train_Fixed(
            Start_Epoch = Start_Epoch,
            Num_Epochs = args.Num_Epochs,
            Size = args.Network_Size,
            Num_Branches = args.Child_Num_Branches,
            channels = args.channels,
            Controller = Controller,
            Shared_Autoencoder = Shared_Autoencoder,
            Dataloader = Dataloader,
            Batch_Size = args.Batch_Size,
            Eval_Every_Epoch = args.Eval_Every_Epoch,
            Log_Every = args.Log_Every,
            Child_lr_Max = args.Child_lr_Max,
            Child_l2_Reg = args.Child_l2_Reg,
            Child_Grad_Bound = args.Child_Grad_Bound,
            Device = Device,
            Output_Filename = args.Output_Filename,
            args = None
            )
        '''

    SA_Logger.close()
    C_Logger.close()

    t_final = time.time()

    display_time(t_final - t_init)

    # Save the parameters:
    Shared_Path = Model_Path + '/shared_network_parameters.pth'
    Controller_Path = Model_Path + '/controller_parameters.pth'

    torch.save(Shared_Autoencoder.state_dict(), Shared_Path)
    torch.save(Controller.state_dict(), Controller_Path)


if __name__ == "__main__":
    main()
