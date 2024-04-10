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
import plotly.graph_objects as go  # Save HTML files for curve analysis

from utilities.utils import CSVLogger, Logger
from utilities.functions import display_time, drop_weights

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='ENAS_SEARCH_DHDN')

parser.add_argument('--output_file', default='ENAS_DHDN', type=str)

# Training:
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--pre_train_epochs', type=int, default=-1)  # Randomly pre-training model
parser.add_argument('--shared_lr', type=float, default=1e-4)  # Learning rate override config TODO: Fix
parser.add_argument('--cell_copy', default=False, type=lambda x: (str(x).lower() == 'true'))  # Full Or Reduced
parser.add_argument('--whole_passes', type=int, default=1)
parser.add_argument('--train_passes', type=int, default=-1)
parser.add_argument('--sample_size', type=int, default=-1)  # How many samples from validation to evaluate
parser.add_argument('--cutout_images', default=False, type=lambda x: (str(x).lower() == 'true'))  # Image cutout in training
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', default='cuda:0', type=str)  # GPU to use
# Put shared network on two devices instead of one
parser.add_argument('--data_parallel', default=True, type=lambda x: (str(x).lower() == 'true'))
# To do outer sums for models
parser.add_argument('--outer_sum', default=False, type=lambda x: (str(x).lower() == 'true'))
# Full Architecture Search
parser.add_argument('--kernel_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--down_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--up_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
# Reduced Cell Search
parser.add_argument('--encoder_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--bottleneck_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--decoder_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
# Training Models and data
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
    config['Shared']['Child_lr'] = args.shared_lr  # Todo: fix
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

    if args.use_controller:
        File_Name_Ctrl = Result_Path + '/controller.log'
        Field_Names_Ctrl = ['Controller_Reward', 'Controller_Accuracy', 'Controller_Loss']
        Ctrl_Logger = CSVLogger(fieldnames=Field_Names_Ctrl, filename=File_Name_Ctrl)
    else:
        Ctrl_Logger = None

    # Create the CSV Logger:
    File_Name = Result_Path + '/data.csv'
    Field_Names = ['Loss_Batch', 'Loss_Val', 'Loss_Original_Train', 'Loss_Original_Val',
                   'SSIM_Batch', 'SSIM_Val', 'SSIM_Original_Train', 'SSIM_Original_Val',
                   'PSNR_Batch', 'PSNR_Val', 'PSNR_Original_Train', 'PSNR_Original_Val']
    CSV_Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

    Network_Logger = [SA_Logger, Ctrl_Logger, CSV_Logger]

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

    sys.stdout = Logger(filename=Result_Path + '/log.log')

    print(args)
    print()
    print(config)
    print()

    if args.use_controller:
        if args.cell_copy:
            Controller = CONTROLLER.ReducedController(
                k_value=config['Shared']['K_Value'],
                encoder=args.encoder_bool,
                bottleneck=args.bottleneck_bool,
                decoder=args.decoder_bool,
                lstm_size=config['Controller']['Controller_LSTM_Size'],
                lstm_num_layers=config['Controller']['Controller_LSTM_Num_Layers']
            )
        else:
            Controller = CONTROLLER.Controller(
                k_value=config['Shared']['K_Value'],
                kernel_bool=args.kernel_bool,
                down_bool=args.down_bool,
                up_bool=args.up_bool,
                lstm_size=config['Controller']['Controller_LSTM_Size'],
                lstm_num_layers=config['Controller']['Controller_LSTM_Num_Layers']
            )
        Controller = Controller.to(device_0)

        if args.load_controller:
            state_dict_controller = torch.load(dir_current + model_controller_path, map_location=device_0)
            if args.drop > 0:
                state_dict_controller = drop_weights(state_dict_controller, p=args.drop, device=device_0)

            Controller.load_state_dict(state_dict_controller)

        # We will use the ADAM optimizer for the controller.
        # https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
        Controller_Optimizer = torch.optim.Adam(params=Controller.parameters(),
                                                lr=config['Controller']['Controller_lr'],
                                                betas=(0.9, 0.999))
    else:
        print('-' * 120 + '\nUsing randomly generated architectures.' + '\n' + '-' * 120)
        Controller = None
        Controller_Optimizer = None

    Shared_Autoencoder = SHARED_DHDN.SharedDHDN(
        k_value=config['Shared']['K_Value'],
        channels=config['Shared']['Channels'],
        outer_sum=args.outer_sum
    )

    if args.data_parallel:
        Shared_Autoencoder = nn.DataParallel(Shared_Autoencoder, device_ids=[0, 1]).cuda()
    else:
        Shared_Autoencoder = Shared_Autoencoder.to(device_0)

    if args.load_shared:
        state_dict_shared = torch.load(dir_current + model_shared_path, map_location=device_0)
        if args.drop > 0:
            state_dict_shared = drop_weights(state_dict_shared, p=args.drop, device=device_0)
        Shared_Autoencoder.load_state_dict(state_dict_shared)

    # We will use ADAM on the child network (Different from Original ENAS paper)
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L213
    # Shared_Autoencoder_Optimizer = torch.optim.Adam(params=Shared_Autoencoder.parameters(),
    #                                                 lr=config['Shared']['Child_lr'],
    #                                                 weight_decay=config['Shared']['Weight_Decay'])

    Shared_Autoencoder_Optimizer = torch.optim.Adam(params=Shared_Autoencoder.parameters(),
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
                                         transform=dataset.RandomProcessing(cutout_images=args.cutout_images),
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

    arc_bools = [args.kernel_bool, args.down_bool, args.up_bool] if not args.cell_copy else None
    results = TRAINING_NETWORKS.Train_ENAS(
        start_epoch=0,
        pre_train_epochs=args.pre_train_epochs,
        num_epochs=args.epochs,
        whole_passes=args.whole_passes,
        train_passes=args.train_passes,
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
        arc_bools=arc_bools,
        sample_size=args.sample_size,
        device=device_0,
        pre_train_controller=args.pre_train_controller,
        cell_copy=args.cell_copy
    )

    SA_Logger.close()
    CSV_Logger.close()

    t_final = time.time()

    display_time(t_final - t_init)

    # Save the parameters:
    if args.use_controller:
        Controller_Path = Model_Path + '/controller_parameters.pth'
        Ctrl_Logger.close()
        torch.save(Controller.state_dict(), Controller_Path)

    Shared_Path = Model_Path + '/shared_network_parameters.pth'
    torch.save(Shared_Autoencoder.state_dict(), Shared_Path)

    # Saving plots:
    loss_fig = go.Figure(data=go.Scatter(y=results['Loss_Batch'], name='Loss_Train'))
    loss_fig.add_trace(go.Scatter(y=results['Loss_Original_Train'], name='Loss_Orig_Train'))
    loss_fig.add_trace(go.Scatter(y=results['Loss_Val'], name='Loss_Val'))
    loss_fig.add_trace(go.Scatter(y=results['Loss_Original_Val'], name='Loss_Orig_Val'))

    loss_fig.update_layout(title='Loss_ENAS',
                           yaxis_title="Loss",
                           xaxis_title="Epochs")
    loss_fig.write_html(Result_Path + "/loss_plot.html")

    ssim_fig = go.Figure(data=go.Scatter(y=results['SSIM_Batch'], name='SSIM_Train'))
    ssim_fig.add_trace(go.Scatter(y=results['SSIM_Original_Train'], name='SSIM_Orig_Train'))
    ssim_fig.add_trace(go.Scatter(y=results['SSIM_Val'], name='SSIM_Val'))
    ssim_fig.add_trace(go.Scatter(y=results['SSIM_Original_Val'], name='SSIM_Orig_Val'))

    ssim_fig.update_layout(title='SSIM_ENAS',
                           yaxis_title="SSIM",
                           xaxis_title="Epochs")
    ssim_fig.write_html(Result_Path + "/ssim_plot.html")

    psnr_fig = go.Figure(data=go.Scatter(y=results['PSNR_Batch'], name='PSNR_Train'))
    psnr_fig.add_trace(go.Scatter(y=results['PSNR_Original_Train'], name='PSNR_Orig_Train'))
    psnr_fig.add_trace(go.Scatter(y=results['PSNR_Val'], name='PSNR_Val'))
    psnr_fig.add_trace(go.Scatter(y=results['PSNR_Original_Val'], name='PSNR_Orig_Val'))

    psnr_fig.update_layout(title='PSNR_ENAS',
                           yaxis_title="PSNR",
                           xaxis_title="Epochs")
    psnr_fig.write_html(Result_Path + "/psnr_plot.html")


if __name__ == "__main__":
    main()
