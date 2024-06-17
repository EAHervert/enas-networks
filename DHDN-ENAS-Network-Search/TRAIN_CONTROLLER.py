import os
import sys
from utilities import dataset
from ENAS_DHDN import SHARED_DHDN
from ENAS_DHDN import CONTROLLER
import datetime
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import visdom
import plotly.graph_objects as go  # Save HTML files for curve analysis

from utilities.functions import display_time
from utilities.utils import CSVLogger, Logger
from ENAS_DHDN.TRAINING_NETWORKS import Train_Controller
from ENAS_DHDN.TRAINING_FUNCTIONS import evaluate_model

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='ENAS_SEARCH_DHDN_CONTROLLER')

parser.add_argument('--output_file', default='Controller_DHDN', type=str)
parser.add_argument('--number', type=int, default=1000)  # Used to generate sampling distribution for Controller
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--cell_copy', default=False, type=lambda x: (str(x).lower() == 'true'))  # Full Or Reduced
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sample_size', type=int, default=-1)  # How many validation samples to evaluate val models
# after a training epoch
parser.add_argument('--validation_samples', type=int, default=8)  # How many samples from validation to train controller at each step of the training epoch
parser.add_argument('--controller_num_aggregate', type=int, default=10)  # Steps in same samples
parser.add_argument('--controller_train_steps', type=int, default=30)  # Total different sample sets
parser.add_argument('--controller_lr', type=float, default=5e-4)  # Controller learning rate
parser.add_argument('--controller_lstm_size', type=int, default=64)  # Size of LSTM (Controller)
parser.add_argument('--controller_lstm_num_layers', type=int, default=1)  # Number of Layers in LSTM (Controller)
parser.add_argument('--load_shared', default=False, type=lambda x: (str(x).lower() == 'true'))  # Load shared model(s)
parser.add_argument('--model_shared_path', default='shared_network_sidd_0032.pth', type=str)
parser.add_argument('--load_controller', default=False, type=lambda x: (str(x).lower() == 'true'))  # Load controller
parser.add_argument('--model_controller_path', default='2023_12_15__16_25_17/controller_parameters.pth', type=str)
parser.add_argument('--device', default='cuda:0', type=str)  # GPU to use
# Put shared network on two devices instead of one
parser.add_argument('--data_parallel', default=True, type=lambda x: (str(x).lower() == 'true'))
# To do outer sums for models
parser.add_argument('--outer_sum', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--kernel_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--down_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--up_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--save_model', default=True, type=lambda x: (str(x).lower() == 'true'))  # Save the final model

args = parser.parse_args()


# Now, let us run all these pieces and have out program train the controller.
def main():
    global args

    current_time = datetime.datetime.now()
    d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

    dir_current = os.getcwd()  # Hyperparameters
    config_path = dir_current + '/configs/config_controller.json'
    config = json.load(open(config_path))
    model_controller_path = '/models/' + args.model_controller_path
    model_shared_path = '/models/' + args.model_shared_path

    Model_Path = 'models/' + d1
    if not os.path.isdir(Model_Path):
        os.mkdir(Model_Path)

    device_0 = torch.device(args.device)  # Define the devices
    # Create the CSV Logger:
    Result_Path = 'results/' + args.output_file + '/' + d1
    if not os.path.isdir(Result_Path):
        os.mkdir(Result_Path)

    # Create the CSV Logger:
    File_Name = Result_Path + '/data.csv'
    Field_Names = ['Loss', 'Loss_Original', 'SSIM', 'SSIM_Original', 'PSNR', 'PSNR_Original']
    CSV_Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

    File_Name_Ctrl = Result_Path + '/controller.log'
    Field_Names_Ctrl = ['Controller_Reward', 'Controller_Accuracy', 'Controller_Loss']
    Ctrl_Logger = CSVLogger(fieldnames=Field_Names_Ctrl, filename=File_Name_Ctrl)

    # This window will show the SSIM and PSNR of the different networks.
    vis = visdom.Visdom(
        server='eng-ml-01.utdallas.edu',
        port=8097,
        use_incoming_socket=False
    )

    vis.env = args.output_file
    vis_window = {
        'Ctrl_Loss_{d1}'.format(d1=d1): None, 'Ctrl_Accuracy_{d1}'.format(d1=d1): None,
        'Ctrl_Reward_{d1}'.format(d1=d1): None
    }

    t_init = time.time()
    np.random.seed(args.seed)
    sys.stdout = Logger(filename=Result_Path + '/log.log')

    print(args)
    print()
    print(config)
    print()

    Shared_Autoencoder = SHARED_DHDN.SharedDHDN(
        k_value=config['Shared']['K_Value'],
        channels=config['Shared']['Channels'],
        outer_sum=args.outer_sum
    )

    if args.load_shared:
        state_dict_shared = torch.load(dir_current + model_shared_path, map_location='cpu')
        Shared_Autoencoder.load_state_dict(state_dict_shared)

    if args.data_parallel:
        Shared_Autoencoder = nn.DataParallel(Shared_Autoencoder, device_ids=[0, 1]).cuda()
    else:
        Shared_Autoencoder = Shared_Autoencoder.to(device_0)

    if args.cell_copy:
        Controller = CONTROLLER.ReducedController(
            k_value=config['Shared']['K_Value'],
            encoder=args.encoder_bool,
            bottleneck=args.bottleneck_bool,
            decoder=args.decoder_bool,
            lstm_size=args.controller_lstm_size,
            lstm_num_layers=args.controller_lstm_num_layers
        )
    else:
        Controller = CONTROLLER.Controller(
            k_value=config['Shared']['K_Value'],
            kernel_bool=args.kernel_bool,
            down_bool=args.down_bool,
            up_bool=args.up_bool,
            lstm_size=args.controller_lstm_size,
            lstm_num_layers=args.controller_lstm_num_layers
        )
    Controller = Controller.to(device_0)

    if args.load_controller:
        state_dict_controller = torch.load(dir_current + model_controller_path, map_location=device_0)
        Controller.load_state_dict(state_dict_controller)

    # We will use the ADAM optimizer for the controller.
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
    Controller_Optimizer = torch.optim.Adam(params=Controller.parameters(),
                                            lr=args.controller_lr,
                                            betas=(0.9, 0.999))

    # Noise Dataset
    path_validation_noisy = dir_current + config['Locations']['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations']['Validation_GT']

    # Todo: Make function that returns these datasets.
    SIDD_validation = dataset.DatasetMAT(mat_noisy_file=path_validation_noisy,
                                         mat_gt_file=path_validation_gt,
                                         device=device_0)

    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                            batch_size=config['Training']['Validation_Batch_Size'],
                                            shuffle=False)
    Shared_Autoencoder.eval()

    Controller.zero_grad()

    # Validation
    loss_batch_val_array = []
    loss_original_batch_val_array = []
    ssim_batch_val_array = []
    ssim_original_batch_val_array = []
    psnr_batch_val_array = []
    psnr_original_batch_val_array = []

    # Modify config to work with Train_Controller
    config['Controller']['Controller_Train_Steps'] = args.controller_train_steps
    config['Controller']['Controller_Num_Aggregate'] = args.controller_num_aggregate
    config['Training']['Validation_Samples'] = args.validation_samples

    for epoch in range(args.epochs):
        controller_dict = Train_Controller(epoch=epoch,
                                           controller=Controller,
                                           shared=Shared_Autoencoder,
                                           controller_optimizer=Controller_Optimizer,
                                           dataloader_sidd_validation=dataloader_sidd_validation,
                                           c_logger=Ctrl_Logger,
                                           config=config,
                                           baseline=None,
                                           device=device_0
                                           )

        vis_window[list(vis_window)[0]] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([controller_dict['Loss']]),
            win=vis_window[list(vis_window)[0]],
            opts=dict(title=list(vis_window)[0], xlabel='Epoch', ylabel='Loss'),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[1]] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([controller_dict['Accuracy']]),
            win=vis_window[list(vis_window)[1]],
            opts=dict(title=list(vis_window)[1], xlabel='Epoch', ylabel='Accuracy'),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[2]] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([controller_dict['Reward']]),
            win=vis_window[list(vis_window)[2]],
            opts=dict(title=list(vis_window)[2], xlabel='Epoch', ylabel='Reward'),
            update='append' if epoch > 0 else None)

        # Controller in eval mode called in evaluate_model
        validation_results = evaluate_model(epoch=epoch,
                                            controller=Controller,
                                            shared=Shared_Autoencoder,
                                            dataloader_sidd_validation=dataloader_sidd_validation,
                                            config=config,
                                            arc_bools=[args.kernel_bool, args.down_bool, args.up_bool],
                                            sample_size=args.sample_size,
                                            device=device_0)

        loss_batch_val_array.append(validation_results['Validation_Loss'])
        loss_original_batch_val_array.append(validation_results['Validation_Loss_Original'])
        ssim_batch_val_array.append(validation_results['Validation_SSIM'])
        ssim_original_batch_val_array.append(validation_results['Validation_SSIM_Original'])
        psnr_batch_val_array.append(validation_results['Validation_PSNR'])
        psnr_original_batch_val_array.append(validation_results['Validation_PSNR_Original'])

        CSV_Logger.writerow({'Loss': validation_results['Validation_Loss'],
                             'Loss_Original': validation_results['Validation_Loss_Original'],
                             'SSIM': validation_results['Validation_SSIM'],
                             'SSIM_Original': validation_results['Validation_SSIM_Original'],
                             'PSNR': validation_results['Validation_PSNR'],
                             'PSNR_Original': validation_results['Validation_PSNR_Original']})

    CSV_Logger.close()
    Ctrl_Logger.close()

    t_final = time.time()
    display_time(t_final - t_init)

    if args.save_model:
        Controller_Path = Model_Path + '/pre_trained_controller_parameters.pth'
        torch.save(Controller.state_dict(), Controller_Path)

    # Saving plots:
    loss_fig = go.Figure(data=go.Scatter(y=loss_batch_val_array, name='Loss_Val'))
    loss_fig.add_trace(go.Scatter(y=loss_original_batch_val_array, name='Loss_Orig_Val'))

    loss_fig.update_layout(title='Loss_' + d1,
                           yaxis_title="Loss",
                           xaxis_title="Epochs")
    loss_fig.write_html(Result_Path + "/loss_plot.html")

    ssim_fig = go.Figure(data=go.Scatter(y=ssim_batch_val_array, name='SSIM_Val'))
    ssim_fig.add_trace(go.Scatter(y=ssim_original_batch_val_array, name='SSIM_Orig_Val'))

    ssim_fig.update_layout(title='SSIM_' + d1,
                           yaxis_title="SSIM",
                           xaxis_title="Epochs")
    ssim_fig.write_html(Result_Path + "/ssim_plot.html")

    psnr_fig = go.Figure(data=go.Scatter(y=psnr_batch_val_array, name='PSNR_Val'))
    psnr_fig.add_trace(go.Scatter(y=psnr_original_batch_val_array, name='PSNR_Orig_Val'))

    psnr_fig.update_layout(title='PSNR_' + d1,
                           yaxis_title="PSNR",
                           xaxis_title="Epochs")
    psnr_fig.write_html(Result_Path + "/psnr_plot.html")


if __name__ == "__main__":
    main()
