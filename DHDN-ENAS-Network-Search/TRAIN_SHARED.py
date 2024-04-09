import os
import sys
from utilities import dataset
from ENAS_DHDN import TRAINING_NETWORKS
from ENAS_DHDN import SHARED_DHDN
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
from utilities.functions import display_time, list_of_ints

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='ENAS_SEARCH_DHDN')

parser.add_argument('--output_file', default='Pre_Train_DHDN', type=str)

# Training:
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--cell_copy', default=False, type=lambda x: (str(x).lower() == 'true'))  # Full Or Reduced
parser.add_argument('--whole_passes', type=int, default=1)
parser.add_argument('--train_passes', type=int, default=-1)
parser.add_argument('--shared_lr', type=float, default=1e-4)  # Shared learning rate
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', default='cuda:0', type=str)  # GPU to use
# Put shared network on two devices instead of one
parser.add_argument('--data_parallel', default=True, type=lambda x: (str(x).lower() == 'true'))
# To do outer sums for models
parser.add_argument('--cutout_images', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--outer_sum', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--fixed_arc', default=[], type=list_of_ints)  # Overrides the controller sample
parser.add_argument('--kernel_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--down_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--up_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--training_csv', default='sidd_np_instances_064_0128.csv', type=str)  # training samples to use
parser.add_argument('--load_shared', default=False, type=lambda x: (str(x).lower() == 'true'))  # Load shared model(s)
parser.add_argument('--model_shared_path', default='shared_network_sidd_0032.pth', type=str)

args = parser.parse_args()


# Now, let us run all these pieces and have out program train the controller.
def main():
    global args

    current_time = datetime.datetime.now()
    d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

    # Hyperparameters
    dir_current = os.getcwd()
    config_path = dir_current + '/configs/config_shared.json'
    config = json.load(open(config_path))
    model_shared_path = '/models/' + args.model_shared_path

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
    File_Name_SA = Result_Path + '/shared_autoencoder.log'
    Field_Names_SA = ['Shared_Loss', 'Shared_Accuracy']
    SA_Logger = CSVLogger(fieldnames=Field_Names_SA, filename=File_Name_SA)

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
        'SN_Loss_{d1}'.format(d1=d1): None, 'SN_SSIM_{d1}'.format(d1=d1): None, 'SN_PSNR_{d1}'.format(d1=d1): None
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

    # We will use ADAM on the child network (Different from Original ENAS paper)
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L213
    # Shared_Autoencoder_Optimizer = torch.optim.Adam(params=Shared_Autoencoder.parameters(),
    #                                                 lr=config['Shared']['Child_lr'],
    #                                                 weight_decay=config['Shared']['Weight_Decay'])

    Shared_Autoencoder_Optimizer = torch.optim.Adam(params=Shared_Autoencoder.parameters(),
                                                    lr=args.shared_lr)

    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L154
    # Use step LR scheduler instead of Cosine Annealing
    Shared_Autoencoder_Scheduler = StepLR(
        optimizer=Shared_Autoencoder_Optimizer,
        step_size=config['Shared']['Step_Size'],
        gamma=config['Shared']['Child_gamma']
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
        print('-' * 120 + '\nUsing Fixed architecture: ' + fixed_arc + '\n' + '-' * 120)

    Controller = None

    # Training
    loss_batch_array = []
    loss_original_batch_array = []
    ssim_batch_array = []
    ssim_original_batch_array = []
    psnr_batch_array = []
    psnr_original_batch_array = []

    for epoch in range(args.epochs):
        training_results = TRAINING_NETWORKS.Train_Shared(epoch=epoch,
                                                          whole_passes=args.whole_passes,
                                                          train_passes=args.train_passes,
                                                          controller=Controller,
                                                          shared=Shared_Autoencoder,
                                                          shared_optimizer=Shared_Autoencoder_Optimizer,
                                                          config=config,
                                                          dataloader_sidd_training=dataloader_sidd_training,
                                                          arc_bools=[args.kernel_bool, args.up_bool,
                                                                     args.down_bool],
                                                          sa_logger=SA_Logger,
                                                          device=device_0,
                                                          fixed_arc=fixed_arc,
                                                          cell_copy=args.cell_copy
                                                          )
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

        loss_batch_array.append(training_results['Loss'])
        loss_original_batch_array.append(training_results['Loss_Original'])
        ssim_batch_array.append(training_results['SSIM'])
        ssim_original_batch_array.append(training_results['SSIM_Original'])
        psnr_batch_array.append(training_results['PSNR'])
        psnr_original_batch_array.append(training_results['PSNR_Original'])

        Shared_Autoencoder_Scheduler.step()

    SA_Logger.close()
    CSV_Logger.close()

    t_final = time.time()

    display_time(t_final - t_init)

    if not args.fixed_arc:
        Shared_Path = Model_Path + '/random_pre_trained_shared_network_parameters.pth'
    else:  # Todo: fix with above
        Shared_Path = Model_Path + '/fixed_arc_parameters.pth'

    if args.data_parallel:
        torch.save(Shared_Autoencoder.module.state_dict(), Shared_Path)
    else:
        torch.save(Shared_Autoencoder.state_dict(), Shared_Path)

    # Saving plots:
    loss_fig = go.Figure(data=go.Scatter(y=loss_batch_array, name='Loss_Train'))
    loss_fig.add_trace(go.Scatter(y=loss_original_batch_array, name='Loss_Orig_Train'))

    loss_fig.update_layout(title='Loss_' + d1,
                           yaxis_title="Loss",
                           xaxis_title="Epochs")
    loss_fig.write_html(Result_Path + "/loss_plot.html")

    ssim_fig = go.Figure(data=go.Scatter(y=ssim_batch_array, name='SSIM_Train'))
    ssim_fig.add_trace(go.Scatter(y=ssim_original_batch_array, name='SSIM_Orig_Train'))

    ssim_fig.update_layout(title='SSIM_' + d1,
                           yaxis_title="SSIM",
                           xaxis_title="Epochs")
    ssim_fig.write_html(Result_Path + "/ssim_plot.html")

    psnr_fig = go.Figure(data=go.Scatter(y=psnr_batch_array, name='PSNR_Train'))
    psnr_fig.add_trace(go.Scatter(y=psnr_original_batch_array, name='PSNR_Orig_Train'))

    psnr_fig.update_layout(title='PSNR_' + d1,
                           yaxis_title="PSNR",
                           xaxis_title="Epochs")
    psnr_fig.write_html(Result_Path + "/psnr_plot.html")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
