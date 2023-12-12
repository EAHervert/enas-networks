import os
import sys
from utilities import dataset
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

from utilities.utils import CSVLogger, Logger
from utilities.functions import SSIM, PSNR, generate_loggers, display_time, random_architecture_generation

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='RANDOM_SEARCH_DHDN')
parser.add_argument('--Output_File', default='RANDOM_DHDN', type=str)

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
parser.add_argument('--training_csv', default='sidd_np_instances_064_128.csv', type=str)  # training samples to use
parser.add_argument('--load_shared', default=False, type=bool)  # Load shared model(s)
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

    model_shared_path = '/models/' + args.model_shared_path

    # Let us create the loggers to keep track of the Losses, Accuracies, and Rewards.
    File_Name_SA = Result_Path + '/shared_autoencoder.log'
    Field_Names_SA = ['Shared_Loss', 'Shared_Accuracy']
    SA_Logger = CSVLogger(fieldnames=Field_Names_SA, filename=File_Name_SA)

    # Define the Loss and evaluation metrics:
    loss_0 = nn.L1Loss().to(device_0)
    MSE = nn.MSELoss().to(device_0)

    # Now, let us define our loggers:
    loggers0 = generate_loggers()

    # Training Batches
    loss_batch, loss_original_batch, ssim_batch, ssim_original_batch, psnr_batch, psnr_original_batch = loggers0[0]

    # Validation Batches
    loss_batch_val, loss_original_batch_val, ssim_batch_val, ssim_original_batch_val = loggers0[1][0:4]
    psnr_batch_val, psnr_original_batch_val = loggers0[1][4:]

    # This window will show the SSIM and PSNR of the different networks.
    vis = visdom.Visdom(
        server='eng-ml-01.utdallas.edu',
        port=8097,
        use_incoming_socket=False
    )

    vis.env = config['Locations']['Output_File_Random']
    vis_window = {'SSIM_{date}'.format(date=d1): None, 'PSNR_{date}'.format(date=d1): None}

    t_init = time.time()
    np.random.seed(args.Seed)

    if config['CUDA']['Device0']:
        torch.cuda.manual_seed(args.Seed)

    if args.Fixed_Arc:
        sys.stdout = Logger(filename=Result_Path + '/log_fixed.log')
    else:
        sys.stdout = Logger(filename=Result_Path + '/log.log')

    # Create the CSV Logger:
    File_Name = Result_Path + '/data.csv'
    Field_Names = ['Loss_Batch', 'Loss_Val', 'Loss_Original_Train', 'Loss_Original_Val',
                   'SSIM_Batch', 'SSIM_Val', 'SSIM_Original_Train', 'SSIM_Original_Val',
                   'PSNR_Batch', 'PSNR_Val', 'PSNR_Original_Train', 'PSNR_Original_Val']
    Logger_ = CSVLogger(fieldnames=Field_Names, filename=File_Name)

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

    SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing())
    SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy, mat_gt_file=path_validation_gt)

    dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=config['Training']['Train_Batch_Size'],
                                          shuffle=True, num_workers=16)

    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                            batch_size=config['Training']['Validation_Batch_Size'],
                                            shuffle=False, num_workers=8)

    for epoch in range(args.Epochs):
        for i_batch, sample_batch in enumerate(dataloader_sidd_training):
            architecture = random_architecture_generation(k_value=config['Shared']['K_Value'],
                                                          kernel_bool=args.Kernel_Bool,
                                                          down_bool=args.Down_Bool,
                                                          up_bool=args.Up_Bool)
            x = sample_batch['NOISY']
            y0 = Shared_Autoencoder(x.to(device_0), architecture)
            t = sample_batch['GT']

            loss_value_0 = loss_0(y0, t.to(device_0))
            loss_batch.update(loss_value_0.item())

            # Calculate values not needing to be backpropagated
            with torch.no_grad():
                loss_original_batch.update(loss_0(x.to(device_0), t.to(device_0)).item())

                ssim_batch.update(SSIM(y0, t.to(device_0)).item())
                ssim_original_batch.update(SSIM(x, t).item())

                psnr_batch.update(PSNR(MSE(y0, t.to(device_0))).item())
                psnr_original_batch.update(PSNR(MSE(x.to(device_0), t.to(device_0))).item())

            # Backpropagate to train model
            Shared_Autoencoder_Optimizer.zero_grad()
            loss_value_0.backward()
            nn.utils.clip_grad_norm_(Shared_Autoencoder.parameters(), config['Shared']['Child_Grad_Bound'])
            Shared_Autoencoder_Optimizer.step()

            if i_batch % 100 == 0:
                Display_Loss = "Loss_SHARED: %.6f" % loss_batch.val + "\tLoss_Original: %.6f" % loss_original_batch.val
                Display_SSIM = "SSIM_SHARED: %.6f" % ssim_batch.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
                Display_PSNR = "PSNR_SHARED: %.6f" % psnr_batch.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

                print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
                print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

            # Free up space in GPU
            del x, y0, t

        Display_Loss = "Loss_SHARED: %.6f" % loss_batch.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
        Display_SSIM = "SSIM_SHARED: %.6f" % ssim_batch.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
        Display_PSNR = "PSNR_SHARED: %.6f" % psnr_batch.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

        print('\n' + '-' * 160)
        print("Training Data for Epoch: ", epoch)
        print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
        print('-' * 160 + '\n')

        for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
            architecture = random_architecture_generation(k_value=config['Shared']['K_Value'],
                                                          kernel_bool=args.Kernel_Bool,
                                                          down_bool=args.Down_Bool,
                                                          up_bool=args.Up_Bool)
            if i_validation % 10 == 0:
                print('Epoch:', epoch, '\tStep:', i_validation, '\tArchitecture:', architecture)
            x_v = validation_batch['NOISY']
            t_v = validation_batch['GT']
            with torch.no_grad():
                y_v0 = Shared_Autoencoder(x_v.to(device_0), architecture)

                loss_batch_val.update(loss_0(y_v0, t_v.to(device_0)).item())
                loss_original_batch_val.update(loss_0(x_v.to(device_0), t_v.to(device_0)).item())

                ssim_batch_val.update(SSIM(y_v0, t_v.to(device_0)).item())
                ssim_original_batch_val.update(SSIM(x_v, t_v).item())

                psnr_batch_val.update(PSNR(MSE(y_v0, t_v.to(device_0))).item())
                psnr_original_batch_val.update(PSNR(MSE(x_v.to(device_0), t_v.to(device_0))).item())

            # Free up space in GPU
            del x_v, y_v0, t_v

        Display_Loss = "Loss_SHARED: %.6f" % loss_batch_val.val + "\tLoss_Original: %.6f" % loss_original_batch_val.val
        Display_SSIM = "SSIM_SHARED: %.6f" % ssim_batch_val.avg + "\tSSIM_Original: %.6f" % ssim_original_batch_val.avg
        Display_PSNR = "PSNR_SHARED: %.6f" % psnr_batch_val.avg + "\tPSNR_Original: %.6f" % psnr_original_batch_val.avg

        print('\n' + '-' * 160)
        print("Validation Data for Epoch: ", epoch)
        print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
        print('-' * 160 + '\n')

        Logger_.writerow({
            'Loss_Batch': loss_batch.avg,
            'Loss_Val': loss_batch_val.avg,
            'Loss_Original_Train': loss_original_batch.avg,
            'Loss_Original_Val': loss_original_batch_val.avg,
            'SSIM_Batch': ssim_batch.avg,
            'SSIM_Val': ssim_batch_val.avg,
            'SSIM_Original_Train': ssim_original_batch.avg,
            'SSIM_Original_Val': ssim_original_batch_val.avg,
            'PSNR_Batch': psnr_batch.avg,
            'PSNR_Val': psnr_batch_val.avg,
            'PSNR_Original_Train': psnr_original_batch.avg,
            'PSNR_Original_Val': psnr_original_batch_val.avg
        })

        SA_Logger.writerow({'Shared_Loss': loss_batch.avg, 'Shared_Accuracy': ssim_batch.avg})

        Legend = ['Shared_Train', 'Orig_Train', 'Shared_Val', 'Orig_Val']

        vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([ssim_batch.avg, ssim_original_batch.avg, ssim_batch_val.avg,
                               ssim_original_batch_val.avg]),
            win=vis_window['SSIM_{date}'.format(date=d1)],
            opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([psnr_batch.avg, psnr_original_batch.avg, psnr_batch_val.avg,
                               psnr_original_batch_val.avg]),
            win=vis_window['PSNR_{date}'.format(date=d1)],
            opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
            update='append' if epoch > 0 else None)

    loss_batch.reset()
    loss_original_batch.reset()
    ssim_batch.reset()
    ssim_original_batch.reset()
    psnr_batch.reset()
    psnr_original_batch.reset()
    loss_batch_val.reset()
    loss_original_batch_val.reset()
    ssim_batch_val.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val.reset()
    psnr_original_batch_val.reset()

    Shared_Autoencoder_Scheduler.step()

    SA_Logger.close()

    t_final = time.time()
    display_time(t_final - t_init)

    # Save the parameters:
    Shared_Path = Model_Path + '/RANDOM__shared_network_parameters.pth'
    torch.save(Shared_Autoencoder.state_dict(), Shared_Path)


if __name__ == "__main__":
    main()
