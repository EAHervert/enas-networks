import os
import sys
from utilities import dataset
from ENAS_DHDN import SHARED_DHDN as DHDN
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import visdom
import argparse

from utilities.utils import CSVLogger, Logger
from utilities.functions import SSIM, PSNR, generate_loggers, drop_weights, gaussian_add_weights, clip_weights

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

# Parser
parser = argparse.ArgumentParser(
    prog='DHDN_Compare_Kernels',
    description='Compares 3 Kernel arrangements',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--gaussian', default='-1', type=float)  # Gaussian noise addition for model weight # initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_compare_kernels.json'
config = json.load(open(config_path))
if not os.path.exists(dir_current + '/models/compare_kernels/'):
    os.makedirs(dir_current + '/models/compare_kernels/')

# Noise Dataset
if args.noise == 'SIDD':
    path_training = dir_current + config['Locations']['Training_File']
    path_validation_noisy = dir_current + config['Locations']['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations']['Validation_GT']
    Result_Path = dir_current + '/SIDD/{date}/'.format(date=d1)
elif args.noise in ['GAUSSIAN_10', 'GAUSSIAN_25', 'GAUSSIAN_50', 'RAIN', 'SALT_PEPPER', 'MIXED']:
    path_training = dir_current + '/instances/davis_np_instances_128.csv'
    path_validation = dir_current + '/instances/davis_np_instances_256.csv'
    Result_Path = dir_current + '/{noise}/{date}/'.format(noise=args.noise, date=d1)
else:
    print('Incorrect Noise Selection!')
    exit()

if not os.path.isdir(Result_Path):
    os.mkdir(Result_Path)

if not os.path.isdir(Result_Path + '/' + config['Locations']['Output_File']):
    os.mkdir(Result_Path + '/' + config['Locations']['Output_File'])
sys.stdout = Logger(Result_Path + '/' + config['Locations']['Output_File'] + '/log.log')

# Create the CSV Logger:
File_Name = Result_Path + '/' + config['Locations']['Output_File'] + '/data.csv'
Field_Names = ['Loss_Batch_3x3', 'Loss_Batch_5x5', 'Loss_Batch_RAN', 'Loss_Original_Train',
               'Loss_Val_3x3', 'Loss_Val_5x5', 'Loss_Val_RAN', 'Loss_Original_Val',
               'SSIM_Batch_3x3', 'SSIM_Batch_5x5', 'SSIM_Batch_RAN', 'SSIM_Original_Train',
               'SSIM_Val_3x3', 'SSIM_Val_5x5', 'SSIM_Val_RAN', 'SSIM_Original_Val',
               'PSNR_Batch_3x3', 'PSNR_Batch_5x5', 'PSNR_Batch_RAN', 'PSNR_Original_Train',
               'PSNR_Val_3x3', 'PSNR_Val_5x5', 'PSNR_Val_RAN', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the Models:
# 3x3 Kernels
architecture_3x3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dhdn_3x3 = DHDN.SharedDHDN(channels=128, architecture=architecture_3x3)

# 5x5 Kernels
architecture_5x5 = [7, 7, 0, 7, 7, 0, 7, 7, 0, 7, 7, 0, 7, 7, 0, 7, 7, 0, 7, 7]
dhdn_5x5 = DHDN.SharedDHDN(channels=128, architecture=architecture_5x5)

# Size 9 - Four steps Down, Four steps Up
architecture_RAN = [3, 5, 0, 7, 4, 0, 0, 5, 0, 3, 4, 0, 4, 7, 0, 7, 2, 0, 0, 6]
dhdn_RAN = DHDN.SharedDHDN(channels=128, architecture=architecture_RAN)

dhdn_3x3 = dhdn_3x3.to(device_0)
dhdn_5x5 = dhdn_5x5.to(device_0)
dhdn_RAN = dhdn_RAN.to(device_1)

if args.load_models:
    state_dict_dhdn_3x3 = torch.load(dir_current + config['Training']['Model_Path_DHDN_3x3'], map_location=device_0)
    state_dict_dhdn_5x5 = torch.load(dir_current + config['Training']['Model_Path_DHDN_5x5'], map_location=device_0)
    state_dict_dhdn_RAN = torch.load(dir_current + config['Training']['Model_Path_DHDN_RAN'], map_location=device_1)

    if args.drop > 0:
        state_dict_dhdn_3x3 = drop_weights(state_dict_dhdn_3x3, p=args.drop, device=device_0)
        state_dict_dhdn_5x5 = drop_weights(state_dict_dhdn_5x5, p=args.drop, device=device_0)
        state_dict_dhdn_RAN = drop_weights(state_dict_dhdn_RAN, p=args.drop, device=device_1)
    if args.gaussian > 0:
        state_dict_dhdn_3x3 = gaussian_add_weights(state_dict_dhdn_3x3, k=args.gaussian, device=device_0)
        state_dict_dhdn_5x5 = gaussian_add_weights(state_dict_dhdn_5x5, k=args.gaussian, device=device_0)
        state_dict_dhdn_RAN = gaussian_add_weights(state_dict_dhdn_RAN, k=args.gaussian, device=device_1)

    dhdn_3x3.load_state_dict(state_dict_dhdn_3x3)
    dhdn_5x5.load_state_dict(state_dict_dhdn_5x5)
    dhdn_RAN.load_state_dict(state_dict_dhdn_RAN)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = 'DHDN_Compare_Kernels_' + str(args.noise)
vis_window = {'SSIM_{date}'.format(date=d1): None, 'PSNR_{date}'.format(date=d1): None}

# Define the optimizers:
optimizer_3x3 = torch.optim.Adam(dhdn_3x3.parameters(), config['Training']['Learning_Rate'])
optimizer_5x5 = torch.optim.Adam(dhdn_5x5.parameters(), config['Training']['Learning_Rate'])
optimizer_RAN = torch.optim.Adam(dhdn_RAN.parameters(), config['Training']['Learning_Rate'])

# Define the Scheduling:
scheduler_3x3 = torch.optim.lr_scheduler.StepLR(optimizer_3x3, 3, 0.5, -1)
scheduler_5x5 = torch.optim.lr_scheduler.StepLR(optimizer_5x5, 3, 0.5, -1)
scheduler_RAN = torch.optim.lr_scheduler.StepLR(optimizer_RAN, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_3x3 = nn.L1Loss().to(device_0)
loss_5x5 = nn.L1Loss().to(device_0)
loss_RAN = nn.L1Loss().to(device_1)
MSE = nn.MSELoss().to(device_1)

# Now, let us define our loggers:
loggers5 = generate_loggers()
loggers7 = generate_loggers()
loggers9 = generate_loggers()

# Training Batches
loss_batch_3x3, loss_original_batch, ssim_batch_3x3, ssim_original_batch, psnr_batch_3x3, psnr_original_batch = loggers5[0]
loss_batch_5x5, _, ssim_batch_5x5, _, psnr_batch_5x5, _ = loggers7[0]
loss_batch_RAN, _, ssim_batch_RAN, _, psnr_batch_RAN, _ = loggers9[0]

# Validation Batches
loss_batch_val_3x3, loss_original_batch_val, ssim_batch_val_3x3, ssim_original_batch_val = loggers5[1][0:4]
psnr_batch_val_3x3, psnr_original_batch_val = loggers5[1][4:]

loss_batch_val_5x5, _, ssim_batch_val_5x5, _, psnr_batch_val_5x5, _ = loggers7[1]
loss_batch_val_RAN, _, ssim_batch_val_RAN, _, psnr_batch_val_RAN, _ = loggers9[1]

# Load the Training and Validation Data:
SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing())
SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy, mat_gt_file=path_validation_gt)

dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=config['Training']['Train_Batch_Size'],
                                      shuffle=True, num_workers=16)
dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=config['Training']['Validation_Batch_Size'],
                                        shuffle=False, num_workers=8)

for epoch in range(config['Training']['Epochs']):

    for i_batch, sample_batch in enumerate(dataloader_sidd_training):
        x = sample_batch['NOISY']
        y5 = dhdn_3x3(x.to(device_0))
        y7 = dhdn_5x5(x.to(device_0))
        y9 = dhdn_RAN(x.to(device_1))
        t = sample_batch['GT']

        loss_value_3x3 = loss_3x3(y5, t.to(device_0))
        loss_value_5x5 = loss_5x5(y7, t.to(device_0))
        loss_value_RAN = loss_RAN(y9, t.to(device_1))
        loss_batch_3x3.update(loss_value_3x3.item())
        loss_batch_5x5.update(loss_value_5x5.item())
        loss_batch_RAN.update(loss_value_RAN.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss_RAN(x.to(device_1), t.to(device_1)).item())

            ssim_batch_3x3.update(SSIM(y5, t.to(device_0)).item())
            ssim_batch_5x5.update(SSIM(y7, t.to(device_0)).item())
            ssim_batch_RAN.update(SSIM(y9, t.to(device_1)).item())
            ssim_original_batch.update(SSIM(x, t).item())

            psnr_batch_3x3.update(PSNR(MSE(y5, t.to(device_0))).item())
            psnr_batch_5x5.update(PSNR(MSE(y7, t.to(device_0))).item())
            psnr_batch_RAN.update(PSNR(MSE(y9, t.to(device_1))).item())
            psnr_original_batch.update(PSNR(MSE(x.to(device_1), t.to(device_1))).item())

        # Backpropagate to train model
        optimizer_3x3.zero_grad()
        loss_value_3x3.backward()
        optimizer_3x3.step()

        optimizer_5x5.zero_grad()
        loss_value_5x5.backward()
        optimizer_5x5.step()

        optimizer_RAN.zero_grad()
        loss_value_RAN.backward()
        optimizer_RAN.step()

        if i_batch % 100 == 0:
            Display_Loss = "Loss_Size_3x3: %.6f" % loss_batch_3x3.val + \
                           "\tLoss_Size_5x5: %.6f" % loss_batch_5x5.val + \
                           "\tLoss_Size_RAN: %.6f" % loss_batch_RAN.val + \
                           "\tLoss_Original: %.6f" % loss_original_batch.val
            Display_SSIM = "SSIM_Size_3x3: %.6f" % ssim_batch_3x3.val + \
                           "\tSSIM_Size_5x5: %.6f" % ssim_batch_5x5.val + \
                           "\tSSIM_Size_RAN: %.6f" % ssim_batch_RAN.val + \
                           "\tSSIM_Original: %.6f" % ssim_original_batch.val
            Display_PSNR = "PSNR_Size_3x3: %.6f" % psnr_batch_3x3.val + \
                           "\tPSNR_Size_5x5: %.6f" % psnr_batch_5x5.val + \
                           "\tPSNR_Size_RAN: %.6f" % psnr_batch_RAN.val + \
                           "\tPSNR_Original: %.6f" % psnr_original_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

        # Free up space in GPU
        del x, y5, y7, y9, t

    Display_Loss = "Loss_Size_3x3: %.6f" % loss_batch_3x3.avg + "\tLoss_Size_5x5: %.6f" % loss_batch_5x5.avg + \
                   "\tLoss_Size_RAN: %.6f" % loss_batch_RAN.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_Size_3x3: %.6f" % ssim_batch_3x3.avg + "\tSSIM_Size_5x5: %.6f" % ssim_batch_5x5.avg + \
                   "\tSSIM_Size_RAN: %.6f" % ssim_batch_RAN.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_Size_3x3: %.6f" % psnr_batch_3x3.avg + "\tPSNR_Size_5x5: %.6f" % psnr_batch_5x5.avg + \
                   "\tPSNR_Size_RAN: %.6f" % psnr_batch_RAN.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    print("\nTotal Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']
        with torch.no_grad():
            y_v5 = dhdn_3x3(x_v.to(device_0))
            y_v7 = dhdn_5x5(x_v.to(device_0))
            y_v9 = dhdn_RAN(x_v.to(device_1))

            loss_batch_val_3x3.update(loss_3x3(y_v5, t_v.to(device_0)).item())
            loss_batch_val_5x5.update(loss_5x5(y_v7, t_v.to(device_0)).item())
            loss_batch_val_RAN.update(loss_RAN(y_v9, t_v.to(device_1)).item())
            loss_original_batch_val.update(loss_RAN(x_v.to(device_1), t_v.to(device_1)).item())

            ssim_batch_val_3x3.update(SSIM(y_v5, t_v.to(device_0)).item())
            ssim_batch_val_5x5.update(SSIM(y_v7, t_v.to(device_0)).item())
            ssim_batch_val_RAN.update(SSIM(y_v9, t_v.to(device_1)).item())
            ssim_original_batch_val.update(SSIM(x_v, t_v).item())

            psnr_batch_val_3x3.update(PSNR(MSE(y_v5, t_v.to(device_0))).item())
            psnr_batch_val_5x5.update(PSNR(MSE(y_v7, t_v.to(device_0))).item())
            psnr_batch_val_RAN.update(PSNR(MSE(y_v9, t_v.to(device_1))).item())
            psnr_original_batch_val.update(PSNR(MSE(x_v.to(device_1), t_v.to(device_1))).item())

        # Free up space in GPU
        del x_v, y_v5, y_v7, y_v9, t_v

    Display_Loss = "Loss_Size_3x3: %.6f" % loss_batch_val_3x3.avg + \
                   "\tLoss_Size_5x5: %.6f" % loss_batch_val_5x5.avg + \
                   "\tLoss_Size_RAN: %.6f" % loss_batch_val_RAN.avg + \
                   "\tLoss_Original: %.6f" % loss_original_batch_val.avg
    Display_SSIM = "SSIM_Size_3x3: %.6f" % ssim_batch_val_3x3.avg + \
                   "\tSSIM_Size_5x5: %.6f" % ssim_batch_val_5x5.avg + \
                   "\tSSIM_Size_RAN: %.6f" % ssim_batch_val_RAN.avg + \
                   "\tSSIM_Original: %.6f" % ssim_original_batch_val.avg
    Display_PSNR = "PSNR_Size_3x3: %.6f" % psnr_batch_val_3x3.avg + \
                   "\tPSNR_Size_5x5: %.6f" % psnr_batch_val_5x5.avg + \
                   "\tPSNR_Size_RAN: %.6f" % psnr_batch_val_RAN.avg + \
                   "\tPSNR_Original: %.6f" % psnr_original_batch_val.avg

    print("Validation Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_Batch_3x3': loss_batch_3x3.avg,
        'Loss_Batch_5x5': loss_batch_5x5.avg,
        'Loss_Batch_RAN': loss_batch_RAN.avg,
        'Loss_Val_3x3': loss_batch_val_3x3.avg,
        'Loss_Val_5x5': loss_batch_val_5x5.avg,
        'Loss_Val_RAN': loss_batch_val_RAN.avg,
        'Loss_Original_Train': loss_original_batch.avg,
        'Loss_Original_Val': loss_original_batch_val.avg,
        'SSIM_Batch_3x3': ssim_batch_3x3.avg,
        'SSIM_Batch_5x5': ssim_batch_5x5.avg,
        'SSIM_Batch_RAN': ssim_batch_RAN.avg,
        'SSIM_Val_3x3': ssim_batch_val_3x3.avg,
        'SSIM_Val_5x5': ssim_batch_val_5x5.avg,
        'SSIM_Val_RAN': ssim_batch_val_RAN.avg,
        'SSIM_Original_Train': ssim_original_batch.avg,
        'SSIM_Original_Val': ssim_original_batch_val.avg,
        'PSNR_Batch_3x3': psnr_batch_3x3.avg,
        'PSNR_Batch_5x5': psnr_batch_5x5.avg,
        'PSNR_Batch_RAN': psnr_batch_RAN.avg,
        'PSNR_Val_3x3': psnr_batch_val_3x3.avg,
        'PSNR_Val_5x5': psnr_batch_val_5x5.avg,
        'PSNR_Val_RAN': psnr_batch_val_RAN.avg,
        'PSNR_Original_Train': psnr_original_batch.avg,
        'PSNR_Original_Val': psnr_original_batch_val.avg
    })

    Legend = ['Size_3x3_Train', 'Size_5x5_Train', 'Size_RAN_Train', 'Orig_Train',
              'Size_3x3_Val', 'Size_5x5_Val', 'Size_RAN_Val', 'Orig_Val']

    vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([ssim_batch_3x3.avg, ssim_batch_5x5.avg, ssim_batch_RAN.avg, ssim_original_batch.avg,
                           ssim_batch_val_3x3.avg, ssim_batch_val_5x5.avg, ssim_batch_val_RAN.avg,
                           ssim_original_batch_val.avg]),
        win=vis_window['SSIM_{date}'.format(date=d1)],
        opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([psnr_batch_3x3.avg, psnr_batch_5x5.avg, psnr_batch_RAN.avg, psnr_original_batch.avg,
                           psnr_batch_val_3x3.avg, psnr_batch_val_5x5.avg, psnr_batch_val_RAN.avg,
                           psnr_original_batch_val.avg]),
        win=vis_window['PSNR_{date}'.format(date=d1)],
        opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_batch_3x3.reset()
    loss_batch_5x5.reset()
    loss_batch_RAN.reset()
    loss_original_batch.reset()
    ssim_batch_3x3.reset()
    ssim_batch_5x5.reset()
    ssim_batch_RAN.reset()
    ssim_original_batch.reset()
    psnr_batch_3x3.reset()
    psnr_batch_5x5.reset()
    psnr_batch_RAN.reset()
    psnr_original_batch.reset()
    loss_batch_val_3x3.reset()
    loss_batch_val_5x5.reset()
    loss_batch_val_RAN.reset()
    loss_original_batch_val.reset()
    ssim_batch_val_3x3.reset()
    ssim_batch_val_5x5.reset()
    ssim_batch_val_RAN.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val_3x3.reset()
    psnr_batch_val_5x5.reset()
    psnr_batch_val_RAN.reset()
    psnr_original_batch_val.reset()

    scheduler_3x3.step()
    scheduler_5x5.step()
    scheduler_RAN.step()

    if epoch > 0 and not epoch % 10:
        model_path_3x3 = dir_current + '/models/compare_kernels/{date}_dhdn_3x3_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)
        model_path_5x5 = dir_current + '/models/compare_kernels/{date}_dhdn_5x5_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)
        model_path_RAN = dir_current + '/models/compare_kernels/{date}_dhdn_RAN_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)

        torch.save(dhdn_3x3.state_dict(), model_path_3x3)
        torch.save(dhdn_5x5.state_dict(), model_path_5x5)
        torch.save(dhdn_RAN.state_dict(), model_path_RAN)

        state_dict_dhdn_3x3 = clip_weights(dhdn_3x3.state_dict(), k=3, device=device_0)
        state_dict_dhdn_5x5 = clip_weights(dhdn_5x5.state_dict(), k=3, device=device_0)
        state_dict_dhdn_RAN = clip_weights(dhdn_RAN.state_dict(), k=3, device=device_1)

        dhdn_3x3.load_state_dict(state_dict_dhdn_3x3)
        dhdn_5x5.load_state_dict(state_dict_dhdn_5x5)
        dhdn_RAN.load_state_dict(state_dict_dhdn_RAN)

# Save final model
model_path_3x3 = dir_current + '/models/compare_kernels/{date}_dhdn_3x3_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_5x5 = dir_current + '/models/compare_kernels/{date}_dhdn_5x5_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_RAN = dir_current + '/models/compare_kernels/{date}_dhdn_RAN_{noise}.pth'.format(date=d1, noise=args.noise)

torch.save(dhdn_3x3.state_dict(), model_path_3x3)
torch.save(dhdn_5x5.state_dict(), model_path_5x5)
torch.save(dhdn_RAN.state_dict(), model_path_RAN)
