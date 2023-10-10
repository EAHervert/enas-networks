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
    prog='DHDN_Compare_Channel_Number',
    description='Compares 3 Different Channel Numbers',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--gaussian', default='-1', type=float)  # Gaussian noise addition for model weight # initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_compare_channel_number.json'
config = json.load(open(config_path))
if not os.path.exists(dir_current + '/models/compare_channel_number/'):
    os.makedirs(dir_current + '/models/compare_channel_number/')

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
Field_Names = ['Loss_Batch_096', 'Loss_Batch_128', 'Loss_Batch_160', 'Loss_Original_Train',
               'Loss_Val_096', 'Loss_Val_128', 'Loss_Val_160', 'Loss_Original_Val',
               'SSIM_Batch_096', 'SSIM_Batch_128', 'SSIM_Batch_160', 'SSIM_Original_Train',
               'SSIM_Val_096', 'SSIM_Val_128', 'SSIM_Val_160', 'SSIM_Original_Val',
               'PSNR_Batch_096', 'PSNR_Batch_128', 'PSNR_Batch_160', 'PSNR_Original_Train',
               'PSNR_Val_096', 'PSNR_Val_128', 'PSNR_Val_160', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the Models:
architecture = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dhdn_096 = DHDN.SharedDHDN(k_value=3, channels=96, architecture=architecture)  # 96 channels
dhdn_128 = DHDN.SharedDHDN(k_value=3, channels=128, architecture=architecture)  # 128 channels
dhdn_160 = DHDN.SharedDHDN(k_value=3, channels=160, architecture=architecture)  # 160 channels

dhdn_096 = dhdn_096.to(device_0)
dhdn_128 = dhdn_128.to(device_0)
dhdn_160 = dhdn_160.to(device_1)

if args.load_models:
    state_dict_dhdn_096 = torch.load(dir_current + config['Training']['Model_Path_DHDN_096'], map_location=device_0)
    state_dict_dhdn_128 = torch.load(dir_current + config['Training']['Model_Path_DHDN_128'], map_location=device_0)
    state_dict_dhdn_160 = torch.load(dir_current + config['Training']['Model_Path_DHDN_160'], map_location=device_1)

    if args.drop > 0:
        state_dict_dhdn_096 = drop_weights(state_dict_dhdn_096, p=args.drop, device=device_0)
        state_dict_dhdn_128 = drop_weights(state_dict_dhdn_128, p=args.drop, device=device_0)
        state_dict_dhdn_160 = drop_weights(state_dict_dhdn_160, p=args.drop, device=device_1)
    if args.gaussian > 0:
        state_dict_dhdn_096 = gaussian_add_weights(state_dict_dhdn_096, k=args.gaussian, device=device_0)
        state_dict_dhdn_128 = gaussian_add_weights(state_dict_dhdn_128, k=args.gaussian, device=device_0)
        state_dict_dhdn_160 = gaussian_add_weights(state_dict_dhdn_160, k=args.gaussian, device=device_1)

    dhdn_096.load_state_dict(state_dict_dhdn_096)
    dhdn_128.load_state_dict(state_dict_dhdn_128)
    dhdn_160.load_state_dict(state_dict_dhdn_160)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = 'DHDN_Compare_Channel_Number_' + str(args.noise)
vis_window = {'SSIM_{date}'.format(date=d1): None, 'PSNR_{date}'.format(date=d1): None}

# Define the optimizers:
optimizer_096 = torch.optim.Adam(dhdn_096.parameters(), config['Training']['Learning_Rate'])
optimizer_128 = torch.optim.Adam(dhdn_128.parameters(), config['Training']['Learning_Rate'])
optimizer_160 = torch.optim.Adam(dhdn_160.parameters(), config['Training']['Learning_Rate'])

# Define the Scheduling:
scheduler_096 = torch.optim.lr_scheduler.StepLR(optimizer_096, 3, 0.5, -1)
scheduler_128 = torch.optim.lr_scheduler.StepLR(optimizer_128, 3, 0.5, -1)
scheduler_160 = torch.optim.lr_scheduler.StepLR(optimizer_160, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_096 = nn.L1Loss().to(device_0)
loss_128 = nn.L1Loss().to(device_0)
loss_160 = nn.L1Loss().to(device_1)
MSE = nn.MSELoss().to(device_1)

# Now, let us define our loggers:
loggers_096 = generate_loggers()
loggers_128 = generate_loggers()
loggers_160 = generate_loggers()

# Training Batches
loss_batch_096, loss_original_batch, ssim_batch_096, ssim_original_batch, psnr_batch_096, psnr_original_batch = loggers_096[0]
loss_batch_128, _, ssim_batch_128, _, psnr_batch_128, _ = loggers_128[0]
loss_batch_160, _, ssim_batch_160, _, psnr_batch_160, _ = loggers_160[0]

# Validation Batches
loss_batch_val_096, loss_original_batch_val, ssim_batch_val_096, ssim_original_batch_val = loggers_096[1][0:4]
psnr_batch_val_096, psnr_original_batch_val = loggers_096[1][4:]

loss_batch_val_128, _, ssim_batch_val_128, _, psnr_batch_val_128, _ = loggers_128[1]
loss_batch_val_160, _, ssim_batch_val_160, _, psnr_batch_val_160, _ = loggers_160[1]

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
        y_096 = dhdn_096(x.to(device_0))
        y_128 = dhdn_128(x.to(device_0))
        y_160 = dhdn_160(x.to(device_1))
        t = sample_batch['GT']

        loss_value_096 = loss_096(y_096, t.to(device_0))
        loss_value_128 = loss_128(y_128, t.to(device_0))
        loss_value_160 = loss_160(y_160, t.to(device_1))
        loss_batch_096.update(loss_value_096.item())
        loss_batch_128.update(loss_value_128.item())
        loss_batch_160.update(loss_value_160.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss_160(x.to(device_1), t.to(device_1)).item())

            ssim_batch_096.update(SSIM(y_096, t.to(device_0)).item())
            ssim_batch_128.update(SSIM(y_128, t.to(device_0)).item())
            ssim_batch_160.update(SSIM(y_160, t.to(device_1)).item())
            ssim_original_batch.update(SSIM(x, t).item())

            psnr_batch_096.update(PSNR(MSE(y_096, t.to(device_0))).item())
            psnr_batch_128.update(PSNR(MSE(y_128, t.to(device_0))).item())
            psnr_batch_160.update(PSNR(MSE(y_160, t.to(device_1))).item())
            psnr_original_batch.update(PSNR(MSE(x.to(device_1), t.to(device_1))).item())

        # Backpropagate to train model
        optimizer_096.zero_grad()
        loss_value_096.backward()
        optimizer_096.step()

        optimizer_128.zero_grad()
        loss_value_128.backward()
        optimizer_128.step()

        optimizer_160.zero_grad()
        loss_value_160.backward()
        optimizer_160.step()

        if i_batch % 100 == 0:
            Display_Loss = "Loss_Size_096: %.6f" % loss_batch_096.val + "\tLoss_Size_128: %.6f" % loss_batch_128.val + \
                           "\tLoss_Size_160: %.6f" % loss_batch_160.val + "\tLoss_Original: %.6f" % loss_original_batch.val
            Display_SSIM = "SSIM_Size_096: %.6f" % ssim_batch_096.val + "\tSSIM_Size_128: %.6f" % ssim_batch_128.val + \
                           "\tSSIM_Size_160: %.6f" % ssim_batch_160.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
            Display_PSNR = "PSNR_Size_096: %.6f" % psnr_batch_096.val + "\tPSNR_Size_128: %.6f" % psnr_batch_128.val + \
                           "\tPSNR_Size_160: %.6f" % psnr_batch_160.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

        # Free up space in GPU
        del x, y_096, y_128, y_160, t

    Display_Loss = "Loss_Size_096: %.6f" % loss_batch_096.avg + "\tLoss_Size_128: %.6f" % loss_batch_128.avg + \
                   "\tLoss_Size_160: %.6f" % loss_batch_160.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_Size_096: %.6f" % ssim_batch_096.avg + "\tSSIM_Size_128: %.6f" % ssim_batch_128.avg + \
                   "\tSSIM_Size_160: %.6f" % ssim_batch_160.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_Size_096: %.6f" % psnr_batch_096.avg + "\tPSNR_Size_128: %.6f" % psnr_batch_128.avg + \
                   "\tPSNR_Size_160: %.6f" % psnr_batch_160.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    print("\nTotal Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']
        with torch.no_grad():
            y_v_096 = dhdn_096(x_v.to(device_0))
            y_v_128 = dhdn_128(x_v.to(device_0))
            y_v_160 = dhdn_160(x_v.to(device_1))

            loss_batch_val_096.update(loss_096(y_v_096, t_v.to(device_0)).item())
            loss_batch_val_128.update(loss_128(y_v_128, t_v.to(device_0)).item())
            loss_batch_val_160.update(loss_160(y_v_160, t_v.to(device_1)).item())
            loss_original_batch_val.update(loss_160(x_v.to(device_1), t_v.to(device_1)).item())

            ssim_batch_val_096.update(SSIM(y_v_096, t_v.to(device_0)).item())
            ssim_batch_val_128.update(SSIM(y_v_128, t_v.to(device_0)).item())
            ssim_batch_val_160.update(SSIM(y_v_160, t_v.to(device_1)).item())
            ssim_original_batch_val.update(SSIM(x_v, t_v).item())

            psnr_batch_val_096.update(PSNR(MSE(y_v_096, t_v.to(device_0))).item())
            psnr_batch_val_128.update(PSNR(MSE(y_v_128, t_v.to(device_0))).item())
            psnr_batch_val_160.update(PSNR(MSE(y_v_160, t_v.to(device_1))).item())
            psnr_original_batch_val.update(PSNR(MSE(x_v.to(device_1), t_v.to(device_1))).item())

        # Free up space in GPU
        del x_v, y_v_096, y_v_128, y_v_160, t_v

    Display_Loss = "Loss_Size_096: %.6f" % loss_batch_val_096.val + "\tLoss_Size_128: %.6f" % loss_batch_val_128.val + \
                   "\tLoss_Size_160: %.6f" % loss_batch_val_160.val + "\tLoss_Original: %.6f" % loss_original_batch_val.val
    Display_SSIM = "SSIM_Size_096: %.6f" % ssim_batch_val_096.val + "\tSSIM_Size_128: %.6f" % ssim_batch_val_128.val + \
                   "\tSSIM_Size_160: %.6f" % ssim_batch_val_160.val + "\tSSIM_Original: %.6f" % ssim_original_batch_val.val
    Display_PSNR = "PSNR_Size_096: %.6f" % psnr_batch_val_096.val + "\tPSNR_Size_128: %.6f" % psnr_batch_val_128.val + \
                   "\tPSNR_Size_160: %.6f" % psnr_batch_val_160.val + "\tPSNR_Original: %.6f" % psnr_original_batch_val.val

    print("Validation Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_Batch_096': loss_batch_096.avg,
        'Loss_Batch_128': loss_batch_128.avg,
        'Loss_Batch_160': loss_batch_160.avg,
        'Loss_Val_096': loss_batch_val_096.avg,
        'Loss_Val_128': loss_batch_val_128.avg,
        'Loss_Val_160': loss_batch_val_160.avg,
        'Loss_Original_Train': loss_original_batch.avg,
        'Loss_Original_Val': loss_original_batch_val.avg,
        'SSIM_Batch_096': ssim_batch_096.avg,
        'SSIM_Batch_128': ssim_batch_128.avg,
        'SSIM_Batch_160': ssim_batch_160.avg,
        'SSIM_Val_096': ssim_batch_val_096.avg,
        'SSIM_Val_128': ssim_batch_val_128.avg,
        'SSIM_Val_160': ssim_batch_val_160.avg,
        'SSIM_Original_Train': ssim_original_batch.avg,
        'SSIM_Original_Val': ssim_original_batch_val.avg,
        'PSNR_Batch_096': psnr_batch_096.avg,
        'PSNR_Batch_128': psnr_batch_128.avg,
        'PSNR_Batch_160': psnr_batch_160.avg,
        'PSNR_Val_096': psnr_batch_val_096.avg,
        'PSNR_Val_128': psnr_batch_val_128.avg,
        'PSNR_Val_160': psnr_batch_val_160.avg,
        'PSNR_Original_Train': psnr_original_batch.avg,
        'PSNR_Original_Val': psnr_original_batch_val.avg
    })

    Legend = ['Size_096_Train', 'Size_128_Train', 'Size_160_Train', 'Orig_Train',
              'Size_096_Val', 'Size_128_Val', 'Size_160_Val', 'Orig_Val']

    vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([ssim_batch_096.avg, ssim_batch_128.avg, ssim_batch_160.avg, ssim_original_batch.avg,
                           ssim_batch_val_096.avg, ssim_batch_val_128.avg, ssim_batch_val_160.avg,
                           ssim_original_batch_val.avg]),
        win=vis_window['SSIM_{date}'.format(date=d1)],
        opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([psnr_batch_096.avg, psnr_batch_128.avg, psnr_batch_160.avg, psnr_original_batch.avg,
                           psnr_batch_val_096.avg, psnr_batch_val_128.avg, psnr_batch_val_160.avg,
                           psnr_original_batch_val.avg]),
        win=vis_window['PSNR_{date}'.format(date=d1)],
        opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_batch_096.reset()
    loss_batch_128.reset()
    loss_batch_160.reset()
    loss_original_batch.reset()
    ssim_batch_096.reset()
    ssim_batch_128.reset()
    ssim_batch_160.reset()
    ssim_original_batch.reset()
    psnr_batch_096.reset()
    psnr_batch_128.reset()
    psnr_batch_160.reset()
    psnr_original_batch.reset()
    loss_batch_val_096.reset()
    loss_batch_val_128.reset()
    loss_batch_val_160.reset()
    loss_original_batch_val.reset()
    ssim_batch_val_096.reset()
    ssim_batch_val_128.reset()
    ssim_batch_val_160.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val_096.reset()
    psnr_batch_val_128.reset()
    psnr_batch_val_160.reset()
    psnr_original_batch_val.reset()

    scheduler_096.step()
    scheduler_128.step()
    scheduler_160.step()

    if epoch > 0 and not epoch % 10:
        model_path_096 = dir_current + \
                         '/models/compare_channel_number/{date}_dhdn_channel_number_096_{noise}_{epoch}.pth'.format(
                             date=d1, noise=args.noise, epoch=epoch)
        model_path_128 = dir_current + \
                         '/models/compare_channel_number/{date}_dhdn_channel_number_128_{noise}_{epoch}.pth'.format(
                             date=d1, noise=args.noise, epoch=epoch)
        model_path_160 = dir_current + \
                         '/models/compare_channel_number/{date}_dhdn_channel_number_160_{noise}_{epoch}.pth'.format(
                             date=d1, noise=args.noise, epoch=epoch)

        torch.save(dhdn_096.state_dict(), model_path_096)
        torch.save(dhdn_128.state_dict(), model_path_128)
        torch.save(dhdn_160.state_dict(), model_path_160)

        state_dict_dhdn_096 = clip_weights(dhdn_096.state_dict(), k=3, device=device_0)
        state_dict_dhdn_128 = clip_weights(dhdn_128.state_dict(), k=3, device=device_0)
        state_dict_dhdn_160 = clip_weights(dhdn_160.state_dict(), k=3, device=device_1)

        dhdn_096.load_state_dict(state_dict_dhdn_096)
        dhdn_128.load_state_dict(state_dict_dhdn_128)
        dhdn_160.load_state_dict(state_dict_dhdn_160)

# Save final model
model_path_096 = dir_current + \
                 '/models/compare_channel_number/{date}_dhdn_channel_number_096_{noise}.pth'.format(date=d1,
                                                                                                    noise=args.noise)
model_path_128 = dir_current + \
                 '/models/compare_channel_number/{date}_dhdn_channel_number_128_{noise}.pth'.format(date=d1,
                                                                                                    noise=args.noise)
model_path_160 = dir_current + \
                 '/models/compare_channel_number/{date}_dhdn_channel_number_160_{noise}.pth'.format(date=d1,
                                                                                                    noise=args.noise)

torch.save(dhdn_096.state_dict(), model_path_096)
torch.save(dhdn_128.state_dict(), model_path_128)
torch.save(dhdn_160.state_dict(), model_path_160)
