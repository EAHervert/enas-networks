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
    prog='DHDN_Compare_Size',
    description='Compares 3 sizes of models based on the DHDN architecture',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--gaussian', default='-1', type=float)  # Gaussian noise addition for model weight # initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_compare_size.json'
config = json.load(open(config_path))
if not os.path.exists(dir_current + '/models/compare_size/'):
    os.makedirs(dir_current + '/models/compare_size/')

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
Field_Names = ['Loss_Batch_5', 'Loss_Batch_7', 'Loss_Batch_9', 'Loss_Original_Train',
               'Loss_Val_5', 'Loss_Val_7', 'Loss_Val_9', 'Loss_Original_Val',
               'SSIM_Batch_5', 'SSIM_Batch_7', 'SSIM_Batch_9', 'SSIM_Original_Train',
               'SSIM_Val_5', 'SSIM_Val_7', 'SSIM_Val_9', 'SSIM_Original_Val',
               'PSNR_Batch_5', 'PSNR_Batch_7', 'PSNR_Batch_9', 'PSNR_Original_Train',
               'PSNR_Val_5', 'PSNR_Val_7', 'PSNR_Val_9', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the Models:
bottleneck = [0, 0]

# Size 5 - Two steps Down, Two steps Up
encoder_5, decoder_5 = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]
architecture_5 = encoder_5 + bottleneck + decoder_5
dhdn_5 = DHDN.SharedDHDN(k_value=2, channels=128, architecture=architecture_5)

# Size 7 - Three steps Down, Three steps Up
encoder_7, decoder_7 = [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]
architecture_7 = encoder_7 + bottleneck + decoder_7
dhdn_7 = DHDN.SharedDHDN(k_value=3, channels=128, architecture=architecture_7)

# Size 9 - Four steps Down, Four steps Up
encoder_9, decoder_9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
architecture_9 = encoder_9 + bottleneck + decoder_9
dhdn_9 = DHDN.SharedDHDN(k_value=4, channels=128, architecture=architecture_9)

dhdn_5 = dhdn_5.to(device_0)
dhdn_7 = dhdn_7.to(device_0)
dhdn_9 = dhdn_9.to(device_1)

if args.load_models:
    state_dict_dhdn_5 = torch.load(dir_current + config['Training']['Model_Path_DHDN_5'], map_location=device_0)
    state_dict_dhdn_7 = torch.load(dir_current + config['Training']['Model_Path_DHDN_7'], map_location=device_0)
    state_dict_dhdn_9 = torch.load(dir_current + config['Training']['Model_Path_DHDN_9'], map_location=device_1)

    if args.drop > 0:
        state_dict_dhdn_5 = drop_weights(state_dict_dhdn_5, p=args.drop, device=device_0)
        state_dict_dhdn_7 = drop_weights(state_dict_dhdn_7, p=args.drop, device=device_0)
        state_dict_dhdn_9 = drop_weights(state_dict_dhdn_9, p=args.drop, device=device_1)
    if args.gaussian > 0:
        state_dict_dhdn_5 = gaussian_add_weights(state_dict_dhdn_5, k=args.gaussian, device=device_0)
        state_dict_dhdn_7 = gaussian_add_weights(state_dict_dhdn_7, k=args.gaussian, device=device_0)
        state_dict_dhdn_9 = gaussian_add_weights(state_dict_dhdn_9, k=args.gaussian, device=device_1)

    dhdn_5.load_state_dict(state_dict_dhdn_5)
    dhdn_7.load_state_dict(state_dict_dhdn_7)
    dhdn_9.load_state_dict(state_dict_dhdn_9)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = 'DHDN_Compare_Size_' + str(args.noise)
vis_window = {'SSIM_{date}'.format(date=d1): None, 'PSNR_{date}'.format(date=d1): None}

# Define the optimizers:
optimizer_5 = torch.optim.Adam(dhdn_5.parameters(), config['Training']['Learning_Rate'])
optimizer_7 = torch.optim.Adam(dhdn_7.parameters(), config['Training']['Learning_Rate'])
optimizer_9 = torch.optim.Adam(dhdn_9.parameters(), config['Training']['Learning_Rate'])

# Define the Scheduling:
scheduler_5 = torch.optim.lr_scheduler.StepLR(optimizer_5, 3, 0.5, -1)
scheduler_7 = torch.optim.lr_scheduler.StepLR(optimizer_7, 3, 0.5, -1)
scheduler_9 = torch.optim.lr_scheduler.StepLR(optimizer_9, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_5 = nn.L1Loss().to(device_0)
loss_7 = nn.L1Loss().to(device_0)
loss_9 = nn.L1Loss().to(device_1)
MSE = nn.MSELoss().to(device_1)

# Now, let us define our loggers:
loggers5 = generate_loggers()
loggers7 = generate_loggers()
loggers9 = generate_loggers()

# Training Batches
loss_batch_5, loss_original_batch, ssim_batch_5, ssim_original_batch, psnr_batch_5, psnr_original_batch = loggers5[0]
loss_batch_7, _, ssim_batch_7, _, psnr_batch_7, _ = loggers7[0]
loss_batch_9, _, ssim_batch_9, _, psnr_batch_9, _ = loggers9[0]

# Validation Batches
loss_batch_val_5, loss_original_batch_val, ssim_batch_val_5, ssim_original_batch_val = loggers5[1][0:4]
psnr_batch_val_5, psnr_original_batch_val = loggers5[1][4:]

loss_batch_val_7, _, ssim_batch_val_7, _, psnr_batch_val_7, _ = loggers7[1]
loss_batch_val_9, _, ssim_batch_val_9, _, psnr_batch_val_9, _ = loggers9[1]

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
        y5 = dhdn_5(x.to(device_0))
        y7 = dhdn_7(x.to(device_0))
        y9 = dhdn_9(x.to(device_1))
        t = sample_batch['GT']

        loss_value_5 = loss_5(y5, t.to(device_0))
        loss_value_7 = loss_7(y7, t.to(device_0))
        loss_value_9 = loss_9(y9, t.to(device_1))
        loss_batch_5.update(loss_value_5.item())
        loss_batch_7.update(loss_value_7.item())
        loss_batch_9.update(loss_value_9.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss_9(x.to(device_1), t.to(device_1)).item())

            ssim_batch_5.update(SSIM(y5, t.to(device_0)).item())
            ssim_batch_7.update(SSIM(y7, t.to(device_0)).item())
            ssim_batch_9.update(SSIM(y9, t.to(device_1)).item())
            ssim_original_batch.update(SSIM(x, t).item())

            psnr_batch_5.update(PSNR(MSE(y5, t.to(device_0))).item())
            psnr_batch_7.update(PSNR(MSE(y7, t.to(device_0))).item())
            psnr_batch_9.update(PSNR(MSE(y9, t.to(device_1))).item())
            psnr_original_batch.update(PSNR(MSE(x.to(device_1), t.to(device_1))).item())

        # Backpropagate to train model
        optimizer_5.zero_grad()
        loss_value_5.backward()
        optimizer_5.step()

        optimizer_7.zero_grad()
        loss_value_7.backward()
        optimizer_7.step()

        optimizer_9.zero_grad()
        loss_value_9.backward()
        optimizer_9.step()

        if i_batch % 100 == 0:
            Display_Loss = "Loss_Size_5: %.6f" % loss_batch_5.val + "\tLoss_Size_7: %.6f" % loss_batch_7.val + \
                           "\tLoss_Size_9: %.6f" % loss_batch_9.val + "\tLoss_Original: %.6f" % loss_original_batch.val
            Display_SSIM = "SSIM_Size_5: %.6f" % ssim_batch_5.val + "\tSSIM_Size_7: %.6f" % ssim_batch_7.val + \
                           "\tSSIM_Size_9: %.6f" % ssim_batch_9.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
            Display_PSNR = "PSNR_Size_5: %.6f" % psnr_batch_5.val + "\tPSNR_Size_7: %.6f" % psnr_batch_7.val + \
                           "\tPSNR_Size_9: %.6f" % psnr_batch_9.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

        # Free up space in GPU
        del x, y5, y7, y9, t

    Display_Loss = "Loss_Size_5: %.6f" % loss_batch_5.avg + "\tLoss_Size_7: %.6f" % loss_batch_7.avg + \
                   "\tLoss_Size_9: %.6f" % loss_batch_9.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_Size_5: %.6f" % ssim_batch_5.avg + "\tSSIM_Size_7: %.6f" % ssim_batch_7.avg + \
                   "\tSSIM_Size_9: %.6f" % ssim_batch_9.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_Size_5: %.6f" % psnr_batch_5.avg + "\tPSNR_Size_7: %.6f" % psnr_batch_7.avg + \
                   "\tPSNR_Size_9: %.6f" % psnr_batch_9.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    print("\nTotal Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']
        with torch.no_grad():
            y_v5 = dhdn_5(x_v.to(device_0))
            y_v7 = dhdn_7(x_v.to(device_0))
            y_v9 = dhdn_9(x_v.to(device_1))

            loss_batch_val_5.update(loss_5(y_v5, t_v.to(device_0)).item())
            loss_batch_val_7.update(loss_7(y_v7, t_v.to(device_0)).item())
            loss_batch_val_9.update(loss_9(y_v9, t_v.to(device_1)).item())
            loss_original_batch_val.update(loss_9(x_v.to(device_1), t_v.to(device_1)).item())

            ssim_batch_val_5.update(SSIM(y_v5, t_v.to(device_0)).item())
            ssim_batch_val_7.update(SSIM(y_v7, t_v.to(device_0)).item())
            ssim_batch_val_9.update(SSIM(y_v9, t_v.to(device_1)).item())
            ssim_original_batch_val.update(SSIM(x_v, t_v).item())

            psnr_batch_val_5.update(PSNR(MSE(y_v5, t_v.to(device_0))).item())
            psnr_batch_val_7.update(PSNR(MSE(y_v7, t_v.to(device_0))).item())
            psnr_batch_val_9.update(PSNR(MSE(y_v9, t_v.to(device_1))).item())
            psnr_original_batch_val.update(PSNR(MSE(x_v.to(device_1), t_v.to(device_1))).item())

        # Free up space in GPU
        del x_v, y_v5, y_v7, y_v9, t_v

    Display_Loss = "Loss_Size_5: %.6f" % loss_batch_val_5.avg + "\tLoss_Size_7: %.6f" % loss_batch_val_7.avg + \
                   "\tLoss_Size_9: %.6f" % loss_batch_val_9.avg + "\tLoss_Original: %.6f" % loss_original_batch_val.avg
    Display_SSIM = "SSIM_Size_5: %.6f" % ssim_batch_val_5.avg + "\tSSIM_Size_7: %.6f" % ssim_batch_val_7.avg + \
                   "\tSSIM_Size_9: %.6f" % ssim_batch_val_9.avg + "\tSSIM_Original: %.6f" % ssim_original_batch_val.avg
    Display_PSNR = "PSNR_Size_5: %.6f" % psnr_batch_val_5.avg + "\tPSNR_Size_7: %.6f" % psnr_batch_val_7.avg + \
                   "\tPSNR_Size_9: %.6f" % psnr_batch_val_9.avg + "\tPSNR_Original: %.6f" % psnr_original_batch_val.avg

    print("Validation Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_Batch_5': loss_batch_5.avg,
        'Loss_Batch_7': loss_batch_7.avg,
        'Loss_Batch_9': loss_batch_9.avg,
        'Loss_Val_5': loss_batch_val_5.avg,
        'Loss_Val_7': loss_batch_val_7.avg,
        'Loss_Val_9': loss_batch_val_9.avg,
        'Loss_Original_Train': loss_original_batch.avg,
        'Loss_Original_Val': loss_original_batch_val.avg,
        'SSIM_Batch_5': ssim_batch_5.avg,
        'SSIM_Batch_7': ssim_batch_7.avg,
        'SSIM_Batch_9': ssim_batch_9.avg,
        'SSIM_Val_5': ssim_batch_val_5.avg,
        'SSIM_Val_7': ssim_batch_val_7.avg,
        'SSIM_Val_9': ssim_batch_val_9.avg,
        'SSIM_Original_Train': ssim_original_batch.avg,
        'SSIM_Original_Val': ssim_original_batch_val.avg,
        'PSNR_Batch_5': psnr_batch_5.avg,
        'PSNR_Batch_7': psnr_batch_7.avg,
        'PSNR_Batch_9': psnr_batch_9.avg,
        'PSNR_Val_5': psnr_batch_val_5.avg,
        'PSNR_Val_7': psnr_batch_val_7.avg,
        'PSNR_Val_9': psnr_batch_val_9.avg,
        'PSNR_Original_Train': psnr_original_batch.avg,
        'PSNR_Original_Val': psnr_original_batch_val.avg
    })

    Legend = ['Size_5_Train', 'Size_7_Train', 'Size_9_Train', 'Orig_Train',
              'Size_5_Val', 'Size_7_Val', 'Size_9_Val', 'Orig_Val']

    vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([ssim_batch_5.avg, ssim_batch_7.avg, ssim_batch_9.avg, ssim_original_batch.avg,
                           ssim_batch_val_5.avg, ssim_batch_val_7.avg, ssim_batch_val_9.avg,
                           ssim_original_batch_val.avg]),
        win=vis_window['SSIM_{date}'.format(date=d1)],
        opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([psnr_batch_5.avg, psnr_batch_7.avg, psnr_batch_9.avg, psnr_original_batch.avg,
                           psnr_batch_val_5.avg, psnr_batch_val_7.avg, psnr_batch_val_9.avg,
                           psnr_original_batch_val.avg]),
        win=vis_window['PSNR_{date}'.format(date=d1)],
        opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_batch_5.reset()
    loss_batch_7.reset()
    loss_batch_9.reset()
    loss_original_batch.reset()
    ssim_batch_5.reset()
    ssim_batch_7.reset()
    ssim_batch_9.reset()
    ssim_original_batch.reset()
    psnr_batch_5.reset()
    psnr_batch_7.reset()
    psnr_batch_9.reset()
    psnr_original_batch.reset()
    loss_batch_val_5.reset()
    loss_batch_val_7.reset()
    loss_batch_val_9.reset()
    loss_original_batch_val.reset()
    ssim_batch_val_5.reset()
    ssim_batch_val_7.reset()
    ssim_batch_val_9.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val_5.reset()
    psnr_batch_val_7.reset()
    psnr_batch_val_9.reset()
    psnr_original_batch_val.reset()

    scheduler_5.step()
    scheduler_7.step()
    scheduler_9.step()

    if epoch > 0 and not epoch % 10:
        model_path_5 = dir_current + '/models/compare_size/{date}_dhdn_size_5_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)
        model_path_7 = dir_current + '/models/compare_size/{date}_dhdn_size_7_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)
        model_path_9 = dir_current + '/models/compare_size/{date}_dhdn_size_9_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)

        torch.save(dhdn_5.state_dict(), model_path_5)
        torch.save(dhdn_7.state_dict(), model_path_7)
        torch.save(dhdn_9.state_dict(), model_path_9)

        state_dict_dhdn_5 = clip_weights(dhdn_5.state_dict(), k=3, device=device_0)
        state_dict_dhdn_7 = clip_weights(dhdn_7.state_dict(), k=3, device=device_0)
        state_dict_dhdn_9 = clip_weights(dhdn_9.state_dict(), k=3, device=device_1)

        dhdn_5.load_state_dict(state_dict_dhdn_5)
        dhdn_7.load_state_dict(state_dict_dhdn_7)
        dhdn_9.load_state_dict(state_dict_dhdn_9)

# Save final model
model_path_5 = dir_current + '/models/compare_size/{date}_dhdn_size_5_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_7 = dir_current + '/models/compare_size/{date}_dhdn_size_7_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_9 = dir_current + '/models/compare_size/{date}_dhdn_size_9_{noise}.pth'.format(date=d1, noise=args.noise)

torch.save(dhdn_5.state_dict(), model_path_5)
torch.save(dhdn_7.state_dict(), model_path_7)
torch.save(dhdn_9.state_dict(), model_path_9)
