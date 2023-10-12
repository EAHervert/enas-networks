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
    prog='DHDN_Compare_Down',
    description='Compares 3 downsampling methods',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--gaussian', default='-1', type=float)  # Gaussian noise addition for model weight # initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_compare_down.json'
config = json.load(open(config_path))
if not os.path.exists(dir_current + '/models/compare_down/'):
    os.makedirs(dir_current + '/models/compare_down/')

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
Field_Names = ['Loss_Batch_MAX', 'Loss_Batch_AVG', 'Loss_Batch_CNV', 'Loss_Original_Train',
               'Loss_Val_MAX', 'Loss_Val_AVG', 'Loss_Val_CNV', 'Loss_Original_Val',
               'SSIM_Batch_MAX', 'SSIM_Batch_AVG', 'SSIM_Batch_CNV', 'SSIM_Original_Train',
               'SSIM_Val_MAX', 'SSIM_Val_AVG', 'SSIM_Val_CNV', 'SSIM_Original_Val',
               'PSNR_Batch_MAX', 'PSNR_Batch_AVG', 'PSNR_Batch_CNV', 'PSNR_Original_Train',
               'PSNR_Val_MAX', 'PSNR_Val_AVG', 'PSNR_Val_CNV', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the Models:
decoder, bottleneck = [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0]

# Max Pooling
encoder_MAX = [0, 0, 0, 0, 0, 0, 0, 0, 0]
architecture_MAX = encoder_MAX + bottleneck + decoder
dhdn_MAX = DHDN.SharedDHDN(channels=128, architecture=architecture_MAX)

# Average Pooling
encoder_AVG = [0, 0, 1, 0, 0, 1, 0, 0, 1]
architecture_AVG = encoder_AVG + bottleneck + decoder
dhdn_AVG = DHDN.SharedDHDN(channels=128, architecture=architecture_AVG)

# Convolutional Downsampling
encoder_CNV = [0, 0, 2, 0, 0, 2, 0, 0, 2]
architecture_CNV = encoder_CNV + bottleneck + decoder
dhdn_CNV = DHDN.SharedDHDN(channels=128, architecture=architecture_CNV)

dhdn_MAX = dhdn_MAX.to(device_0)
dhdn_AVG = dhdn_AVG.to(device_0)
dhdn_CNV = dhdn_CNV.to(device_1)

if args.load_models:
    state_dict_dhdn_MAX = torch.load(dir_current + config['Training']['Model_Path_DHDN_MAX'], map_location=device_0)
    state_dict_dhdn_AVG = torch.load(dir_current + config['Training']['Model_Path_DHDN_AVG'], map_location=device_0)
    state_dict_dhdn_CNV = torch.load(dir_current + config['Training']['Model_Path_DHDN_CNV'], map_location=device_1)

    if args.drop > 0:
        state_dict_dhdn_MAX = drop_weights(state_dict_dhdn_MAX, p=args.drop, device=device_0)
        state_dict_dhdn_AVG = drop_weights(state_dict_dhdn_AVG, p=args.drop, device=device_0)
        state_dict_dhdn_CNV = drop_weights(state_dict_dhdn_CNV, p=args.drop, device=device_1)
    if args.gaussian > 0:
        state_dict_dhdn_MAX = gaussian_add_weights(state_dict_dhdn_MAX, k=args.gaussian, device=device_0)
        state_dict_dhdn_AVG = gaussian_add_weights(state_dict_dhdn_AVG, k=args.gaussian, device=device_0)
        state_dict_dhdn_CNV = gaussian_add_weights(state_dict_dhdn_CNV, k=args.gaussian, device=device_1)

    dhdn_MAX.load_state_dict(state_dict_dhdn_MAX)
    dhdn_AVG.load_state_dict(state_dict_dhdn_AVG)
    dhdn_CNV.load_state_dict(state_dict_dhdn_CNV)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = 'DHDN_Compare_Down_' + str(args.noise)
vis_window = {'SSIM_{date}'.format(date=d1): None, 'PSNR_{date}'.format(date=d1): None}

# Define the optimizers:
optimizer_MAX = torch.optim.Adam(dhdn_MAX.parameters(), config['Training']['Learning_Rate'])
optimizer_AVG = torch.optim.Adam(dhdn_AVG.parameters(), config['Training']['Learning_Rate'])
optimizer_CNV = torch.optim.Adam(dhdn_CNV.parameters(), config['Training']['Learning_Rate'])

# Define the Scheduling:
scheduler_MAX = torch.optim.lr_scheduler.StepLR(optimizer_MAX, 3, 0.5, -1)
scheduler_AVG = torch.optim.lr_scheduler.StepLR(optimizer_AVG, 3, 0.5, -1)
scheduler_CNV = torch.optim.lr_scheduler.StepLR(optimizer_CNV, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_MAX = nn.L1Loss().to(device_0)
loss_AVG = nn.L1Loss().to(device_0)
loss_CNV = nn.L1Loss().to(device_1)
MSE = nn.MSELoss().to(device_1)

# Now, let us define our loggers:
loggersPS = generate_loggers()
loggersTC = generate_loggers()
loggersBL = generate_loggers()

# Training Batches
loss_batch_MAX, loss_original_batch, ssim_batch_MAX, ssim_original_batch = loggersPS[0][0:4]
psnr_batch_MAX, psnr_original_batch = loggersPS[0][4:]
loss_batch_AVG, _, ssim_batch_AVG, _, psnr_batch_AVG, _ = loggersTC[0]
loss_batch_CNV, _, ssim_batch_CNV, _, psnr_batch_CNV, _ = loggersBL[0]

# Validation Batches
loss_batch_val_MAX, loss_original_batch_val, ssim_batch_val_MAX, ssim_original_batch_val = loggersPS[1][0:4]
psnr_batch_val_MAX, psnr_original_batch_val = loggersPS[1][4:]

loss_batch_val_AVG, _, ssim_batch_val_AVG, _, psnr_batch_val_AVG, _ = loggersTC[1]
loss_batch_val_CNV, _, ssim_batch_val_CNV, _, psnr_batch_val_CNV, _ = loggersBL[1]

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
        y_MAX = dhdn_MAX(x.to(device_0))
        y_AVG = dhdn_AVG(x.to(device_0))
        y_CNV = dhdn_CNV(x.to(device_1))
        t = sample_batch['GT']

        loss_value_MAX = loss_MAX(y_MAX, t.to(device_0))
        loss_value_AVG = loss_AVG(y_AVG, t.to(device_0))
        loss_value_CNV = loss_CNV(y_CNV, t.to(device_1))
        loss_batch_MAX.update(loss_value_MAX.item())
        loss_batch_AVG.update(loss_value_AVG.item())
        loss_batch_CNV.update(loss_value_CNV.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss_CNV(x.to(device_1), t.to(device_1)).item())

            ssim_batch_MAX.update(SSIM(y_MAX, t.to(device_0)).item())
            ssim_batch_AVG.update(SSIM(y_AVG, t.to(device_0)).item())
            ssim_batch_CNV.update(SSIM(y_CNV, t.to(device_1)).item())
            ssim_original_batch.update(SSIM(x, t).item())

            psnr_batch_MAX.update(PSNR(MSE(y_MAX, t.to(device_0))).item())
            psnr_batch_AVG.update(PSNR(MSE(y_AVG, t.to(device_0))).item())
            psnr_batch_CNV.update(PSNR(MSE(y_CNV, t.to(device_1))).item())
            psnr_original_batch.update(PSNR(MSE(x.to(device_1), t.to(device_1))).item())

        # Backpropagate to train model
        optimizer_MAX.zero_grad()
        loss_value_MAX.backward()
        optimizer_MAX.step()

        optimizer_AVG.zero_grad()
        loss_value_AVG.backward()
        optimizer_AVG.step()

        optimizer_CNV.zero_grad()
        loss_value_CNV.backward()
        optimizer_CNV.step()

        if i_batch % 100 == 0:
            Display_Loss = "Loss_Size_MAX: %.6f" % loss_batch_MAX.val + "\tLoss_Size_AVG: %.6f" % loss_batch_AVG.val + \
                           "\tLoss_Size_CNV: %.6f" % loss_batch_CNV.val + "\tLoss_Original: %.6f" % loss_original_batch.val
            Display_SSIM = "SSIM_Size_MAX: %.6f" % ssim_batch_MAX.val + "\tSSIM_Size_AVG: %.6f" % ssim_batch_AVG.val + \
                           "\tSSIM_Size_CNV: %.6f" % ssim_batch_CNV.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
            Display_PSNR = "PSNR_Size_MAX: %.6f" % psnr_batch_MAX.val + "\tPSNR_Size_AVG: %.6f" % psnr_batch_AVG.val + \
                           "\tPSNR_Size_CNV: %.6f" % psnr_batch_CNV.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

        # Free up space in GPU
        del x, y_MAX, y_AVG, y_CNV, t

    Display_Loss = "Loss_Size_MAX: %.6f" % loss_batch_MAX.avg + "\tLoss_Size_AVG: %.6f" % loss_batch_AVG.avg + \
                   "\tLoss_Size_CNV: %.6f" % loss_batch_CNV.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_Size_MAX: %.6f" % ssim_batch_MAX.avg + "\tSSIM_Size_AVG: %.6f" % ssim_batch_AVG.avg + \
                   "\tSSIM_Size_CNV: %.6f" % ssim_batch_CNV.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_Size_MAX: %.6f" % psnr_batch_MAX.avg + "\tPSNR_Size_AVG: %.6f" % psnr_batch_AVG.avg + \
                   "\tPSNR_Size_CNV: %.6f" % psnr_batch_CNV.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    print("\nTotal Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']
        with torch.no_grad():
            y_v_MAX = dhdn_MAX(x_v.to(device_0))
            y_v_AVG = dhdn_AVG(x_v.to(device_0))
            y_v_CNV = dhdn_CNV(x_v.to(device_1))

            loss_batch_val_MAX.update(loss_MAX(y_v_MAX, t_v.to(device_0)).item())
            loss_batch_val_AVG.update(loss_AVG(y_v_AVG, t_v.to(device_0)).item())
            loss_batch_val_CNV.update(loss_CNV(y_v_CNV, t_v.to(device_1)).item())
            loss_original_batch_val.update(loss_CNV(x_v.to(device_1), t_v.to(device_1)).item())

            ssim_batch_val_MAX.update(SSIM(y_v_MAX, t_v.to(device_0)).item())
            ssim_batch_val_AVG.update(SSIM(y_v_AVG, t_v.to(device_0)).item())
            ssim_batch_val_CNV.update(SSIM(y_v_CNV, t_v.to(device_1)).item())
            ssim_original_batch_val.update(SSIM(x_v, t_v).item())

            psnr_batch_val_MAX.update(PSNR(MSE(y_v_MAX, t_v.to(device_0))).item())
            psnr_batch_val_AVG.update(PSNR(MSE(y_v_AVG, t_v.to(device_0))).item())
            psnr_batch_val_CNV.update(PSNR(MSE(y_v_CNV, t_v.to(device_1))).item())
            psnr_original_batch_val.update(PSNR(MSE(x_v.to(device_1), t_v.to(device_1))).item())

        # Free up space in GPU
        del x_v, y_v_MAX, y_v_AVG, y_v_CNV, t_v

    Display_Loss = "Loss_Size_MAX: %.6f" % loss_batch_val_MAX.val + "\tLoss_Size_AVG: %.6f" % loss_batch_val_AVG.val + \
                   "\tLoss_Size_CNV: %.6f" % loss_batch_val_CNV.val + "\tLoss_Original: %.6f" % loss_original_batch_val.val
    Display_SSIM = "SSIM_Size_MAX: %.6f" % ssim_batch_val_MAX.val + "\tSSIM_Size_AVG: %.6f" % ssim_batch_val_AVG.val + \
                   "\tSSIM_Size_CNV: %.6f" % ssim_batch_val_CNV.val + "\tSSIM_Original: %.6f" % ssim_original_batch_val.val
    Display_PSNR = "PSNR_Size_MAX: %.6f" % psnr_batch_val_MAX.val + "\tPSNR_Size_AVG: %.6f" % psnr_batch_val_AVG.val + \
                   "\tPSNR_Size_CNV: %.6f" % psnr_batch_val_CNV.val + "\tPSNR_Original: %.6f" % psnr_original_batch_val.val

    print("Validation Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_Batch_MAX': loss_batch_MAX.avg,
        'Loss_Batch_AVG': loss_batch_AVG.avg,
        'Loss_Batch_CNV': loss_batch_CNV.avg,
        'Loss_Val_MAX': loss_batch_val_MAX.avg,
        'Loss_Val_AVG': loss_batch_val_AVG.avg,
        'Loss_Val_CNV': loss_batch_val_CNV.avg,
        'Loss_Original_Train': loss_original_batch.avg,
        'Loss_Original_Val': loss_original_batch_val.avg,
        'SSIM_Batch_MAX': ssim_batch_MAX.avg,
        'SSIM_Batch_AVG': ssim_batch_AVG.avg,
        'SSIM_Batch_CNV': ssim_batch_CNV.avg,
        'SSIM_Val_MAX': ssim_batch_val_MAX.avg,
        'SSIM_Val_AVG': ssim_batch_val_AVG.avg,
        'SSIM_Val_CNV': ssim_batch_val_CNV.avg,
        'SSIM_Original_Train': ssim_original_batch.avg,
        'SSIM_Original_Val': ssim_original_batch_val.avg,
        'PSNR_Batch_MAX': psnr_batch_MAX.avg,
        'PSNR_Batch_AVG': psnr_batch_AVG.avg,
        'PSNR_Batch_CNV': psnr_batch_CNV.avg,
        'PSNR_Val_MAX': psnr_batch_val_MAX.avg,
        'PSNR_Val_AVG': psnr_batch_val_AVG.avg,
        'PSNR_Val_CNV': psnr_batch_val_CNV.avg,
        'PSNR_Original_Train': psnr_original_batch.avg,
        'PSNR_Original_Val': psnr_original_batch_val.avg
    })

    Legend = ['Size_MAX_Train', 'Size_AVG_Train', 'Size_CNV_Train', 'Orig_Train',
              'Size_MAX_Val', 'Size_AVG_Val', 'Size_CNV_Val', 'Orig_Val']

    vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([ssim_batch_MAX.avg, ssim_batch_AVG.avg, ssim_batch_CNV.avg, ssim_original_batch.avg,
                           ssim_batch_val_MAX.avg, ssim_batch_val_AVG.avg, ssim_batch_val_CNV.avg,
                           ssim_original_batch_val.avg]),
        win=vis_window['SSIM_{date}'.format(date=d1)],
        opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([psnr_batch_MAX.avg, psnr_batch_AVG.avg, psnr_batch_CNV.avg, psnr_original_batch.avg,
                           psnr_batch_val_MAX.avg, psnr_batch_val_AVG.avg, psnr_batch_val_CNV.avg,
                           psnr_original_batch_val.avg]),
        win=vis_window['PSNR_{date}'.format(date=d1)],
        opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_batch_MAX.reset()
    loss_batch_AVG.reset()
    loss_batch_CNV.reset()
    loss_original_batch.reset()
    ssim_batch_MAX.reset()
    ssim_batch_AVG.reset()
    ssim_batch_CNV.reset()
    ssim_original_batch.reset()
    psnr_batch_MAX.reset()
    psnr_batch_AVG.reset()
    psnr_batch_CNV.reset()
    psnr_original_batch.reset()
    loss_batch_val_MAX.reset()
    loss_batch_val_AVG.reset()
    loss_batch_val_CNV.reset()
    loss_original_batch_val.reset()
    ssim_batch_val_MAX.reset()
    ssim_batch_val_AVG.reset()
    ssim_batch_val_CNV.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val_MAX.reset()
    psnr_batch_val_AVG.reset()
    psnr_batch_val_CNV.reset()
    psnr_original_batch_val.reset()

    scheduler_MAX.step()
    scheduler_AVG.step()
    scheduler_CNV.step()

    if epoch > 0 and not epoch % 10:
        model_path_MAX = dir_current + '/models/compare_down/{date}_dhdn_down_MAX_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)
        model_path_AVG = dir_current + '/models/compare_down/{date}_dhdn_down_AVG_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)
        model_path_CNV = dir_current + '/models/compare_down/{date}_dhdn_down_CNV_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)

        torch.save(dhdn_MAX.state_dict(), model_path_MAX)
        torch.save(dhdn_AVG.state_dict(), model_path_AVG)
        torch.save(dhdn_CNV.state_dict(), model_path_CNV)

        state_dict_dhdn_MAX = clip_weights(dhdn_MAX.state_dict(), k=3, device=device_0)
        state_dict_dhdn_AVG = clip_weights(dhdn_AVG.state_dict(), k=3, device=device_0)
        state_dict_dhdn_CNV = clip_weights(dhdn_CNV.state_dict(), k=3, device=device_1)

        dhdn_MAX.load_state_dict(state_dict_dhdn_MAX)
        dhdn_AVG.load_state_dict(state_dict_dhdn_AVG)
        dhdn_CNV.load_state_dict(state_dict_dhdn_CNV)

# Save final model
model_path_MAX = dir_current + '/models/compare_down/{date}_dhdn_down_MAX_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_AVG = dir_current + '/models/compare_down/{date}_dhdn_down_AVG_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_CNV = dir_current + '/models/compare_down/{date}_dhdn_down_CNV_{noise}.pth'.format(date=d1, noise=args.noise)

torch.save(dhdn_MAX.state_dict(), model_path_MAX)
torch.save(dhdn_AVG.state_dict(), model_path_AVG)
torch.save(dhdn_CNV.state_dict(), model_path_CNV)
