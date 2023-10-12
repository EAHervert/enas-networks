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
    prog='DHDN_Compare_Up',
    description='Compares 3 upsampling methods',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--gaussian', default='-1', type=float)  # Gaussian noise addition for model weight # initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_compare_up.json'
config = json.load(open(config_path))
if not os.path.exists(dir_current + '/models/compare_up/'):
    os.makedirs(dir_current + '/models/compare_up/')

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
Field_Names = ['Loss_Batch_PS', 'Loss_Batch_TC', 'Loss_Batch_BL', 'Loss_Original_Train',
               'Loss_Val_PS', 'Loss_Val_TC', 'Loss_Val_BL', 'Loss_Original_Val',
               'SSIM_Batch_PS', 'SSIM_Batch_TC', 'SSIM_Batch_BL', 'SSIM_Original_Train',
               'SSIM_Val_PS', 'SSIM_Val_TC', 'SSIM_Val_BL', 'SSIM_Original_Val',
               'PSNR_Batch_PS', 'PSNR_Batch_TC', 'PSNR_Batch_BL', 'PSNR_Original_Train',
               'PSNR_Val_PS', 'PSNR_Val_TC', 'PSNR_Val_BL', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the Models:
encoder, bottleneck = [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0]

# Pixel Shuffling
decoder_PS = [0, 0, 0, 0, 0, 0, 0, 0, 0]
architecture_PS = encoder + bottleneck + decoder_PS
dhdn_PS = DHDN.SharedDHDN(channels=128, architecture=architecture_PS)

# Transpose Convolution
decoder_TC = [1, 0, 0, 1, 0, 0, 1, 0, 0]
architecture_TC = encoder + bottleneck + decoder_TC
dhdn_TC = DHDN.SharedDHDN(channels=128, architecture=architecture_TC)

# Bilinear Interpolation
decoder_BL = [2, 0, 0, 2, 0, 0, 2, 0, 0]
architecture_BL = encoder + bottleneck + decoder_BL
dhdn_BL = DHDN.SharedDHDN(channels=128, architecture=architecture_BL)

dhdn_PS = dhdn_PS.to(device_0)
dhdn_TC = dhdn_TC.to(device_0)
dhdn_BL = dhdn_BL.to(device_1)

if args.load_models:
    state_dict_dhdn_PS = torch.load(dir_current + config['Training']['Model_Path_DHDN_PS'], map_location=device_0)
    state_dict_dhdn_TC = torch.load(dir_current + config['Training']['Model_Path_DHDN_TC'], map_location=device_0)
    state_dict_dhdn_BL = torch.load(dir_current + config['Training']['Model_Path_DHDN_BL'], map_location=device_1)

    if args.drop > 0:
        state_dict_dhdn_PS = drop_weights(state_dict_dhdn_PS, p=args.drop, device=device_0)
        state_dict_dhdn_TC = drop_weights(state_dict_dhdn_TC, p=args.drop, device=device_0)
        state_dict_dhdn_BL = drop_weights(state_dict_dhdn_BL, p=args.drop, device=device_1)
    if args.gaussian > 0:
        state_dict_dhdn_PS = gaussian_add_weights(state_dict_dhdn_PS, k=args.gaussian, device=device_0)
        state_dict_dhdn_TC = gaussian_add_weights(state_dict_dhdn_TC, k=args.gaussian, device=device_0)
        state_dict_dhdn_BL = gaussian_add_weights(state_dict_dhdn_BL, k=args.gaussian, device=device_1)

    dhdn_PS.load_state_dict(state_dict_dhdn_PS)
    dhdn_TC.load_state_dict(state_dict_dhdn_TC)
    dhdn_BL.load_state_dict(state_dict_dhdn_BL)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = 'DHDN_Compare_Up_' + str(args.noise)
vis_window = {'SSIM_{date}'.format(date=d1): None, 'PSNR_{date}'.format(date=d1): None}

# Define the optimizers:
optimizer_PS = torch.optim.Adam(dhdn_PS.parameters(), config['Training']['Learning_Rate'])
optimizer_TC = torch.optim.Adam(dhdn_TC.parameters(), config['Training']['Learning_Rate'])
optimizer_BL = torch.optim.Adam(dhdn_BL.parameters(), config['Training']['Learning_Rate'])

# Define the Scheduling:
scheduler_PS = torch.optim.lr_scheduler.StepLR(optimizer_PS, 3, 0.5, -1)
scheduler_TC = torch.optim.lr_scheduler.StepLR(optimizer_TC, 3, 0.5, -1)
scheduler_BL = torch.optim.lr_scheduler.StepLR(optimizer_BL, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_PS = nn.L1Loss().to(device_0)
loss_TC = nn.L1Loss().to(device_0)
loss_BL = nn.L1Loss().to(device_1)
MSE = nn.MSELoss().to(device_1)

# Now, let us define our loggers:
loggersPS = generate_loggers()
loggersTC = generate_loggers()
loggersBL = generate_loggers()

# Training Batches
loss_batch_PS, loss_original_batch, ssim_batch_PS, ssim_original_batch = loggersPS[0][0:4]
psnr_batch_PS, psnr_original_batch = loggersPS[0][4:]
loss_batch_TC, _, ssim_batch_TC, _, psnr_batch_TC, _ = loggersTC[0]
loss_batch_BL, _, ssim_batch_BL, _, psnr_batch_BL, _ = loggersBL[0]

# Validation Batches
loss_batch_val_PS, loss_original_batch_val, ssim_batch_val_PS, ssim_original_batch_val = loggersPS[1][0:4]
psnr_batch_val_PS, psnr_original_batch_val = loggersPS[1][4:]

loss_batch_val_TC, _, ssim_batch_val_TC, _, psnr_batch_val_TC, _ = loggersTC[1]
loss_batch_val_BL, _, ssim_batch_val_BL, _, psnr_batch_val_BL, _ = loggersBL[1]

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
        y_PS = dhdn_PS(x.to(device_0))
        y_TC = dhdn_TC(x.to(device_0))
        y_BL = dhdn_BL(x.to(device_1))
        t = sample_batch['GT']

        loss_value_PS = loss_PS(y_PS, t.to(device_0))
        loss_value_TC = loss_TC(y_TC, t.to(device_0))
        loss_value_BL = loss_BL(y_BL, t.to(device_1))
        loss_batch_PS.update(loss_value_PS.item())
        loss_batch_TC.update(loss_value_TC.item())
        loss_batch_BL.update(loss_value_BL.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss_BL(x.to(device_1), t.to(device_1)).item())

            ssim_batch_PS.update(SSIM(y_PS, t.to(device_0)).item())
            ssim_batch_TC.update(SSIM(y_TC, t.to(device_0)).item())
            ssim_batch_BL.update(SSIM(y_BL, t.to(device_1)).item())
            ssim_original_batch.update(SSIM(x, t).item())

            psnr_batch_PS.update(PSNR(MSE(y_PS, t.to(device_0))).item())
            psnr_batch_TC.update(PSNR(MSE(y_TC, t.to(device_0))).item())
            psnr_batch_BL.update(PSNR(MSE(y_BL, t.to(device_1))).item())
            psnr_original_batch.update(PSNR(MSE(x.to(device_1), t.to(device_1))).item())

        # Backpropagate to train model
        optimizer_PS.zero_grad()
        loss_value_PS.backward()
        optimizer_PS.step()

        optimizer_TC.zero_grad()
        loss_value_TC.backward()
        optimizer_TC.step()

        optimizer_BL.zero_grad()
        loss_value_BL.backward()
        optimizer_BL.step()

        if i_batch % 100 == 0:
            Display_Loss = "Loss_Size_PS: %.6f" % loss_batch_PS.val + "\tLoss_Size_TC: %.6f" % loss_batch_TC.val + \
                           "\tLoss_Size_BL: %.6f" % loss_batch_BL.val + "\tLoss_Original: %.6f" % loss_original_batch.val
            Display_SSIM = "SSIM_Size_PS: %.6f" % ssim_batch_PS.val + "\tSSIM_Size_TC: %.6f" % ssim_batch_TC.val + \
                           "\tSSIM_Size_BL: %.6f" % ssim_batch_BL.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
            Display_PSNR = "PSNR_Size_PS: %.6f" % psnr_batch_PS.val + "\tPSNR_Size_TC: %.6f" % psnr_batch_TC.val + \
                           "\tPSNR_Size_BL: %.6f" % psnr_batch_BL.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

        # Free up space in GPU
        del x, y_PS, y_TC, y_BL, t

    Display_Loss = "Loss_Size_PS: %.6f" % loss_batch_PS.avg + "\tLoss_Size_TC: %.6f" % loss_batch_TC.avg + \
                   "\tLoss_Size_BL: %.6f" % loss_batch_BL.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_Size_PS: %.6f" % ssim_batch_PS.avg + "\tSSIM_Size_TC: %.6f" % ssim_batch_TC.avg + \
                   "\tSSIM_Size_BL: %.6f" % ssim_batch_BL.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_Size_PS: %.6f" % psnr_batch_PS.avg + "\tPSNR_Size_TC: %.6f" % psnr_batch_TC.avg + \
                   "\tPSNR_Size_BL: %.6f" % psnr_batch_BL.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    print("\nTotal Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']
        with torch.no_grad():
            y_v_PS = dhdn_PS(x_v.to(device_0))
            y_v_TC = dhdn_TC(x_v.to(device_0))
            y_v_BL = dhdn_BL(x_v.to(device_1))

            loss_batch_val_PS.update(loss_PS(y_v_PS, t_v.to(device_0)).item())
            loss_batch_val_TC.update(loss_TC(y_v_TC, t_v.to(device_0)).item())
            loss_batch_val_BL.update(loss_BL(y_v_BL, t_v.to(device_1)).item())
            loss_original_batch_val.update(loss_BL(x_v.to(device_1), t_v.to(device_1)).item())

            ssim_batch_val_PS.update(SSIM(y_v_PS, t_v.to(device_0)).item())
            ssim_batch_val_TC.update(SSIM(y_v_TC, t_v.to(device_0)).item())
            ssim_batch_val_BL.update(SSIM(y_v_BL, t_v.to(device_1)).item())
            ssim_original_batch_val.update(SSIM(x_v, t_v).item())

            psnr_batch_val_PS.update(PSNR(MSE(y_v_PS, t_v.to(device_0))).item())
            psnr_batch_val_TC.update(PSNR(MSE(y_v_TC, t_v.to(device_0))).item())
            psnr_batch_val_BL.update(PSNR(MSE(y_v_BL, t_v.to(device_1))).item())
            psnr_original_batch_val.update(PSNR(MSE(x_v.to(device_1), t_v.to(device_1))).item())

        # Free up space in GPU
        del x_v, y_v_PS, y_v_TC, y_v_BL, t_v

    Display_Loss = "Loss_Size_PS: %.6f" % loss_batch_val_PS.val + "\tLoss_Size_TC: %.6f" % loss_batch_val_TC.val + \
                   "\tLoss_Size_BL: %.6f" % loss_batch_val_BL.val + "\tLoss_Original: %.6f" % loss_original_batch_val.val
    Display_SSIM = "SSIM_Size_PS: %.6f" % ssim_batch_val_PS.val + "\tSSIM_Size_TC: %.6f" % ssim_batch_val_TC.val + \
                   "\tSSIM_Size_BL: %.6f" % ssim_batch_val_BL.val + "\tSSIM_Original: %.6f" % ssim_original_batch_val.val
    Display_PSNR = "PSNR_Size_PS: %.6f" % psnr_batch_val_PS.val + "\tPSNR_Size_TC: %.6f" % psnr_batch_val_TC.val + \
                   "\tPSNR_Size_BL: %.6f" % psnr_batch_val_BL.val + "\tPSNR_Original: %.6f" % psnr_original_batch_val.val

    print("Validation Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_Batch_PS': loss_batch_PS.avg,
        'Loss_Batch_TC': loss_batch_TC.avg,
        'Loss_Batch_BL': loss_batch_BL.avg,
        'Loss_Val_PS': loss_batch_val_PS.avg,
        'Loss_Val_TC': loss_batch_val_TC.avg,
        'Loss_Val_BL': loss_batch_val_BL.avg,
        'Loss_Original_Train': loss_original_batch.avg,
        'Loss_Original_Val': loss_original_batch_val.avg,
        'SSIM_Batch_PS': ssim_batch_PS.avg,
        'SSIM_Batch_TC': ssim_batch_TC.avg,
        'SSIM_Batch_BL': ssim_batch_BL.avg,
        'SSIM_Val_PS': ssim_batch_val_PS.avg,
        'SSIM_Val_TC': ssim_batch_val_TC.avg,
        'SSIM_Val_BL': ssim_batch_val_BL.avg,
        'SSIM_Original_Train': ssim_original_batch.avg,
        'SSIM_Original_Val': ssim_original_batch_val.avg,
        'PSNR_Batch_PS': psnr_batch_PS.avg,
        'PSNR_Batch_TC': psnr_batch_TC.avg,
        'PSNR_Batch_BL': psnr_batch_BL.avg,
        'PSNR_Val_PS': psnr_batch_val_PS.avg,
        'PSNR_Val_TC': psnr_batch_val_TC.avg,
        'PSNR_Val_BL': psnr_batch_val_BL.avg,
        'PSNR_Original_Train': psnr_original_batch.avg,
        'PSNR_Original_Val': psnr_original_batch_val.avg
    })

    Legend = ['Size_PS_Train', 'Size_TC_Train', 'Size_BL_Train', 'Orig_Train',
              'Size_PS_Val', 'Size_TC_Val', 'Size_BL_Val', 'Orig_Val']

    vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([ssim_batch_PS.avg, ssim_batch_TC.avg, ssim_batch_BL.avg, ssim_original_batch.avg,
                           ssim_batch_val_PS.avg, ssim_batch_val_TC.avg, ssim_batch_val_BL.avg,
                           ssim_original_batch_val.avg]),
        win=vis_window['SSIM_{date}'.format(date=d1)],
        opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([psnr_batch_PS.avg, psnr_batch_TC.avg, psnr_batch_BL.avg, psnr_original_batch.avg,
                           psnr_batch_val_PS.avg, psnr_batch_val_TC.avg, psnr_batch_val_BL.avg,
                           psnr_original_batch_val.avg]),
        win=vis_window['PSNR_{date}'.format(date=d1)],
        opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_batch_PS.reset()
    loss_batch_TC.reset()
    loss_batch_BL.reset()
    loss_original_batch.reset()
    ssim_batch_PS.reset()
    ssim_batch_TC.reset()
    ssim_batch_BL.reset()
    ssim_original_batch.reset()
    psnr_batch_PS.reset()
    psnr_batch_TC.reset()
    psnr_batch_BL.reset()
    psnr_original_batch.reset()
    loss_batch_val_PS.reset()
    loss_batch_val_TC.reset()
    loss_batch_val_BL.reset()
    loss_original_batch_val.reset()
    ssim_batch_val_PS.reset()
    ssim_batch_val_TC.reset()
    ssim_batch_val_BL.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val_PS.reset()
    psnr_batch_val_TC.reset()
    psnr_batch_val_BL.reset()
    psnr_original_batch_val.reset()

    scheduler_PS.step()
    scheduler_TC.step()
    scheduler_BL.step()

    if epoch > 0 and not epoch % 10:
        model_path_PS = dir_current + '/models/compare_up/{date}_dhdn_up_PS_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)
        model_path_TC = dir_current + '/models/compare_up/{date}_dhdn_up_TC_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)
        model_path_BL = dir_current + '/models/compare_up/{date}_dhdn_up_BL_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch)

        torch.save(dhdn_PS.state_dict(), model_path_PS)
        torch.save(dhdn_TC.state_dict(), model_path_TC)
        torch.save(dhdn_BL.state_dict(), model_path_BL)

        state_dict_dhdn_PS = clip_weights(dhdn_PS.state_dict(), k=3, device=device_0)
        state_dict_dhdn_TC = clip_weights(dhdn_TC.state_dict(), k=3, device=device_0)
        state_dict_dhdn_BL = clip_weights(dhdn_BL.state_dict(), k=3, device=device_1)

        dhdn_PS.load_state_dict(state_dict_dhdn_PS)
        dhdn_TC.load_state_dict(state_dict_dhdn_TC)
        dhdn_BL.load_state_dict(state_dict_dhdn_BL)

# Save final model
model_path_PS = dir_current + '/models/compare_up/{date}_dhdn_up_PS_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_TC = dir_current + '/models/compare_up/{date}_dhdn_up_TC_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_BL = dir_current + '/models/compare_up/{date}_dhdn_up_BL_{noise}.pth'.format(date=d1, noise=args.noise)

torch.save(dhdn_PS.state_dict(), model_path_PS)
torch.save(dhdn_TC.state_dict(), model_path_TC)
torch.save(dhdn_BL.state_dict(), model_path_BL)
