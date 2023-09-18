import os
import sys
from utilities import dataset
from ENAS_DHDN import SHARED_DHDN as DHDN
from datetime import date
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import visdom
import argparse

from utilities.utils import CSVLogger, Logger
from utilities.functions import SSIM, PSNR, generate_loggers

# Parser
parser = argparse.ArgumentParser(
    prog='DHDN_Compare_Size',
    description='Compares 3 sizes of models based on the DHDN architecture',
)
parser.add_argument('Noise')  # positional argument
args = parser.parse_args()

# Config
config_path = os.getcwd() + '/configs/config_compare_size.json'
config = json.load(open(config_path))
config['Locations']['Output_File'] += '_' + str(args.Noise)

today = date.today()  # Date to label the models

# Noise Dataset
if args.Noise == 'SSID':
    path_training = os.getcwd() + '/instances/sidd_np_instances_064.csv'
    path_validation = os.getcwd() + '/instances/sidd_np_instances_256.csv'
    Result_Path = os.getcwd() + '/SIDD/'
elif args.Noise in ['GAUSSIAN_10', 'GAUSSIAN_25', 'GAUSSIAN_50', 'RAIN', 'SALT_PEPPER', 'MIXED']:
    path_training = os.getcwd() + '/instances/davis_np_instances_128.csv'
    path_validation = os.getcwd() + '/instances/davis_np_instances_256.csv'
    Result_Path = os.getcwd() + '/{noise}/'.format(noise=args.Noise)
else:
    print('Incorrect Noise Selection!')
    exit()

if not os.path.isdir(Result_Path):
    os.mkdir(Result_Path)

if not os.path.isdir(Result_Path + '/' + config['Locations']['Output_File']):
    os.mkdir(Result_Path + '/' + config['Locations']['Output_File'])
sys.stdout = Logger(Result_Path + '/' + config['Locations']['Output_File'] + 'log.log')

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

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = 'DHDN_Compare_Size_' + str(args.Noise)
vis_window = {'SSIM': None, 'PSNR': None}

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
if args.Noise == 'SSID':
    index_validation = config['Training']['List_Validation_SSID']
    index_training = [i for i in range(config['Training']['Number_Images_SSID']) if i not in index_validation]
else:
    index_validation = config['Training']['List_Validation_DAVIS']
    index_training = [i for i in range(config['Training']['Number_Images_DAVIS']) if i not in index_validation]

SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing(),
                                    index_set=index_training)
SIDD_validation = dataset.DatasetSIDD(csv_file=path_validation, index_set=index_validation)

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

        Display_Loss = "Loss_Size_5: %.6f" % loss_batch_5.val + "\tLoss_Size_7: %.6f" % loss_batch_7.val + \
                       "\tLoss_Size_9: %.6f" % loss_batch_9.val + "\tLoss_Original: %.6f" % loss_original_batch.val
        Display_SSIM = "SSIM_Size_5: %.6f" % ssim_batch_5.val + "\tSSIM_Size_7: %.6f" % ssim_batch_7.val + \
                       "\tSSIM_Size_9: %.6f" % ssim_batch_9.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
        Display_PSNR = "PSNR_Size_5: %.6f" % psnr_batch_5.val + "\tPSNR_Size_7: %.6f" % psnr_batch_7.val + \
                       "\tPSNR_Size_9: %.6f" % psnr_batch_9.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

        if i_batch % 50 == 0:
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

        # Only do up to 50 passes for Validation
        if i_validation > 50:
            break

    Display_Loss = "Loss_Size_5: %.6f" % loss_batch_val_5.val + "\tLoss_Size_7: %.6f" % loss_batch_val_7.val + \
                   "\tLoss_Size_9: %.6f" % loss_batch_val_9.val + "\tLoss_Original: %.6f" % loss_original_batch_val.val
    Display_SSIM = "SSIM_Size_5: %.6f" % ssim_batch_val_5.val + "\tSSIM_Size_7: %.6f" % ssim_batch_val_7.val + \
                   "\tSSIM_Size_9: %.6f" % ssim_batch_val_9.val + "\tSSIM_Original: %.6f" % ssim_original_batch_val.val
    Display_PSNR = "PSNR_Size_5: %.6f" % psnr_batch_val_5.val + "\tPSNR_Size_7: %.6f" % psnr_batch_val_7.val + \
                   "\tPSNR_Size_9: %.6f" % psnr_batch_val_9.val + "\tPSNR_Original: %.6f" % psnr_original_batch_val.val

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

    vis_window['SSIM'] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([ssim_batch_5.avg, ssim_batch_7.avg, ssim_batch_9.avg, ssim_original_batch.avg,
                           ssim_batch_val_5.avg, ssim_batch_val_7.avg,  ssim_batch_val_9.avg,
                           ssim_original_batch_val.avg]),
        win=vis_window['SSIM'],
        opts=dict(title='SSIM', xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR'] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([psnr_batch_5.avg, psnr_batch_7.avg, psnr_batch_9.avg, psnr_original_batch.avg,
                           psnr_batch_val_5.avg, psnr_batch_val_7.avg, psnr_batch_val_9.avg,
                           psnr_original_batch_val.avg]),
        win=vis_window['PSNR'],
        opts=dict(title='PSNR', xlabel='Epoch', ylabel='PSNR', legend=Legend),
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

d1 = today.strftime("%Y_%m_%d")

if not os.path.exists(os.getcwd() + '/models/'):
    os.makedirs(os.getcwd() + '/models/')

model_path_5 = os.getcwd() + '/models/{date}_dhdn_size_5_{noise}.pth'.format(date=d1, noise=args.Noise)
model_path_7 = os.getcwd() + '/models/{date}_dhdn_size_7_{noise}.pth'.format(date=d1, noise=args.Noise)
model_path_9 = os.getcwd() + '/models/{date}_dhdn_size_9_{noise}.pth'.format(date=d1, noise=args.Noise)
torch.save(dhdn_5.state_dict(), model_path_5)
torch.save(dhdn_7.state_dict(), model_path_7)
torch.save(dhdn_9.state_dict(), model_path_9)
