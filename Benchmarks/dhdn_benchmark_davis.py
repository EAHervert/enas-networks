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
    prog='DHDN_DAVIS',
    description='Runs DHDN Denoising Comparison for DAVIS Dataset',
)
parser.add_argument('Noise')  # positional argument
args = parser.parse_args()

# Hyperparameters
config_path = os.getcwd() + '/configs/config_dhdn_davis.json'
config = json.load(open(config_path))

config['Locations']['Output_File'] += '_' + str(args.Noise)

today = date.today()  # Date to label the models

path_training = os.getcwd() + config['Locations']['Training_File']
path_validation = os.getcwd() + config['Locations']['Validation_File']

Result_Path = os.getcwd() + '/results/'
if not os.path.isdir(Result_Path):
    os.mkdir(Result_Path)
d1 = today.strftime("%Y_%m_%d")

if not os.path.isdir(Result_Path + '/' + config['Locations']['Output_File']):
    os.mkdir(Result_Path + '/' + config['Locations']['Output_File'])
sys.stdout = Logger(Result_Path + '/' + config['Locations']['Output_File'] + '/log.log')

# Create the CSV Logger:
File_Name = Result_Path + '/' + config['Locations']['Output_File'] + '/data.csv'
Field_Names = ['Loss_Batch_0', 'Loss_Batch_1', 'Loss_Val_0', 'Loss_Val_1', 'Loss_Original_Train', 'Loss_Original_Val',
               'SSIM_Batch_0', 'SSIM_Batch_1', 'SSIM_Val_0', 'SSIM_Val_1', 'SSIM_Original_Train', 'SSIM_Original_Val',
               'PSNR_Batch_0', 'PSNR_Batch_1', 'PSNR_Val_0', 'PSNR_Val_1', 'PSNR_Original_Train', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
if config['CUDA']['Device0'] != 'None':
    device_0 = torch.device(config['CUDA']['Device0'])
else:
    device_0 = torch.device("cpu")

if config['CUDA']['Device1'] != 'None':
    device_1 = torch.device(config['CUDA']['Device1'])
else:
    device_1 = torch.device("cpu")

# Load the models:
encoder_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # vanilla DHDN
encoder_1 = [0, 0, 2, 0, 0, 2, 0, 0, 2]  # Best searched model
bottleneck = [0, 0]
decoder = [0, 0, 0, 0, 0, 0, 0, 0, 0]
dhdn_architecture = encoder_0 + bottleneck + decoder
edhdn_architecture = encoder_1 + bottleneck + decoder

dhdn = DHDN.SharedDHDN(architecture=dhdn_architecture)
edhdn = DHDN.SharedDHDN(architecture=edhdn_architecture)

dhdn.to(device_0)
edhdn.to(device_1)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = config['Locations']['Output_File']
vis_window = {'SSIM': None, 'PSNR': None}

# Training Optimization and Scheduling:
optimizer_0 = torch.optim.Adam(dhdn.parameters(), config['Training']['Learning_Rate'])
optimizer_1 = torch.optim.Adam(edhdn.parameters(), config['Training']['Learning_Rate'])
scheduler_0 = torch.optim.lr_scheduler.StepLR(optimizer_0, 3, 0.5, -1)
scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_0 = nn.L1Loss().to(device_0)
loss_1 = nn.L1Loss().to(device_1)
MSE = nn.MSELoss().to(device_0)

# Now, let us define our loggers:
loggers0 = generate_loggers()
loggers1 = generate_loggers()

# Training Batches
loss_batch_0, loss_original_batch, ssim_batch_0, ssim_original_batch, psnr_batch_0, psnr_original_batch = loggers0[0]
loss_batch_1, _, ssim_batch_1, _, psnr_batch_1, _ = loggers1[0]

# Validation Batches
loss_batch_val_0, loss_original_batch_val, ssim_batch_val_0, ssim_original_batch_val = loggers0[1][0:4]
psnr_batch_val_0, psnr_original_batch_val = loggers0[1][4:]

loss_batch_val_1, _, ssim_batch_val_1, _, psnr_batch_val_1, _ = loggers1[1]

# Load the Training and Validation Data:
index_validation = config['Training']['List_Validation']
index_training = [i for i in range(config['Training']['Number_Images']) if i not in index_validation]
SIDD_training = dataset.DatasetDAVIS(csv_file=path_training, noise_choice=str(args.Noise),
                                     transform=dataset.RandomProcessing(), index_set=index_training)
SIDD_validation = dataset.DatasetDAVIS(csv_file=path_validation, noise_choice=str(args.Noise),
                                       index_set=index_validation)

dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=config['Training']['Train_Batch_Size'],
                                      shuffle=True, num_workers=16)
dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=config['Training']['Validation_Batch_Size'],
                                        shuffle=False, num_workers=8)

for epoch in range(config['Training']['Epochs']):
    for i_batch, sample_batch in enumerate(dataloader_sidd_training):
        if i_batch > 1000:
            break  # Only train on 1000 random image samples each epoch

        x = sample_batch['NOISY']
        y0 = dhdn(x.to(device_0))
        y1 = edhdn(x.to(device_1))
        t = sample_batch['GT']

        loss_value_0 = loss_0(y0, t.to(device_0))
        loss_value_1 = loss_1(y1, t.to(device_1))
        loss_batch_0.update(loss_value_0.item())
        loss_batch_1.update(loss_value_1.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss_0(x.to(device_0), t.to(device_0)).item())

            ssim_batch_0.update(SSIM(y0, t.to(device_0)).item())
            ssim_batch_1.update(SSIM(y1, t.to(device_1)).item())
            ssim_original_batch.update(SSIM(x, t).item())

            psnr_batch_0.update(PSNR(MSE(y0, t.to(device_0))).item())
            psnr_batch_1.update(PSNR(MSE(y1.to(device_0), t.to(device_0))).item())
            psnr_original_batch.update(PSNR(MSE(x.to(device_0), t.to(device_0))).item())

        # Backpropagate to train model
        optimizer_0.zero_grad()
        loss_value_0.backward()
        optimizer_0.step()

        optimizer_1.zero_grad()
        loss_value_1.backward()
        optimizer_1.step()

        if i_batch % 50 == 0:
            Display_Loss = "Loss_DHDN: %.6f" % loss_batch_0.val + "\tLoss_eDHDN: %.6f" % loss_batch_1.val + \
                           "\tLoss_Original: %.6f" % loss_original_batch.val
            Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch_0.val + "\tSSIM_eDHDN: %.6f" % ssim_batch_1.val + \
                           "\tSSIM_Original: %.6f" % ssim_original_batch.val
            Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch_0.val + "\tPSNR_eDHDN: %.6f" % psnr_batch_1.val + \
                           "\tPSNR_Original: %.6f" % psnr_original_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

        # Free up space in GPU
        del x, y0, y1, t

    Display_Loss = "Loss_DHDN: %.6f" % loss_batch_0.avg + "\tLoss_eDHDN: %.6f" % loss_batch_1.avg + \
                   "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch_0.avg + "\tSSIM_eDHDN: %.6f" % ssim_batch_1.avg + \
                   "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch_0.avg + "\tPSNR_eDHDN: %.6f" % psnr_batch_1.avg + \
                   "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    print("\nTotal Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']
        with torch.no_grad():
            y_v0 = dhdn(x_v.to(device_0))
            y_v1 = edhdn(x_v.to(device_1))

            loss_batch_val_0.update(loss_0(y_v0, t_v.to(device_0)).item())
            loss_batch_val_1.update(loss_1(y_v1, t_v.to(device_1)).item())
            loss_original_batch_val.update(loss_0(x_v.to(device_0), t_v.to(device_0)).item())

            ssim_batch_val_0.update(SSIM(y_v0, t_v.to(device_0)).item())
            ssim_batch_val_1.update(SSIM(y_v1, t_v.to(device_1)).item())
            ssim_original_batch_val.update(SSIM(x_v, t_v).item())

            psnr_batch_val_0.update(PSNR(MSE(y_v0, t_v.to(device_0))).item())
            psnr_batch_val_1.update(PSNR(MSE(y_v1.to(device_0), t_v.to(device_0))).item())
            psnr_original_batch_val.update(PSNR(MSE(x_v.to(device_0), t_v.to(device_0))).item())

        # Free up space in GPU
        del x_v, y_v0, y_v1, t_v

        # Only do up to 25 passes for Validation
        if i_validation > 25:
            break

    Display_Loss = "Loss_DHDN: %.6f" % loss_batch_val_0.val + "\tLoss_eDHDN: %.6f" % loss_batch_val_1.val + \
                   "\tLoss_Original: %.6f" % loss_original_batch_val.val
    Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch_val_0.avg + "\tSSIM_eDHDN: %.6f" % ssim_batch_val_1.avg + \
                   "\tSSIM_Original: %.6f" % ssim_original_batch_val.avg
    Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch_val_0.avg + "\tPSNR_eDHDN: %.6f" % psnr_batch_val_1.avg + \
                   "\tPSNR_Original: %.6f" % psnr_original_batch_val.avg

    print("Validation Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_Batch_0': loss_batch_0.avg,
        'Loss_Batch_1': loss_batch_1.avg,
        'Loss_Val_0': loss_batch_val_0.avg,
        'Loss_Val_1': loss_batch_val_1.avg,
        'Loss_Original_Train': loss_original_batch.avg,
        'Loss_Original_Val': loss_original_batch_val.avg,
        'SSIM_Batch_0': ssim_batch_0.avg,
        'SSIM_Batch_1': ssim_batch_1.avg,
        'SSIM_Val_0': ssim_batch_val_0.avg,
        'SSIM_Val_1': ssim_batch_val_1.avg,
        'SSIM_Original_Train': ssim_original_batch.avg,
        'SSIM_Original_Val': ssim_original_batch_val.avg,
        'PSNR_Batch_0': psnr_batch_0.avg,
        'PSNR_Batch_1': psnr_batch_1.avg,
        'PSNR_Val_0': psnr_batch_val_0.avg,
        'PSNR_Val_1': psnr_batch_val_1.avg,
        'PSNR_Original_Train': psnr_original_batch.avg,
        'PSNR_Original_Val': psnr_original_batch_val.avg
    })

    Legend = ['DHDN_Train', 'eDHDN_Train', 'Orig_Train', 'DHDN_Val', 'eDHDN_Val', 'Orig_Val']

    vis_window['SSIM'] = vis.line(
        X=np.column_stack([epoch] * 6),
        Y=np.column_stack([ssim_batch_0.avg, ssim_batch_1.avg, ssim_original_batch.avg,
                           ssim_batch_val_0.avg, ssim_batch_val_1.avg, ssim_original_batch_val.avg]),
        win=vis_window['SSIM'],
        opts=dict(title='SSIM', xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR'] = vis.line(
        X=np.column_stack([epoch] * 6),
        Y=np.column_stack([psnr_batch_0.avg, psnr_batch_1.avg, psnr_original_batch.avg,
                           psnr_batch_val_0.avg, psnr_batch_val_1.avg, psnr_original_batch_val.avg]),
        win=vis_window['PSNR'],
        opts=dict(title='PSNR', xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_batch_0.reset()
    loss_batch_1.reset()
    loss_original_batch.reset()
    ssim_batch_0.reset()
    ssim_batch_1.reset()
    ssim_original_batch.reset()
    psnr_batch_0.reset()
    psnr_batch_1.reset()
    psnr_original_batch.reset()
    loss_batch_val_0.reset()
    loss_batch_val_1.reset()
    loss_original_batch_val.reset()
    ssim_batch_val_0.reset()
    ssim_batch_val_1.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val_0.reset()
    psnr_batch_val_1.reset()
    psnr_original_batch_val.reset()

    scheduler_0.step()
    scheduler_1.step()

d1 = today.strftime("%Y_%m_%d")

model_path_0 = os.getcwd() + '/models/{date}_dhdn_davis_{noise}.pth'.format(date=d1, noise=str(args.Noise))
model_path_1 = os.getcwd() + '/models/{date}_edhdn_davis_{noise}.pth'.format(date=d1, noise=str(args.Noise))
torch.save(dhdn.state_dict(), model_path_0)
torch.save(edhdn.state_dict(), model_path_1)
