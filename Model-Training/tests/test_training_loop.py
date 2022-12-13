import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import visdom
import utilities.utils
import utilities.functions as functions
import ENAS_DHDN.SHARED_DHDN as SHARED_DHDN
import Model_Training.training as training
from utilities.utils import CSVLogger, Logger
import time

config_path = '/Users/esauhervert/PycharmProjects/enas-networks/config.json'
with open(config_path) as f:
    config = json.load(f)

Result_Path = 'Results'
if not os.path.isdir(Result_Path):
    os.mkdir(Result_Path)

if not os.path.isdir(Result_Path + '/' + config['Locations']['Output_File']):
    os.mkdir(Result_Path + '/' + config['Locations']['Output_File'])
sys.stdout = Logger(Result_Path + '/' + config['Locations']['Output_File'] + '/Log.log')

# Create the CSV Logger:
File_Name = Result_Path + '/' + config['Locations']['Output_File'] + '/Data.csv'
Field_Names = ['Loss_Train', 'Loss_Val', 'Loss_Original_Train', 'Loss_Original_Val', 'SSIM_Train', 'SSIM_Val',
               'SSIM_Original_Train', 'SSIM_Original_Val', 'PSNR_Train', 'PSNR_Val', 'PSNR_Original_Train',
               'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

test = functions.image_csv(config_path=config_path)
batches_training = functions.create_batches(test.training_instances, 4)
batches_validation = functions.create_batches(test.validation_instances, 4)

# Model
architecture = np.zeros(3 * 7 - 1, dtype=int).tolist()
DHDN = SHARED_DHDN.SharedDHDN(architecture=architecture)

# Define the devices:
if config['CUDA']['Device']:
    Device = torch.device(config['CUDA']['Device'])
else:
    Device = torch.device('cpu')

# Cast to GPU(s)
if config['CUDA']['DataParallel'] == 1:
    DHDN = nn.DataParallel(DHDN)
DHDN = DHDN.to(Device)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom()

# Display the data to the window:
vis.env = config['Locations']['Output_File']
vis_window = {'DHDN_SSIM': None, 'DHDN_PSNR': None}

# Define the optimizers:
Optimizer = torch.optim.Adam(DHDN.parameters(), config['Training']['Learning_Rate'])

# Define the Scheduling:
Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
Loss = nn.L1Loss().to(Device)
MSE = nn.MSELoss().to(Device)

# Now, let us define our loggers:
loggers = training.generate_loggers()

# Image Batches
loss_batch, loss_original_batch, ssim_batch, ssim_original_batch, psnr_batch, psnr_original_batch = loggers[0]
# Total Training
loss_meter_train, loss_original_meter_train, ssim_meter_train, ssim_original_meter_train = loggers[1][0:4]
psnr_meter_train, psnr_original_meter_train = loggers[1][4:]
# Validation
loss_meter_val, loss_original_meter_val, ssim_meter_val, ssim_original_meter_val = loggers[2][0:4]
psnr_meter_val, psnr_original_meter_val = loggers[2][4:]

epoch = 0
# Now, let us loop through the training Data:
for epoch in [0]:

    t0 = time.time()
    training.train_loop(epoch, config, batches_training, DHDN, architecture, Device, Loss, MSE, loss_meter_train,
                        loss_original_meter_train, ssim_meter_train, ssim_original_meter_train, psnr_meter_train,
                        psnr_original_meter_train, loss_batch, loss_original_batch, ssim_batch, ssim_original_batch,
                        psnr_batch, psnr_original_batch, backpropagate=True, optimizer=Optimizer)

    t1 = time.time()

    print('-' * 160)
    print("Total Training Data for Epoch: ", epoch)
    Display_Loss = "Loss_DHDN: %.6f" % loss_meter_train.avg + \
                   "\tLoss_Original: %.6f" % loss_original_meter_train.avg
    Display_SSIM = "SSIM_DHDN: %.6f" % ssim_meter_train.avg + \
                   "\tSSIM_Original: %.6f" % ssim_original_meter_train.avg
    Display_PSNR = "PSNR_DHDN: %.6f" % psnr_meter_train.avg + \
                   "\tPSNR_Original: %.6f" % psnr_original_meter_train.avg

    print(Display_Loss)
    print(Display_SSIM)
    print(Display_PSNR)

    for param_group in Optimizer.param_groups:
        LR = param_group['lr']
        print("Learning Rate: ", LR)

    training.display_time(t1 - t0)

    t2 = time.time()
    training.train_loop(epoch, config, batches_validation, DHDN, architecture, Device, Loss, MSE, loss_meter_val,
                        loss_original_meter_val, ssim_meter_val, ssim_original_meter_val, psnr_meter_val,
                        psnr_original_meter_val, loss_batch, loss_original_batch, ssim_batch, ssim_original_batch,
                        psnr_batch, psnr_original_batch, backpropagate=False, optimizer=Optimizer, mode='Validation')

    t3 = time.time()

    print('-' * 160)
    print("Validation Data for Epoch: ", epoch)
    Display_Loss = "Loss_DHDN: %.6f" % loss_meter_val.avg + \
                   "\tLoss_Original: %.6f" % loss_original_meter_val.avg
    Display_SSIM = "SSIM_DHDN: %.6f" % ssim_meter_val.avg + \
                   "\tSSIM_Original: %.6f" % ssim_original_meter_val.avg
    Display_PSNR = "PSNR_DHDN: %.6f" % psnr_meter_val.avg + \
                   "\tPSNR_Original: %.6f" % psnr_original_meter_val.avg
    print(Display_Loss)
    print(Display_SSIM)
    print(Display_PSNR)

    training.display_time(t3 - t2)

    print('-' * 160)
    print()

    Logger.writerow({
        'Loss_Train': loss_meter_train.avg,
        'Loss_Val': loss_meter_val.avg,
        'Loss_Original_Train': loss_original_meter_train.avg,
        'Loss_Original_Val': loss_original_meter_val.avg,
        'SSIM_Train': ssim_meter_train.avg,
        'SSIM_Val': ssim_meter_val.avg,
        'SSIM_Original_Train': ssim_original_meter_train.avg,
        'SSIM_Original_Val': ssim_original_meter_val.avg,
        'PSNR_Train': psnr_meter_train.avg,
        'PSNR_Val': psnr_meter_val.avg,
        'PSNR_Original_Train': psnr_original_meter_train.avg,
        'PSNR_Original_Val': psnr_original_meter_val.avg
    })

    Legend = ['Train', 'Val', 'Original_Train', 'Original_Val']

    vis_window['DHDN_SSIM'] = vis.line(
        X=np.column_stack([epoch] * 4),
        Y=np.column_stack([ssim_meter_train.avg, ssim_meter_val.avg,
                           ssim_original_meter_train.avg, ssim_original_meter_val.avg]),
        win=vis_window['DHDN_SSIM'],
        opts=dict(title='DHDN_SSIM', xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['DHDN_PSNR'] = vis.line(
        X=np.column_stack([epoch] * 4),
        Y=np.column_stack([psnr_meter_train.avg, psnr_meter_val.avg,
                           psnr_original_meter_train.avg, psnr_original_meter_val.avg]),
        win=vis_window['DHDN_PSNR'],
        opts=dict(title='DHDN_PSNR', xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    # Now, let us reset our loggers:
    loss_meter_train.reset()
    loss_meter_val.reset()
    loss_original_meter_train.reset()
    loss_original_meter_val.reset()
    ssim_meter_train.reset()
    ssim_meter_val.reset()
    ssim_original_meter_train.reset()
    ssim_original_meter_val.reset()
    psnr_meter_train.reset()
    psnr_meter_val.reset()
    psnr_original_meter_train.reset()
    psnr_original_meter_val.reset()
