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
batches = functions.create_batches(test.training_instances, 4)

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
loop = training.train_loop()
