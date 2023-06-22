import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from ENAS_DHDN.SHARED_DHDN import SharedDHDN
from utilities.utils import CSVLogger, Logger

# Load the config file
config_path = '/scratch1/eah170630/enas-networks/config_remote.json'
config = json.load(open(config_path))

# Create directories
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

# Define the devices:
if config['CUDA']['Device'] != 'None':
    Device = torch.device(config['CUDA']['Device'])

print(Device)

# Load the Models:
# Architecture and Size parameters:
architecture = np.zeros(3 * 7 - 1, dtype=int).tolist()
DHDN = SharedDHDN(architecture=architecture)

# Cast to GPU(s)
if config['CUDA']['DataParallel'] == 1:
    DHDN = nn.DataParallel(DHDN)
DHDN = DHDN.to(Device)

print(DHDN)
