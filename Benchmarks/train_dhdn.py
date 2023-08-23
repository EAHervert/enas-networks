import os
import sys
import dataset
import dhdn
import time
from datetime import date
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import visdom

# from utilities.functions import list_instances, display_time, create_batches
from utilities.utils import CSVLogger, Logger

# Hyperparameters
config_path = os.getcwd() + '/configs/config_sidd.json'
config = json.load(open(config_path))

today = date.today()  # Date to label the models

path_training = os.getcwd() + '/instances/instances_064.csv'
path_validation = os.getcwd() + '/instances/instances_256.csv'

Result_Path = os.getcwd() + '/results/'
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
    device = torch.device(config['CUDA']['Device'])
else:
    device = torch.device("cpu")

# Load the model:
dhdn = dhdn.Net()
dhdn.to(device)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = config['Locations']['Output_File']
vis_window = {'DHDN_SSIM': None, 'DHDN_PSNR': None}

# Training Optimization and Scheduling:
optimizer = torch.optim.Adam(dhdn.parameters(), config['Training']['Learning_Rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss = nn.L1Loss().to(device)
MSE = nn.MSELoss().to(device)

# Now, let us define our loggers:
# loggers = generate_loggers()

# # Image Batches
# loss_batch, loss_original_batch, ssim_batch, ssim_original_batch, psnr_batch, psnr_original_batch = loggers[0]
# # Total Training
# loss_meter_train, loss_original_meter_train, ssim_meter_train, ssim_original_meter_train = loggers[1][0:4]
# psnr_meter_train, psnr_original_meter_train = loggers[1][4:]
# # Validation
# loss_meter_val, loss_original_meter_val, ssim_meter_val, ssim_original_meter_val = loggers[2][0:4]
# psnr_meter_val, psnr_original_meter_val = loggers[2][4:]

# Load the Training and Validation Data:
SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing())
SIDD_validation = dataset.DatasetSIDD(csv_file=path_validation, transform=dataset.RandomProcessing())

if torch.cuda.is_available():
    dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=64, shuffle=True, num_workers=16)
    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=32, shuffle=True, num_workers=8)
else:
    dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=16, shuffle=True, num_workers=0)
    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=4, shuffle=True, num_workers=0)

t_init = time.time()

for epoch in range(config['Training']['Epochs']):
    loss_val_epoch = 0
    mse_val_validation = 0
    for i_batch, sample_batch in enumerate(dataloader_sidd_training):
        x = sample_batch['NOISY'].to(device)
        y = dhdn(x)
        loss_val = loss(y, sample_batch['GT'].to(device))
        print('EPOCH:', epoch, 'ITERATION:', i_batch, 'Loss', loss_val.item())

        index = i_batch + 1
        loss_val_epoch = ((index - 1) * loss_val_epoch + loss_val.item()) / index
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    print('\nEPOCH:', epoch, 'Loss:', loss_val_epoch)

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x = validation_batch['NOISY'].to(device)
        with torch.no_grad():
            y = dhdn(x)
            MSE_val = MSE(y, validation_batch['GT'].to(device))

        index = i_validation + 1
        mse_val_validation = ((index - 1) * mse_val_validation + MSE_val.item()) / index
        break  # Only do one pass for Validation

    print('\nEPOCH:', epoch, 'Validation MSE:', mse_val_validation)

d1 = today.strftime("%Y_%m_%d")

model_path = os.getcwd() + '/models/{date}_sidd_dhdn.pth'.format(date=d1)
torch.save(dhdn.state_dict(), model_path)
