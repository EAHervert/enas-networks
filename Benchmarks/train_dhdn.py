import os
import sys
import dataset
import dhdn
import time
from datetime import date
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import visdom

# from utilities.functions import list_instances, display_time, create_batches
from utilities.utils import CSVLogger, Logger
from utilities.functions import SSIM, PSNR, generate_loggers

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
loggers = generate_loggers()

# # Image Batches
loss_batch, loss_original_batch, ssim_batch, ssim_original_batch, psnr_batch, psnr_original_batch = loggers[0]
# # Validation
loss_batch_val, loss_original_batch_val, ssim_batch_val, ssim_original_batch_val = loggers[2][0:4]
psnr_batch_val, psnr_original_batch_val = loggers[0][4:]

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
    for i_batch, sample_batch in enumerate(dataloader_sidd_training):
        x = sample_batch['NOISY'].to(device)
        y = dhdn(x)
        t = sample_batch['GT'].to(device)
        loss_value = loss(y, t)
        loss_batch.update(loss_value.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss(x, t).item())
            ssim_batch.update(SSIM(y, t).item())
            ssim_original_batch.update(SSIM(x, t).item())
            psnr_batch.update(PSNR(MSE(y, t)).item())
            psnr_original_batch.update(PSNR(MSE(x, t)).item())

        # Backpropagate to train model
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        Display_Loss = "Loss_DHDN: %.6f" % loss_batch.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
        Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
        Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

        print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
        print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

    Display_Loss = "Loss_DHDN: %.6f" % loss_batch.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    print("Total Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x = validation_batch['NOISY'].to(device)
        t = validation_batch['GT'].to(device)
        with torch.no_grad():
            y = dhdn(x)
            loss_batch_val.update(loss(y, t).item())
            loss_original_batch_val.update(loss(x, t).item())
            ssim_batch_val.update(SSIM(y, t).item())
            ssim_original_batch_val.update(SSIM(x, t).item())
            psnr_batch_val.update(PSNR(MSE(y, t)).item())
            psnr_original_batch_val.update(PSNR(MSE(x, t)).item())

        # Only do up to 3 passes for Validation
        if i_validation > 3:
            break

    Display_Loss = "Loss_DHDN: %.6f" % loss_batch_val.avg + \
                   "\tLoss_Original: %.6f" % loss_original_batch_val.avg
    Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch_val.avg + \
                   "\tSSIM_Original: %.6f" % ssim_original_batch_val.avg
    Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch_val.avg + \
                   "\tPSNR_Original: %.6f" % psnr_original_batch_val.avg
    print("Validation Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_Train': loss_batch.avg,
        'Loss_Val': loss_batch_val.avg,
        'Loss_Original_Train': loss_original_batch.avg,
        'Loss_Original_Val': loss_original_batch_val.avg,
        'SSIM_Train': ssim_batch.avg,
        'SSIM_Val': ssim_batch_val.avg,
        'SSIM_Original_Train': ssim_original_batch.avg,
        'SSIM_Original_Val': ssim_original_batch_val.avg,
        'PSNR_Train': psnr_batch.avg,
        'PSNR_Val': psnr_batch_val.avg,
        'PSNR_Original_Train': psnr_original_batch.avg,
        'PSNR_Original_Val': psnr_original_batch_val.avg
    })

    Legend = ['Train', 'Val', 'Original_Train', 'Original_Val']

    vis_window['DHDN_SSIM'] = vis.line(
        X=np.column_stack([epoch] * 4),
        Y=np.column_stack([ssim_batch.avg, ssim_batch_val.avg,
                           ssim_original_batch.avg, ssim_original_batch_val.avg]),
        win=vis_window['DHDN_SSIM'],
        opts=dict(title='DHDN_SSIM', xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['DHDN_PSNR'] = vis.line(
        X=np.column_stack([epoch] * 4),
        Y=np.column_stack([psnr_batch.avg, psnr_batch_val.avg,
                           psnr_original_batch.avg, psnr_original_batch_val.avg]),
        win=vis_window['DHDN_PSNR'],
        opts=dict(title='DHDN_PSNR', xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_batch.reset()
    loss_original_batch.reset()
    ssim_batch.reset()
    ssim_original_batch.reset()
    psnr_batch.reset()
    psnr_original_batch.reset()
    loss_batch_val.reset()
    loss_original_batch_val.reset()
    ssim_batch_val.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val.reset()
    psnr_original_batch_val.reset()

d1 = today.strftime("%Y_%m_%d")

model_path = os.getcwd() + '/models/{date}_sidd_dhdn.pth'.format(date=d1)
torch.save(dhdn.state_dict(), model_path)
