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

from utilities.utils import CSVLogger, Logger
from utilities.functions import SSIM, PSNR, generate_loggers

# Hyperparameters
config_path = os.getcwd() + '/configs/config_dhdn_sidd.json'
config = json.load(open(config_path))

today = date.today()  # Date to label the models

path_training = os.getcwd() + '/instances/sidd_np_instances_064.csv'
path_validation = os.getcwd() + '/instances/sidd_np_instances_256.csv'

Result_Path = os.getcwd() + '/results/'
if not os.path.isdir(Result_Path):
    os.mkdir(Result_Path)

if not os.path.isdir(Result_Path + '/' + config['Locations']['Output_File']):
    os.mkdir(Result_Path + '/' + config['Locations']['Output_File'])
sys.stdout = Logger(Result_Path + '/' + config['Locations']['Output_File'] + '/log.log')

# Create the CSV Logger:
File_Name = Result_Path + '/' + config['Locations']['Output_File'] + '/data.csv'
Field_Names = ['Loss_Batch0', 'Loss_Batch1', 'Loss_Val0', 'Loss_Val1', 'Loss_Original_Train', 'Loss_Original_Val',
               'SSIM_Batch0', 'SSIM_Batch1', 'SSIM_Val0', 'SSIM_Val1', 'SSIM_Original_Train', 'SSIM_Original_Val',
               'PSNR_Batch0', 'PSNR_Batch1', 'PSNR_Val0', 'PSNR_Val1', 'PSNR_Original_Train', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
if config['CUDA']['Device0'] != 'None':
    device0 = torch.device(config['CUDA']['Device0'])
else:
    device0 = torch.device("cpu")

if config['CUDA']['Device1'] != 'None':
    device1 = torch.device(config['CUDA']['Device1'])
else:
    device1 = torch.device("cpu")

# Load the models:
encoder_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # vanilla DHDN
encoder_1 = [0, 0, 2, 0, 0, 2, 0, 0, 2]  # Best searched model
bottleneck = [0, 0]
decoder = [0, 0, 0, 0, 0, 0, 0, 0, 0]
dhdn_architecture = encoder_0 + bottleneck + decoder
edhdn_architecture = encoder_1 + bottleneck + decoder

dhdn = DHDN.SharedDHDN(architecture=dhdn_architecture)
edhdn = DHDN.SharedDHDN(architecture=edhdn_architecture)

dhdn.to(device0)
edhdn.to(device1)

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
optimizer0 = torch.optim.Adam(dhdn.parameters(), config['Training']['Learning_Rate'])
optimizer1 = torch.optim.Adam(edhdn.parameters(), config['Training']['Learning_Rate'])
scheduler0 = torch.optim.lr_scheduler.StepLR(optimizer0, 3, 0.5, -1)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss0 = nn.L1Loss().to(device0)
loss1 = nn.L1Loss().to(device1)
MSE0 = nn.MSELoss().to(device0)
MSE1 = nn.MSELoss().to(device1)

# Now, let us define our loggers:
loggers0 = generate_loggers()
loggers1 = generate_loggers()

# Training Batches
loss_batch0, loss_original_batch, ssim_batch0, ssim_original_batch, psnr_batch0, psnr_original_batch = loggers0[0]
loss_batch1, _, ssim_batch1, _, psnr_batch1, _ = loggers1[0]
# Validation Batches
loss_batch_val0, loss_original_batch_val, ssim_batch_val0, ssim_original_batch_val = loggers0[1][0:4]
psnr_batch_val0, psnr_original_batch_val = loggers0[1][4:]

loss_batch_val1, _, ssim_batch_val1, _, psnr_batch_val1, _ = loggers1[1]

# Load the Training and Validation Data:
index_validation = config['Training']['List_Validation']
index_training = [i for i in range(config['Training']['Number_Images']) if i not in index_validation]
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
        y0 = dhdn(x.to(device0))
        y1 = edhdn(x.to(device1))
        t = sample_batch['GT']

        loss_value0 = loss0(y0, t.to(device0))
        loss_value1 = loss1(y1, t.to(device1))
        loss_batch0.update(loss_value0.item())
        loss_batch1.update(loss_value1.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss0(x.to(device0), t.to(device0)).item())

            ssim_batch0.update(SSIM(y0, t.to(device0)).item())
            ssim_batch1.update(SSIM(y1, t.to(device1)).item())
            ssim_original_batch.update(SSIM(x, t).item())

            psnr_batch0.update(PSNR(MSE0(y0, t.to(device0))).item())
            psnr_batch1.update(PSNR(MSE1(y1, t.to(device1))).item())
            psnr_original_batch.update(PSNR(MSE0(x.to(device0), t.to(device0))).item())

        # Backpropagate to train model
        optimizer0.zero_grad()
        loss_value0.backward()
        optimizer0.step()

        optimizer1.zero_grad()
        loss_value1.backward()
        optimizer1.step()

        Display_Loss = "Loss_DHDN: %.6f" % loss_batch0.val + "\tLoss_eDHDN: %.6f" % loss_batch1.val + \
                       "\tLoss_Original: %.6f" % loss_original_batch.val
        Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch0.val + "\tSSIM_eDHDN: %.6f" % ssim_batch1.val + \
                       "\tSSIM_Original: %.6f" % ssim_original_batch.val
        Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch0.val + "\tPSNR_eDHDN: %.6f" % psnr_batch1.val + \
                       "\tPSNR_Original: %.6f" % psnr_original_batch.val

        if i_batch % 25 == 0:
            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

    Display_Loss = "Loss_DHDN: %.6f" % loss_batch0.avg + "\tLoss_eDHDN: %.6f" % loss_batch1.avg + \
                   "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch0.avg + "\tSSIM_eDHDN: %.6f" % ssim_batch1.avg + \
                   "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch0.avg + "\tPSNR_eDHDN: %.6f" % psnr_batch1.avg + \
                   "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    print("\nTotal Training Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

    # Free up space in GPU
    del x, y0, y1, t

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']
        with torch.no_grad():
            y_v0 = dhdn(x_v.to(device0))
            y_v1 = edhdn(x_v.to(device1))

            loss_batch_val0.update(loss0(y_v0, t_v.to(device0)).item())
            loss_batch_val1.update(loss1(y_v1, t_v.to(device1)).item())
            loss_original_batch_val.update(loss0(x_v.to(device0), t_v.to(device0)).item())

            ssim_batch_val0.update(SSIM(y_v0, t_v.to(device0)).item())
            ssim_batch_val1.update(SSIM(y_v1, t_v.to(device1)).item())
            ssim_original_batch_val.update(SSIM(x_v, t_v).item())

            psnr_batch_val0.update(PSNR(MSE0(y_v0, t_v.to(device0))).item())
            psnr_batch_val1.update(PSNR(MSE1(y_v1, t_v.to(device1))).item())
            psnr_original_batch_val.update(PSNR(MSE0(x_v, t_v)).item())

        # Only do up to 10 passes for Validation
        if i_validation > 10:
            break

    Display_Loss = "Loss_DHDN: %.6f" % loss_batch_val0.val + "\tLoss_eDHDN: %.6f" % loss_batch_val1.val + \
                   "\tLoss_Original: %.6f" % loss_original_batch_val.val
    Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch_val0.avg + "\tSSIM_eDHDN: %.6f" % ssim_batch_val1.avg + \
                   "\tSSIM_Original: %.6f" % ssim_original_batch_val.avg
    Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch_val0.avg + "\tPSNR_eDHDN: %.6f" % psnr_batch_val1.avg + \
                   "\tPSNR_Original: %.6f" % psnr_original_batch_val.avg

    print("Validation Data for Epoch: ", epoch)
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

    # Free up space in GPU
    del x_v, y_v0, y_v1, t_v

    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_Batch0': loss_batch0.avg,
        'Loss_Batch1': loss_batch1.avg,
        'Loss_Val0': loss_batch_val0.avg,
        'Loss_Val1': loss_batch_val1.avg,
        'Loss_Original_Train': loss_original_batch.avg,
        'Loss_Original_Val': loss_original_batch_val.avg,
        'SSIM_Batch0': ssim_batch0.avg,
        'SSIM_Batch1': ssim_batch1.avg,
        'SSIM_Val0': ssim_batch_val0.avg,
        'SSIM_Val1': ssim_batch_val1.avg,
        'SSIM_Original_Train': ssim_original_batch.avg,
        'SSIM_Original_Val': ssim_original_batch_val.avg,
        'PSNR_Batch0': psnr_batch0.avg,
        'PSNR_Batch1': psnr_batch1.avg,
        'PSNR_Val0': psnr_batch_val0.avg,
        'PSNR_Val1': psnr_batch_val1.avg,
        'PSNR_Original_Train': psnr_original_batch.avg,
        'PSNR_Original_Val': psnr_original_batch_val.avg
    })

    Legend = ['DHDN_Train', 'eDHDN_Train', 'Train_Orig', 'DHDN_Val', 'eDHDN_Val', 'Val_Orig']

    vis_window['SSIM'] = vis.line(
        X=np.column_stack([epoch] * 6),
        Y=np.column_stack([ssim_batch0.avg, ssim_batch1.avg, ssim_original_batch.avg,
                           ssim_batch_val0.avg, ssim_batch_val1.avg, ssim_original_batch_val.avg]),
        win=vis_window['SSIM'],
        opts=dict(title='SSIM', xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR'] = vis.line(
        X=np.column_stack([epoch] * 6),
        Y=np.column_stack([psnr_batch0.avg, psnr_batch1.avg, psnr_original_batch.avg,
                           psnr_batch_val0.avg, psnr_batch_val1.avg, psnr_original_batch_val.avg]),
        win=vis_window['PSNR'],
        opts=dict(title='PSNR', xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_batch0.reset()
    loss_batch1.reset()
    loss_original_batch.reset()
    ssim_batch0.reset()
    ssim_batch1.reset()
    ssim_original_batch.reset()
    psnr_batch0.reset()
    psnr_batch1.reset()
    psnr_original_batch.reset()
    loss_batch_val0.reset()
    loss_batch_val1.reset()
    loss_original_batch_val.reset()
    ssim_batch_val0.reset()
    ssim_batch_val1.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val0.reset()
    psnr_batch_val1.reset()
    psnr_original_batch_val.reset()

    scheduler0.step()
    scheduler1.step()

d1 = today.strftime("%Y_%m_%d")

model_path0 = os.getcwd() + '/models/{date}_dhdn_ssid.pth'.format(date=d1)
model_path1 = os.getcwd() + '/models/{date}_edhdn_ssid.pth'.format(date=d1)
torch.save(dhdn.state_dict(), model_path0)
torch.save(edhdn.state_dict(), model_path1)
