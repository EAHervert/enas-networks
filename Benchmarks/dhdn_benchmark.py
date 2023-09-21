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
from utilities.functions import SSIM, PSNR, generate_loggers, drop_weights, gaussian_add_weights

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

# Parser
parser = argparse.ArgumentParser(
    prog='DHDN_Benchmark_{date}'.format(date=d1),
    description='Compares Vanilla DHDN to optimized DHDN',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--gaussian', default='-1', type=float)  # Gaussian noise addition for model weight # initialization
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_dhdn.json'
config = json.load(open(config_path))

# Noise Dataset
if args.noise == 'SIDD':
    path_training = dir_current + '/instances/sidd_np_instances_064.csv'
    path_validation = dir_current + '/instances/sidd_np_instances_256.csv'
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
Field_Names = ['Loss_Batch_0', 'Loss_Batch_1', 'Loss_Val_0', 'Loss_Val_1', 'Loss_Original_Train', 'Loss_Original_Val',
               'SSIM_Batch_0', 'SSIM_Batch_1', 'SSIM_Val_0', 'SSIM_Val_1', 'SSIM_Original_Train', 'SSIM_Original_Val',
               'PSNR_Batch_0', 'PSNR_Batch_1', 'PSNR_Val_0', 'PSNR_Val_1', 'PSNR_Original_Train', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the models:
encoder_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # vanilla DHDN
encoder_1 = [0, 0, 2, 0, 0, 2, 0, 0, 2]  # Best searched model
bottleneck, decoder = [0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]
dhdn_architecture = encoder_0 + bottleneck + decoder
edhdn_architecture = encoder_1 + bottleneck + decoder

dhdn = DHDN.SharedDHDN(architecture=dhdn_architecture)
edhdn = DHDN.SharedDHDN(architecture=edhdn_architecture)

dhdn.to(device_0)
edhdn.to(device_1)

if config['Training']['Load_Previous_Model']:
    state_dict_dhdn = torch.load(dir_current + config['Training']['Model_Path_DHDN'], map_location=device_0)
    state_dict_edhdn = torch.load(dir_current + config['Training']['Model_Path_EDHDN'], map_location=device_1)

    if args.drop > 0:
        state_dict_dhdn = drop_weights(state_dict_dhdn, p=args.drop, device=device_0)
        state_dict_edhdn = drop_weights(state_dict_edhdn, p=args.drop, device=device_1)
    if args.gaussian > 0:
        state_dict_dhdn = gaussian_add_weights(state_dict_dhdn, k=args.gaussian, device=device_0)
        state_dict_edhdn = gaussian_add_weights(state_dict_edhdn, k=args.gaussian, device=device_1)

    dhdn.load_state_dict(state_dict_dhdn)
    edhdn.load_state_dict(state_dict_edhdn)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = config['Locations']['Output_File']
vis_window = {'SSIM_{date}'.format(date=d1): None, 'PSNR_{date}'.format(date=d1): None}

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
SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing(),
                                    index_set=index_training)
SIDD_validation = dataset.DatasetSIDD(csv_file=path_validation, index_set=index_validation, raw_images=True)

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
        x_v_raw = validation_batch['NOISY_RAW']
        t_v = validation_batch['GT']
        t_v_raw = validation_batch['GT_RAW']
        with torch.no_grad():
            y_v0 = dhdn(x_v.to(device_0))
            y_v1 = edhdn(x_v.to(device_1))

            y_v0_raw = torch.round(y_v0 * 255)
            y_v1_raw = torch.round(y_v1 * 255)

            loss_batch_val_0.update(loss_0(y_v0, t_v.to(device_0)).item())
            loss_batch_val_1.update(loss_1(y_v1, t_v.to(device_1)).item())
            loss_original_batch_val.update(loss_0(x_v.to(device_0), t_v.to(device_0)).item())

            ssim_batch_val_0.update(SSIM(y_v0_raw, t_v_raw.to(device_0)).item())
            ssim_batch_val_1.update(SSIM(y_v1_raw, t_v_raw.to(device_1)).item())
            ssim_original_batch_val.update(SSIM(x_v_raw, t_v_raw).item())

            psnr_batch_val_0.update(PSNR(MSE(y_v0_raw, t_v_raw.to(device_0))).item())
            psnr_batch_val_1.update(PSNR(MSE(y_v1_raw, t_v_raw.to(device_1))).item())
            psnr_original_batch_val.update(PSNR(MSE(x_v_raw.to(device_0), t_v_raw.to(device_0))).item())

        # Free up space in GPU
        del x_v, y_v0, y_v1, t_v

        # Only do up to 50 passes for Validation
        if i_validation > 50:
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

    vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 6),
        Y=np.column_stack([ssim_batch_0.avg, ssim_batch_1.avg, ssim_original_batch.avg,
                           ssim_batch_val_0.avg, ssim_batch_val_1.avg, ssim_original_batch_val.avg]),
        win=vis_window['SSIM_{date}'.format(date=d1)],
        opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 6),
        Y=np.column_stack([psnr_batch_0.avg, psnr_batch_1.avg, psnr_original_batch.avg,
                           psnr_batch_val_0.avg, psnr_batch_val_1.avg, psnr_original_batch_val.avg]),
        win=vis_window['PSNR_{date}'.format(date=d1)],
        opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
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

if not os.path.exists(dir_current + '/models/'):
    os.makedirs(dir_current + '/models/')

model_path_0 = dir_current + '/models/{date}_dhdn_SIDD.pth'.format(date=d1)
model_path_1 = dir_current + '/models/{date}_edhdn_SIDD.pth'.format(date=d1)
torch.save(dhdn.state_dict(), model_path_0)
torch.save(edhdn.state_dict(), model_path_1)
