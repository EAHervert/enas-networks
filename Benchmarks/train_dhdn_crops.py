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
    prog='DHDN_Benchmark_{date}'.format(date=d1),
    description='Compares Vanilla DHDN to optimized DHDN',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
parser.add_argument('--model_path_dhdn', default='2023_09_11_dhdn_SIDD.pth', type=str)  # Model path dhdn
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_dhdn.json'
config = json.load(open(config_path))
if not os.path.exists(dir_current + '/models/'):
    os.makedirs(dir_current + '/models/')

# Noise Dataset
if args.noise == 'SIDD':
    path_validation_noisy = dir_current + config['Locations']['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations']['Validation_GT']
    Result_Path = dir_current + '/SIDD'
else:
    print('Incorrect Noise Selection!')
    exit()

if not os.path.isdir(Result_Path):
    os.mkdir(Result_Path)

out_path = Result_Path + '/' + config['Locations']['Output_File'] + '/' + str(d1)
if not os.path.isdir(out_path):
    os.mkdir(out_path)
sys.stdout = Logger(out_path + '/log.log')

# Create the CSV Logger:
File_Name = out_path + '/data.csv'
Field_Names = ['Loss_Batch_0', 'Loss_Val_0', 'Loss_Original_Train', 'Loss_Original_Val',
               'SSIM_Batch_0', 'SSIM_Val_0', 'SSIM_Original_Train', 'SSIM_Original_Val',
               'PSNR_Batch_0', 'PSNR_Val_0', 'PSNR_Original_Train', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the models:
encoder_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # vanilla DHDN
bottleneck, decoder = [0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]
dhdn_architecture = encoder_0 + bottleneck + decoder

dhdn = DHDN.SharedDHDN(architecture=dhdn_architecture, outer_sum=args.outer_sum)

dhdn.to(device_0)

model_dhdn_path = '/models/' + args.model_path_dhdn
if args.load_models:
    state_dict_dhdn = torch.load(dir_current + model_dhdn_path, map_location=device_0)

    if args.drop > 0:
        state_dict_dhdn = drop_weights(state_dict_dhdn, p=args.drop, device=device_0)
    if args.gaussian > 0:
        state_dict_dhdn = gaussian_add_weights(state_dict_dhdn, k=args.gaussian, device=device_0)

    dhdn.load_state_dict(state_dict_dhdn)

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
scheduler_0 = torch.optim.lr_scheduler.StepLR(optimizer_0, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_0 = nn.L1Loss().to(device_0)
MSE = nn.MSELoss().to(device_0)

# Now, let us define our loggers:
loggers0 = generate_loggers()

# Training Batches
loss_batch_0, loss_original_batch, ssim_batch_0, ssim_original_batch, psnr_batch_0, psnr_original_batch = loggers0[0]

# Validation Batches
loss_batch_val_0, loss_original_batch_val, ssim_batch_val_0, ssim_original_batch_val = loggers0[1][0:4]
psnr_batch_val_0, psnr_original_batch_val = loggers0[1][4:]


# Load the Training and Validation Data:
SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy, mat_gt_file=path_validation_gt)
dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=config['Training']['Validation_Batch_Size'],
                                        shuffle=False, num_workers=8)
epochs_set = [8, 7, 5, 4, 3, 3]
csv_files = ['sidd_np_instances_064_16.csv',
             'sidd_np_instances_064_32.csv',
             'sidd_np_instances_064_64.csv',
             'sidd_np_instances_064_128.csv',
             'sidd_np_instances_064_256.csv',
             'sidd_np_instances_064_512.csv']
for epoch_set, csv_path in zip(epochs_set, csv_files):
    path_training = dir_current + '/instances/' + csv_path

    SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing())
    dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=config['Training']['Train_Batch_Size'],
                                          shuffle=True, num_workers=16)
    for epoch in range(epoch_set):
        for i_batch, sample_batch in enumerate(dataloader_sidd_training):
            x = sample_batch['NOISY']
            y0 = dhdn(x.to(device_0))
            t = sample_batch['GT']

            loss_value_0 = loss_0(y0, t.to(device_0))
            loss_batch_0.update(loss_value_0.item())

            # Calculate values not needing to be backpropagated
            with torch.no_grad():
                loss_original_batch.update(loss_0(x.to(device_0), t.to(device_0)).item())

                ssim_batch_0.update(SSIM(y0, t.to(device_0)).item())
                ssim_original_batch.update(SSIM(x, t).item())

                psnr_batch_0.update(PSNR(MSE(y0, t.to(device_0))).item())
                psnr_original_batch.update(PSNR(MSE(x.to(device_0), t.to(device_0))).item())

            # Backpropagate to train model
            optimizer_0.zero_grad()
            loss_value_0.backward()
            optimizer_0.step()

            if i_batch % 100 == 0:
                Display_Loss = "Loss_DHDN: %.6f" % loss_batch_0.val + "\tLoss_eDHDN: %.6f" % loss_batch_1.val + \
                               "\tLoss_Original: %.6f" % loss_original_batch.val
                Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch_0.val + "\tSSIM_eDHDN: %.6f" % ssim_batch_1.val + \
                               "\tSSIM_Original: %.6f" % ssim_original_batch.val
                Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch_0.val + "\tPSNR_eDHDN: %.6f" % psnr_batch_1.val + \
                               "\tPSNR_Original: %.6f" % psnr_original_batch.val

                print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
                print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

            # Free up space in GPU
            del x, y0, t

        Display_Loss = "Loss_DHDN: %.6f" % loss_batch_0.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
        Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch_0.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
        Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch_0.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

        print("\nTotal Training Data for Epoch: ", epoch)
        print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

        for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
            x_v = validation_batch['NOISY']
            t_v = validation_batch['GT']
            with torch.no_grad():
                y_v0 = dhdn(x_v.to(device_0))

                loss_batch_val_0.update(loss_0(y_v0, t_v.to(device_0)).item())
                loss_original_batch_val.update(loss_0(x_v.to(device_0), t_v.to(device_0)).item())

                ssim_batch_val_0.update(SSIM(y_v0, t_v.to(device_0)).item())
                ssim_original_batch_val.update(SSIM(x_v, t_v).item())

                psnr_batch_val_0.update(PSNR(MSE(y_v0, t_v.to(device_0))).item())
                psnr_original_batch_val.update(PSNR(MSE(x_v.to(device_0), t_v.to(device_0))).item())

            # Free up space in GPU
            del x_v, y_v0, t_v

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
            'Loss_Val_0': loss_batch_val_0.avg,
            'Loss_Original_Train': loss_original_batch.avg,
            'Loss_Original_Val': loss_original_batch_val.avg,
            'SSIM_Batch_0': ssim_batch_0.avg,
            'SSIM_Val_0': ssim_batch_val_0.avg,
            'SSIM_Original_Train': ssim_original_batch.avg,
            'SSIM_Original_Val': ssim_original_batch_val.avg,
            'PSNR_Batch_0': psnr_batch_0.avg,
            'PSNR_Val_0': psnr_batch_val_0.avg,
            'PSNR_Original_Train': psnr_original_batch.avg,
            'PSNR_Original_Val': psnr_original_batch_val.avg
        })

        Legend = ['DHDN_Train', 'Orig_Train', 'DHDN_Val', 'Orig_Val']

        vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([ssim_batch_0.avg, ssim_original_batch.avg,
                               ssim_batch_val_0.avg, ssim_original_batch_val.avg]),
            win=vis_window['SSIM_{date}'.format(date=d1)],
            opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack([psnr_batch_0.avg, psnr_original_batch.avg,
                               psnr_batch_val_0.avg, psnr_original_batch_val.avg]),
            win=vis_window['PSNR_{date}'.format(date=d1)],
            opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
            update='append' if epoch > 0 else None)

        loss_batch_0.reset()
        loss_original_batch.reset()
        ssim_batch_0.reset()
        ssim_original_batch.reset()
        psnr_batch_0.reset()
        psnr_original_batch.reset()
        loss_batch_val_0.reset()
        loss_original_batch_val.reset()
        ssim_batch_val_0.reset()
        ssim_original_batch_val.reset()
        psnr_batch_val_0.reset()
        psnr_original_batch_val.reset()

        scheduler_0.step()

        if epoch > 0 and not epoch % 5:
            if epoch % 10:
                model_path_0 = dir_current + '/models/{date}_dhdn_SIDD_{epoch}.pth'.format(date=d1, epoch=epoch)

                torch.save(dhdn.state_dict(), model_path_0)

            state_dict_dhdn = clip_weights(dhdn.state_dict(), k=3, device=device_0)

            dhdn.load_state_dict(state_dict_dhdn)

# Save final model
model_path_0 = dir_current + '/models/{date}_dhdn_SIDD.pth'.format(date=d1)
torch.save(dhdn.state_dict(), model_path_0)
