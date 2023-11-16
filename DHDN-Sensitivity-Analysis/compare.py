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
from utilities.functions import SSIM, PSNR, generate_loggers, drop_weights, clip_weights

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

# Parser
parser = argparse.ArgumentParser(
    prog='DHDN_Compare',
    description='Compares 3 different methods for DHDN',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--tag', default='Upsample', type=str)  # Which dataset to train on
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--gaussian', default='-1', type=float)  # Gaussian noise addition for model weight # initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
parser.add_argument('--training_csv', default='sidd_np_instances_064_64.csv', type=str)  # training samples to use
parser.add_argument('--epochs', default=25, type=int)  # number of epochs to train on
parser.add_argument('--learning_rate', default=1e-4, type=float)  # number of epochs to train on
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_compare.json'
config = json.load(open(config_path))

model_folder = '/models/compare_{tag}/'.format(tag=args.tag)
tag_1, tag_2, tag_3 = config['Training'][args.tag]['Tags']
if not os.path.exists(dir_current + model_folder):
    os.makedirs(dir_current + model_folder)

# Noise Dataset
if args.noise == 'SIDD':
    path_training = dir_current + '/instances/' + args.training_csv
    path_validation_noisy = dir_current + config['Locations']['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations']['Validation_GT']
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

if not os.path.isdir(Result_Path + '/' + config['Locations']['Output_File'] + '_' + args.tag):
    os.mkdir(Result_Path + '/' + config['Locations']['Output_File'] + '_' + args.tag)
sys.stdout = Logger(Result_Path + '/' + config['Locations']['Output_File'] + '_' + args.tag + '/log.log')

# Create the CSV Logger:
File_Name = Result_Path + '/' + config['Locations']['Output_File'] + '_' + args.tag + '/data.csv'
Field_Names = ['Loss_Batch_{tag_1}', 'Loss_Batch_{tag_2}', 'Loss_Batch_{tag_3}', 'Loss_Original_Train',
               'Loss_Val_{tag_1}', 'Loss_Val_{tag_2}', 'Loss_Val_{tag_3}', 'Loss_Original_Val',
               'SSIM_Batch_{tag_1}', 'SSIM_Batch_{tag_2}', 'SSIM_Batch_{tag_3}', 'SSIM_Original_Train',
               'SSIM_Val_{tag_1}', 'SSIM_Val_{tag_2}', 'SSIM_Val_{tag_3}', 'SSIM_Original_Val',
               'PSNR_Batch_{tag_1}', 'PSNR_Batch_{tag_2}', 'PSNR_Batch_{tag_3}', 'PSNR_Original_Train',
               'PSNR_Val_{tag_1}', 'PSNR_Val_{tag_2}', 'PSNR_Val_{tag_3}', 'PSNR_Original_Val']

Field_Names = [name.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3) for name in Field_Names]
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the Models:
architecture_1, architecture_2, architecture_3 = config['Training'][args.tag]['Architecture']
channels_1, channels_2, channels_3 = config['Training'][args.tag]['Channels']
k_value_1, k_value_2, k_value_3 = config['Training'][args.tag]['K_Value']

architecture_1 = architecture_1[0] + architecture_1[1] + architecture_1[2]
architecture_2 = architecture_2[0] + architecture_2[1] + architecture_2[2]
architecture_3 = architecture_3[0] + architecture_3[1] + architecture_3[2]

# Model_1
dhdn_1 = DHDN.SharedDHDN(channels=channels_1, k_value=k_value_1, architecture=architecture_1)
dhdn_2 = DHDN.SharedDHDN(channels=channels_2, k_value=k_value_2, architecture=architecture_2)
dhdn_3 = DHDN.SharedDHDN(channels=channels_3, k_value=k_value_3, architecture=architecture_3)

dhdn_1 = dhdn_1.to(device_0)
dhdn_2 = dhdn_2.to(device_0)
dhdn_3 = dhdn_3.to(device_1)

if args.load_models:
    model_paths = config['Training'][args.tag]['Models']

    state_dict_dhdn_1 = torch.load(dir_current + model_paths[0], map_location=device_0)
    state_dict_dhdn_2 = torch.load(dir_current + model_paths[1], map_location=device_0)
    state_dict_dhdn_3 = torch.load(dir_current + model_paths[2], map_location=device_1)

    if args.drop > 0:
        state_dict_dhdn_1 = drop_weights(state_dict_dhdn_1, p=args.drop, device=device_0)
        state_dict_dhdn_2 = drop_weights(state_dict_dhdn_2, p=args.drop, device=device_0)
        state_dict_dhdn_3 = drop_weights(state_dict_dhdn_3, p=args.drop, device=device_1)

    dhdn_1.load_state_dict(state_dict_dhdn_1)
    dhdn_2.load_state_dict(state_dict_dhdn_2)
    dhdn_3.load_state_dict(state_dict_dhdn_3)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = 'DHDN_Compare_' + args.tag + '_' + str(args.noise)
vis_window = {'Loss_{date}'.format(date=d1): None,
              'SSIM_{date}'.format(date=d1): None,
              'PSNR_{date}'.format(date=d1): None}

# Define the optimizers:
optimizer_1 = torch.optim.Adam(dhdn_1.parameters(), args.learning_rate)
optimizer_2 = torch.optim.Adam(dhdn_2.parameters(), args.learning_rate)
optimizer_3 = torch.optim.Adam(dhdn_3.parameters(), args.learning_rate)

# Define the Scheduling:
scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, 3, 0.5, -1)
scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, 3, 0.5, -1)
scheduler_3 = torch.optim.lr_scheduler.StepLR(optimizer_3, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_0 = nn.L1Loss().to(device_0)
loss_1 = nn.L1Loss().to(device_1)
mse_0 = nn.MSELoss().to(device_0)
mse_1 = nn.MSELoss().to(device_1)

# Now, let us define our loggers:
loggers_1 = generate_loggers()
loggers_2 = generate_loggers()
loggers_3 = generate_loggers()

# Training Batches
loss_batch_1, loss_original_batch, ssim_batch_1, ssim_original_batch = loggers_1[0][0:4]
psnr_batch_1, psnr_original_batch = loggers_1[0][4:]
loss_batch_2, _, ssim_batch_2, _, psnr_batch_2, _ = loggers_2[0]
loss_batch_3, _, ssim_batch_3, _, psnr_batch_3, _ = loggers_3[0]

# Validation Batches
loss_batch_val_1, loss_original_batch_val, ssim_batch_val_1, ssim_original_batch_val = loggers_1[1][0:4]
psnr_batch_val_1, psnr_original_batch_val = loggers_1[1][4:]

loss_batch_val_2, _, ssim_batch_val_2, _, psnr_batch_val_2, _ = loggers_2[1]
loss_batch_val_3, _, ssim_batch_val_3, _, psnr_batch_val_3, _ = loggers_3[1]

# Load the Training and Validation Data:
SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing())
SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy, mat_gt_file=path_validation_gt)

dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=config['Training']['Train_Batch_Size'],
                                      shuffle=True, num_workers=16)
dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=config['Training']['Validation_Batch_Size'],
                                        shuffle=False, num_workers=8)

for epoch in range(args.epochs):

    for i_batch, sample_batch in enumerate(dataloader_sidd_training):
        x = sample_batch['NOISY']
        y_1 = dhdn_1(x.to(device_0))
        y_2 = dhdn_2(x.to(device_0))
        y_3 = dhdn_3(x.to(device_1))
        t = sample_batch['GT']

        loss_value_1 = loss_0(y_1, t.to(device_0))
        loss_value_2 = loss_0(y_2, t.to(device_0))
        loss_value_3 = loss_1(y_3, t.to(device_1))
        loss_batch_1.update(loss_value_1.item())
        loss_batch_2.update(loss_value_2.item())
        loss_batch_3.update(loss_value_3.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss_1(x.to(device_1), t.to(device_1)).item())

            ssim_batch_1.update(SSIM(y_1, t.to(device_0)).item())
            ssim_batch_2.update(SSIM(y_2, t.to(device_0)).item())
            ssim_batch_3.update(SSIM(y_3, t.to(device_1)).item())
            ssim_original_batch.update(SSIM(x, t).item())

            psnr_batch_1.update(PSNR(mse_0(y_1, t.to(device_0))).item())
            psnr_batch_2.update(PSNR(mse_0(y_2, t.to(device_0))).item())
            psnr_batch_3.update(PSNR(mse_1(y_3, t.to(device_1))).item())
            psnr_original_batch.update(PSNR(mse_1(x.to(device_1), t.to(device_1))).item())

        # Backpropagate to train model
        optimizer_1.zero_grad()
        loss_value_1.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        loss_value_2.backward()
        optimizer_2.step()

        optimizer_3.zero_grad()
        loss_value_3.backward()
        optimizer_3.step()

        if i_batch % 100 == 0:
            Display_Loss = "Loss_Size_{tag_1}: %.6f" % loss_batch_1.val + \
                           "\tLoss_Size_{tag_2}: %.6f" % loss_batch_2.val + \
                           "\tLoss_Size_{tag_3}: %.6f" % loss_batch_3.val + \
                           "\tLoss_Original: %.6f" % loss_original_batch.val
            Display_SSIM = "SSIM_Size_{tag_1}: %.6f" % ssim_batch_1.val + \
                           "\tSSIM_Size_{tag_2}: %.6f" % ssim_batch_2.val + \
                           "\tSSIM_Size_{tag_3}: %.6f" % ssim_batch_3.val + \
                           "\tSSIM_Original: %.6f" % ssim_original_batch.val
            Display_PSNR = "PSNR_Size_{tag_1}: %.6f" % psnr_batch_1.val + \
                           "\tPSNR_Size_{tag_2}: %.6f" % psnr_batch_2.val + \
                           "\tPSNR_Size_{tag_3}: %.6f" % psnr_batch_3.val + \
                           "\tPSNR_Original: %.6f" % psnr_original_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3) + '\n'
                  + Display_SSIM.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3) + '\n'
                  + Display_PSNR.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3))

        # Free up space in GPU
        del x, y_1, y_2, y_3, t

    Display_Loss = "Loss_Size_{tag_1}: %.6f" % loss_batch_1.avg + "\tLoss_Size_{tag_2}: %.6f" % loss_batch_2.avg + \
                   "\tLoss_Size_{tag_3}: %.6f" % loss_batch_3.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
    Display_SSIM = "SSIM_Size_{tag_1}: %.6f" % ssim_batch_1.avg + "\tSSIM_Size_{tag_2}: %.6f" % ssim_batch_2.avg + \
                   "\tSSIM_Size_{tag_3}: %.6f" % ssim_batch_3.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
    Display_PSNR = "PSNR_Size_{tag_1}: %.6f" % psnr_batch_1.avg + "\tPSNR_Size_{tag_2}: %.6f" % psnr_batch_2.avg + \
                   "\tPSNR_Size_{tag_3}: %.6f" % psnr_batch_3.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

    print('\n' + '-' * 160)
    print("Training Data for Epoch: ", epoch)
    print(Display_Loss.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3) + '\n'
          + Display_SSIM.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3) + '\n'
          + Display_PSNR.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3))
    print('-' * 160 + '\n')

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']
        with torch.no_grad():
            y_v_1 = dhdn_1(x_v.to(device_0))
            y_v_2 = dhdn_2(x_v.to(device_0))
            y_v_3 = dhdn_3(x_v.to(device_1))

            loss_batch_val_1.update(loss_0(y_v_1, t_v.to(device_0)).item())
            loss_batch_val_2.update(loss_0(y_v_2, t_v.to(device_0)).item())
            loss_batch_val_3.update(loss_1(y_v_3, t_v.to(device_1)).item())
            loss_original_batch_val.update(loss_1(x_v.to(device_1), t_v.to(device_1)).item())

            ssim_batch_val_1.update(SSIM(y_v_1, t_v.to(device_0)).item())
            ssim_batch_val_2.update(SSIM(y_v_2, t_v.to(device_0)).item())
            ssim_batch_val_3.update(SSIM(y_v_3, t_v.to(device_1)).item())
            ssim_original_batch_val.update(SSIM(x_v, t_v).item())

            psnr_batch_val_1.update(PSNR(mse_0(y_v_1, t_v.to(device_0))).item())
            psnr_batch_val_2.update(PSNR(mse_0(y_v_2, t_v.to(device_0))).item())
            psnr_batch_val_3.update(PSNR(mse_1(y_v_3, t_v.to(device_1))).item())
            psnr_original_batch_val.update(PSNR(mse_1(x_v.to(device_1), t_v.to(device_1))).item())

        # Free up space in GPU
        del x_v, y_v_1, y_v_2, y_v_3, t_v

    Display_Loss = "Loss_Size_{tag_1}: %.6f" % loss_batch_val_1.avg + \
                   "\tLoss_Size_{tag_2}: %.6f" % loss_batch_val_2.avg + \
                   "\tLoss_Size_{tag_3}: %.6f" % loss_batch_val_3.avg + \
                   "\tLoss_Original: %.6f" % loss_original_batch_val.avg
    Display_SSIM = "SSIM_Size_{tag_1}: %.6f" % ssim_batch_val_1.avg + \
                   "\tSSIM_Size_{tag_2}: %.6f" % ssim_batch_val_2.avg + \
                   "\tSSIM_Size_{tag_3}: %.6f" % ssim_batch_val_3.avg + \
                   "\tSSIM_Original: %.6f" % ssim_original_batch_val.avg
    Display_PSNR = "PSNR_Size_{tag_1}: %.6f" % psnr_batch_val_1.avg + \
                   "\tPSNR_Size_{tag_2}: %.6f" % psnr_batch_val_2.avg + \
                   "\tPSNR_Size_{tag_3}: %.6f" % psnr_batch_val_3.avg + \
                   "\tPSNR_Original: %.6f" % psnr_original_batch_val.avg

    print('\n' + '-' * 160)
    print("Validation Data for Epoch: ", epoch)
    print(Display_Loss.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3) + '\n'
          + Display_SSIM.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3) + '\n'
          + Display_PSNR.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3))
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_Batch_{tag}'.format(tag=tag_1): loss_batch_1.avg,
        'Loss_Batch_{tag}'.format(tag=tag_2): loss_batch_2.avg,
        'Loss_Batch_{tag}'.format(tag=tag_3): loss_batch_3.avg,
        'Loss_Val_{tag}'.format(tag=tag_1): loss_batch_val_1.avg,
        'Loss_Val_{tag}'.format(tag=tag_2): loss_batch_val_2.avg,
        'Loss_Val_{tag}'.format(tag=tag_3): loss_batch_val_3.avg,
        'Loss_Original_Train': loss_original_batch.avg,
        'Loss_Original_Val': loss_original_batch_val.avg,
        'SSIM_Batch_{tag}'.format(tag=tag_1): ssim_batch_1.avg,
        'SSIM_Batch_{tag}'.format(tag=tag_2): ssim_batch_2.avg,
        'SSIM_Batch_{tag}'.format(tag=tag_3): ssim_batch_3.avg,
        'SSIM_Val_{tag}'.format(tag=tag_1): ssim_batch_val_1.avg,
        'SSIM_Val_{tag}'.format(tag=tag_2): ssim_batch_val_2.avg,
        'SSIM_Val_{tag}'.format(tag=tag_3): ssim_batch_val_3.avg,
        'SSIM_Original_Train': ssim_original_batch.avg,
        'SSIM_Original_Val': ssim_original_batch_val.avg,
        'PSNR_Batch_{tag}'.format(tag=tag_1): psnr_batch_1.avg,
        'PSNR_Batch_{tag}'.format(tag=tag_2): psnr_batch_2.avg,
        'PSNR_Batch_{tag}'.format(tag=tag_3): psnr_batch_3.avg,
        'PSNR_Val_{tag}'.format(tag=tag_1): psnr_batch_val_1.avg,
        'PSNR_Val_{tag}'.format(tag=tag_2): psnr_batch_val_2.avg,
        'PSNR_Val_{tag}'.format(tag=tag_3): psnr_batch_val_3.avg,
        'PSNR_Original_Train': psnr_original_batch.avg,
        'PSNR_Original_Val': psnr_original_batch_val.avg
    })

    Legend = ['{tag_1}_Train', '{tag_2}_Train', '{tag_3}_Train', 'Orig_Train',
              '{tag_1}_Val', '{tag_2}_Val', '{tag_3}_Val', 'Orig_Val']
    Legend = [name.format(tag_1=tag_1, tag_2=tag_2, tag_3=tag_3) for name in Legend]

    vis_window['Loss_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([loss_batch_1.avg, loss_batch_2.avg, loss_batch_3.avg, loss_original_batch.avg,
                           loss_batch_val_1.avg, loss_batch_val_2.avg, loss_batch_val_3.avg,
                           loss_original_batch_val.avg]),
        win=vis_window['Loss_{date}'.format(date=d1)],
        opts=dict(title='Loss_{date}'.format(date=d1), xlabel='Epoch', ylabel='Loss', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([ssim_batch_1.avg, ssim_batch_2.avg, ssim_batch_3.avg, ssim_original_batch.avg,
                           ssim_batch_val_1.avg, ssim_batch_val_2.avg, ssim_batch_val_3.avg,
                           ssim_original_batch_val.avg]),
        win=vis_window['SSIM_{date}'.format(date=d1)],
        opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 8),
        Y=np.column_stack([psnr_batch_1.avg, psnr_batch_2.avg, psnr_batch_3.avg, psnr_original_batch.avg,
                           psnr_batch_val_1.avg, psnr_batch_val_2.avg, psnr_batch_val_3.avg,
                           psnr_original_batch_val.avg]),
        win=vis_window['PSNR_{date}'.format(date=d1)],
        opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_batch_1.reset()
    loss_batch_2.reset()
    loss_batch_3.reset()
    loss_original_batch.reset()
    ssim_batch_1.reset()
    ssim_batch_2.reset()
    ssim_batch_3.reset()
    ssim_original_batch.reset()
    psnr_batch_1.reset()
    psnr_batch_2.reset()
    psnr_batch_3.reset()
    psnr_original_batch.reset()
    loss_batch_val_1.reset()
    loss_batch_val_2.reset()
    loss_batch_val_3.reset()
    loss_original_batch_val.reset()
    ssim_batch_val_1.reset()
    ssim_batch_val_2.reset()
    ssim_batch_val_3.reset()
    ssim_original_batch_val.reset()
    psnr_batch_val_1.reset()
    psnr_batch_val_2.reset()
    psnr_batch_val_3.reset()
    psnr_original_batch_val.reset()

    scheduler_1.step()
    scheduler_2.step()
    scheduler_3.step()

    if epoch > 0 and not epoch % 10:
        model_path_1 = dir_current + model_folder + '{date}_dhdn_{tag}_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch, tag=tag_1)
        model_path_2 = dir_current + model_folder + '{date}_dhdn_{tag}_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch, tag=tag_2)
        model_path_3 = dir_current + model_folder + '{date}_dhdn_{tag}_{noise}_{epoch}.pth'.format(
            date=d1, noise=args.noise, epoch=epoch, tag=tag_3)

        torch.save(dhdn_1.state_dict(), model_path_1)
        torch.save(dhdn_2.state_dict(), model_path_2)
        torch.save(dhdn_3.state_dict(), model_path_3)

        state_dict_dhdn_1 = clip_weights(dhdn_1.state_dict(), k=3, device=device_0)
        state_dict_dhdn_2 = clip_weights(dhdn_2.state_dict(), k=3, device=device_0)
        state_dict_dhdn_3 = clip_weights(dhdn_3.state_dict(), k=3, device=device_1)

        dhdn_1.load_state_dict(state_dict_dhdn_1)
        dhdn_2.load_state_dict(state_dict_dhdn_2)
        dhdn_3.load_state_dict(state_dict_dhdn_3)

# Save final model
model_path_1 = dir_current + model_folder + '{date}_dhdn_{tag}_{noise}.pth'.format(date=d1, noise=args.noise, tag=tag_1)
model_path_2 = dir_current + model_folder + '{date}_dhdn_{tag}_{noise}.pth'.format(date=d1, noise=args.noise, tag=tag_2)
model_path_3 = dir_current + model_folder + '{date}_dhdn_{tag}_{noise}.pth'.format(date=d1, noise=args.noise, tag=tag_3)

torch.save(dhdn_1.state_dict(), model_path_1)
torch.save(dhdn_2.state_dict(), model_path_2)
torch.save(dhdn_3.state_dict(), model_path_3)
