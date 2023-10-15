import os
import sys
from utilities import dataset
from ENAS_CycleGAN import Generator, Discriminator
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import visdom
import argparse

from utilities.utils import CSVLogger, Logger
from utilities.functions import SSIM, PSNR, generate_cyclegan_loggers, drop_weights, clip_weights

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

# Parser
parser = argparse.ArgumentParser(
    prog='Train_CycleGAN',
    description='Training cyclegan with different parameters',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--lambda_11', default=1.0, type=float)  # Cycle loss X -> Y -> X
parser.add_argument('--lambda_12', default=1.0, type=float)  # Cycle loss X -> Y -> X
parser.add_argument('--lambda_21', default=0.0, type=float)  # Identity loss G(y) approx y
parser.add_argument('--lambda_22', default=0.0, type=float)  # Identity loss F(x) approx x
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_cyclegan.json'
config = json.load(open(config_path))
if not os.path.exists(dir_current + '/models/'):
    os.makedirs(dir_current + '/models/')

# Noise Dataset
if args.noise == 'SIDD':
    path_training = dir_current + config['Locations']['Training_File']
    path_validation_noisy = dir_current + config['Locations']['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations']['Validation_GT']
    Result_Path = dir_current + '/SIDD/{date}/'.format(date=d1)
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
Field_Names = ['Loss_DX', 'Loss_DY', 'Loss_GANG', 'Loss_GANF', 'Loss_Cyc', 'Loss_IX', 'Loss_IY',
               'SSIM_Batch', 'SSIM_Original_Train', 'SSIM_Val', 'SSIM_Original_Val',
               'PSNR_Batch', 'PSNR_Original_Train', 'PSNR_Val', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the Models:
DX = Discriminator.NLayerDiscriminator(input_nc=3)
DY = Discriminator.NLayerDiscriminator(input_nc=3)
G = Generator.UnetGenerator(input_nc=3, output_nc=3, num_downs=6)  # G: X -> Y
F = Generator.UnetGenerator(input_nc=3, output_nc=3, num_downs=6)  # F: Y -> X

DX = DX.to(device_0)
DY = DY.to(device_1)
F = F.to(device_1)
G = G.to(device_0)

if args.load_models:
    state_dict_DX = torch.load(dir_current + config['Training']['Model_Path_DX'], map_location=device_0)
    state_dict_DY = torch.load(dir_current + config['Training']['Model_Path_DY'], map_location=device_1)
    state_dict_F = torch.load(dir_current + config['Training']['Model_Path_F'], map_location=device_0)
    state_dict_G = torch.load(dir_current + config['Training']['Model_Path_G'], map_location=device_1)

    if args.drop > 0:
        state_dict_DX = drop_weights(state_dict_DX, p=args.drop, device=device_0)
        state_dict_DY = drop_weights(state_dict_DY, p=args.drop, device=device_1)
        state_dict_F = drop_weights(state_dict_F, p=args.drop, device=device_1)
        state_dict_G = drop_weights(state_dict_G, p=args.drop, device=device_0)

    DX.load_state_dict(state_dict_DX)
    DY.load_state_dict(state_dict_DY)
    F.load_state_dict(state_dict_F)
    G.load_state_dict(state_dict_G)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = 'DHDN_Compare_Size_' + str(args.noise)
vis_window = {'Loss_{date}'.format(date=d1): None,
              'SSIM_{date}'.format(date=d1): None,
              'PSNR_{date}'.format(date=d1): None}

# Define the optimizers:
optimizer_DX = torch.optim.Adam(DX.parameters(), config['Training']['Learning_Rate'])
optimizer_DY = torch.optim.Adam(DY.parameters(), config['Training']['Learning_Rate'])
optimizer_F = torch.optim.Adam(F.parameters(), config['Training']['Learning_Rate'])
optimizer_G = torch.optim.Adam(G.parameters(), config['Training']['Learning_Rate'])

# Define the Scheduling:
scheduler_DX = torch.optim.lr_scheduler.StepLR(optimizer_DX, 3, 0.5, -1)
scheduler_DY = torch.optim.lr_scheduler.StepLR(optimizer_DY, 3, 0.5, -1)
scheduler_F = torch.optim.lr_scheduler.StepLR(optimizer_F, 3, 0.5, -1)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_0 = nn.L1Loss().to(device_0)
loss_1 = nn.L1Loss().to(device_1)
mse_0 = nn.MSELoss().to(device_0)
mse_1 = nn.MSELoss().to(device_1)

# Now, let us define our loggers:
loggers = generate_cyclegan_loggers()
ssim_meter_batch, ssim_original_meter_batch, psnr_meter_batch, psnr_original_meter_batch = loggers[0][0:4]
loss_DX, loss_DY, loss_GANG, loss_GANF, loss_Cyc, loss_IX, loss_IY = loggers[0][4:]
ssim_meter_val, ssim_original_meter_val, psnr_meter_val, psnr_original_meter_val = loggers[1]

# Load the Noisy and GT Data (X and Y):
SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing())
SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy, mat_gt_file=path_validation_gt)

dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=config['Training']['Train_Batch_Size'],
                                      shuffle=True, num_workers=16)
dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=config['Training']['Validation_Batch_Size'],
                                        shuffle=False, num_workers=8)

for epoch in range(config['Training']['Epochs']):
    for i_batch, sample_batch in enumerate(dataloader_sidd_training):
        x = sample_batch['NOISY']
        y = sample_batch['GT']

        # Generator Operations
        F_y = F(y.to(device_1)).to(device_0)  # y -> F(y)
        G_x = G(x.to(device_0)).to(device_1)  # x -> G(x)

        # Calculate raw values between x and y
        with torch.no_grad():
            ssim_original_meter_batch.update(SSIM(y, x).item())
            ssim_meter_batch.update(SSIM(x, G_x.to('cpu')).item())

            psnr_original_meter_batch.update(PSNR(mse_0(y.to(device_0), x.to(device_0))).item())
            psnr_meter_batch.update(PSNR(mse_1(y.to(device_1), G_x.to(device_1))).item())

        # Discriminator Operators
        DX_x = DX(x.to(device_0))
        DY_y = DY(y.to(device_1))
        DX_F_y = DX(F_y)
        DY_G_x = DY(G_x)
        Target_1 = torch.ones_like(DX_x)

        # Calculate Losses (Discriminators):
        Loss_DX_calc = mse_0(DX_x, Target_1.to(device_0)) + mse_1(DX_F_y, Target_1.to(device_0))
        Loss_DY_calc = mse_0(DY_y, Target_1.to(device_1)) + mse_1(DY_G_x, Target_1.to(device_1))

        # Update the Discriminators:
        optimizer_DX.zero_grad()
        Loss_DX_calc.backward()
        optimizer_DX.step()

        optimizer_DY.zero_grad()
        Loss_DY_calc.backward()
        optimizer_DY.step()

        del F_y, G_x, DX_x, DY_y, DX_F_y, DY_G_x, Target_1

        # Generator Operators
        F_y = F(y.to(device_1)).to(device_0)  # y -> F(y)
        G_x = G(x.to(device_0)).to(device_1)  # x -> G(x)
        F_G_x_dev_0 = F(G_x).clone().detach().to(device_0)  # x -> G_x -> F_G_x
        G_F_y_dev_0 = G(F_y.clone().detach()).to(device_0)  # y -> F_y -> G_F_y
        F_G_x_dev_1 = F(G_x.clone().detach()).to(device_1)  # x -> G_x -> F_G_x
        G_F_y_dev_1 = G(F_y).clone().detach().to(device_1)  # y -> F_y -> G_F_y
        F_x = F(x.to(device_1)).to(device_0)  # x -> F_x
        G_y = G(y.to(device_0)).to(device_1)  # y -> G_y

        # Updated Discriminator Operations
        DX_F_y = DX(F_y)
        DY_G_x = DY(G_x)
        Target_1 = torch.ones_like(DX_F_y)

        # Calculate Losses (Generators):
        Loss_GANG_calc = mse_0(DY_G_x.to(device_0), Target_1.to(device_0))
        Loss_GANF_calc = mse_1(DX_F_y.to(device_1), Target_1.to(device_1))
        Loss_Cyc_calc = loss_0(F_G_x_dev_0, x.to(device_0)) + loss_0(G_F_y_dev_0, y.to(device_0))
        Loss_Cyc_calc_dev_1 = loss_1(F_G_x_dev_1, x.to(device_1)) + loss_1(G_F_y_dev_1, y.to(device_1))
        Loss_IX_calc = loss_0(F_x, x.to(device_0)).to(device_1)
        Loss_IY_calc = loss_1(G_y, y.to(device_1)).to(device_0)

        Loss_G_calc = Loss_GANG_calc + args.lambda_11 * Loss_Cyc_calc + args.lambda_21 * Loss_IY_calc
        Loss_F_calc = Loss_GANF_calc + args.lambda_12 * Loss_Cyc_calc_dev_1 + args.lambda_22 * Loss_IX_calc

        # Update the Generators:
        optimizer_G.zero_grad()
        Loss_G_calc.backward()
        optimizer_G.step()

        optimizer_F.zero_grad()
        Loss_F_calc.backward()
        optimizer_F.step()

        loss_DX.update(Loss_DX_calc.item())
        loss_DY.update(Loss_DY_calc.item())
        loss_GANG.update(Loss_GANG_calc.item())
        loss_GANF.update(Loss_GANF_calc.item())
        loss_Cyc.update(Loss_Cyc_calc.item())
        loss_IX.update(Loss_IX_calc.item())
        loss_IY.update(Loss_IY_calc.item())

        if i_batch % 100 == 0:
            Display_Loss_D = "Loss_DX: %.6f" % loss_DX.val + "\tLoss_DY: %.6f" % loss_DY.val
            Display_Loss_Cyc = "Loss_Cyc: %.6f" % loss_Cyc.val
            Display_Loss_G = "Loss_GANG: %.6f" % loss_GANG.val + "\tLoss_IY: %.6f" % loss_IY.val
            Display_Loss_F = "Loss_GANF: %.6f" % loss_GANF.val + "\tLoss_IX: %.6f" % loss_IX.val
            Display_SSIM = "SSIM_Batch: %.6f" % ssim_meter_batch.val + \
                           "\tSSIM_Original_Batch: %.6f" % ssim_original_meter_batch.val
            Display_PSNR = "PSNR_Batch: %.6f" % psnr_meter_batch.val + \
                           "\tPSNR_Original_Batch: %.6f" % psnr_original_meter_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss_D + '\n' + Display_Loss_Cyc + '\n' + Display_Loss_G + '\n' + Display_Loss_F + '\n' +
                  Display_SSIM + '\n' + Display_PSNR)

        # Free up space in GPU
        del x, y, F_y, G_x, F_G_x_dev_0, G_F_y_dev_0, F_G_x_dev_1, G_F_y_dev_1, F_x, G_y, DX_F_y, DY_G_x

    Display_Loss_D = "Loss_DX: %.6f" % loss_DX.avg + "\tLoss_DY: %.6f" % loss_DY.avg
    Display_Loss_Cyc = "Loss_Cyc: %.6f" % loss_Cyc.avg
    Display_Loss_G = "Loss_GANG: %.6f" % loss_GANG.avg + "\tLoss_IY: %.6f" % loss_IY.avg
    Display_Loss_F = "Loss_GANF: %.6f" % loss_GANF.avg + "\tLoss_IX: %.6f" % loss_IX.avg
    Display_SSIM = "SSIM_Batch: %.6f" % ssim_meter_batch.avg + \
                   "\tSSIM_Original_Batch: %.6f" % ssim_original_meter_batch.avg
    Display_PSNR = "PSNR_Batch: %.6f" % psnr_meter_batch.avg + \
                   "\tPSNR_Original_Batch: %.6f" % psnr_original_meter_batch.avg

    print("Training Data for Epoch: ", epoch)
    print(Display_Loss_D + '\n' + Display_Loss_Cyc + '\n' + Display_Loss_G + '\n' + Display_Loss_F + '\n' +
          Display_SSIM + '\n' + Display_PSNR)

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        y_v = validation_batch['GT']

        with torch.no_grad():
            G_x_v = G(x_v.to(device_0))

            ssim_original_meter_val.update(SSIM(G_x_v, y_v.to(device_0)).item())
            ssim_meter_val.update(SSIM(x_v.to(device_1), y_v.to(device_1)).item())

            psnr_original_meter_val.update(PSNR(mse_0(G_x_v, y_v.to(device_0))).item())
            psnr_meter_val.update(PSNR(mse_1(x_v.to(device_1), y_v.to(device_1))).item())

        # Free up space in GPU
        del x_v, y_v, G_x_v

    Display_SSIM = "SSIM: %.6f" % ssim_meter_val.avg + "\tSSIM_Original: %.6f" % ssim_original_meter_val.avg
    Display_PSNR = "PSNR_Size_5: %.6f" % psnr_meter_val.avg + "\tPSNR_Original: %.6f" % psnr_original_meter_val.avg

    print("Validation Data for Epoch: ", epoch)
    print(Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_DX': loss_DX.avg,
        'Loss_DY': loss_DY.avg,
        'Loss_GANG': loss_GANG.avg,
        'Loss_GANF': loss_GANF.avg,
        'Loss_Cyc': loss_Cyc.avg,
        'Loss_IX': loss_IX.avg,
        'Loss_IY': loss_IY.avg,
        'SSIM_Batch': ssim_meter_batch.avg,
        'SSIM_Original_Train': ssim_original_meter_batch.avg,
        'SSIM_Val': ssim_meter_val.avg,
        'SSIM_Original_Val': ssim_original_meter_val.val,
        'PSNR_Batch': psnr_meter_batch.avg,
        'PSNR_Original_Train': psnr_original_meter_batch.avg,
        'PSNR_Val': psnr_meter_val.avg,
        'PSNR_Original_Val': psnr_original_meter_val.avg
    })

    # Loss Plotting
    Legend_Loss = ['Loss_DX', 'Loss_DY', 'Loss_GANG', 'Loss_GANF', 'Loss_Cyc', 'Loss_IX', 'Loss_IY']

    vis_window['Loss_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 7),
        Y=np.column_stack([loss_DX.avg, loss_DY.avg, loss_GANG.avg, loss_GANF.avg,
                           loss_Cyc.avg, loss_IX.avg, loss_IY.avg]),
        win=vis_window['Loss_{date}'.format(date=d1)],
        opts=dict(title='Loss_{date}'.format(date=d1), xlabel='Epoch', ylabel='Loss', legend=Legend_Loss),
        update='append' if epoch > 0 else None)

    Legend = ['Train', 'Orig_Train',
              'Val', 'Orig_Val']

    vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 4),
        Y=np.column_stack([ssim_meter_batch.avg, ssim_original_meter_batch.avg,
                           ssim_meter_val.avg, ssim_original_meter_val.avg]),
        win=vis_window['SSIM_{date}'.format(date=d1)],
        opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
        update='append' if epoch > 0 else None)

    vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 4),
        Y=np.column_stack([psnr_meter_batch.avg, psnr_original_meter_batch.avg,
                           psnr_meter_val.avg, psnr_original_meter_val.avg]),
        win=vis_window['PSNR_{date}'.format(date=d1)],
        opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
        update='append' if epoch > 0 else None)

    loss_DX.reset()
    loss_DY.reset()
    loss_GANG.reset()
    loss_GANF.reset()
    loss_Cyc.reset()
    loss_IX.reset()
    loss_IY.reset()
    ssim_meter_batch.reset()
    ssim_original_meter_batch.reset()
    psnr_meter_batch.reset()
    psnr_original_meter_batch.reset()
    ssim_meter_val.reset()
    ssim_original_meter_val.reset()
    psnr_meter_val.reset()
    psnr_original_meter_val.reset()

    scheduler_DX.step()
    scheduler_DY.step()
    scheduler_G.step()
    scheduler_F.step()

    if epoch > 0 and not epoch % 10:
        model_path_DX = dir_current + '/models/{date}_DX_{noise}_{epoch}.pth'.format(date=d1, noise=args.noise,
                                                                                     epoch=epoch)
        model_path_DY = dir_current + '/models/{date}_DY_{noise}_{epoch}.pth'.format(date=d1, noise=args.noise,
                                                                                     epoch=epoch)
        model_path_F = dir_current + '/models/{date}_F_{noise}_{epoch}.pth'.format(date=d1, noise=args.noise,
                                                                                   epoch=epoch)
        model_path_G = dir_current + '/models/{date}_G_{noise}_{epoch}.pth'.format(date=d1, noise=args.noise,
                                                                                   epoch=epoch)

        torch.save(DX.state_dict(), model_path_DX)
        torch.save(DY.state_dict(), model_path_DY)
        torch.save(F.state_dict(), model_path_F)
        torch.save(G.state_dict(), model_path_G)

        state_dict_DX = clip_weights(DX.state_dict(), k=3, device=device_0)
        state_dict_DY = clip_weights(DY.state_dict(), k=3, device=device_1)
        state_dict_F = clip_weights(F.state_dict(), k=3, device=device_0)
        state_dict_G = clip_weights(G.state_dict(), k=3, device=device_1)

        DX.load_state_dict(state_dict_DX)
        DY.load_state_dict(state_dict_DY)
        F.load_state_dict(state_dict_F)
        G.load_state_dict(state_dict_G)

# Save final model
model_path_DX = dir_current + '/models/{date}_DX_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_DY = dir_current + '/models/{date}_DY_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_F = dir_current + '/models/{date}_F_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_G = dir_current + '/models/{date}_G_{noise}.pth'.format(date=d1, noise=args.noise)

torch.save(DX.state_dict(), model_path_DX)
torch.save(DY.state_dict(), model_path_DY)
torch.save(F.state_dict(), model_path_F)
torch.save(G.state_dict(), model_path_G)
