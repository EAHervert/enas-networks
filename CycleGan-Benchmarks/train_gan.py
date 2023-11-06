import os
import sys
from utilities import dataset
from ENAS_CycleGAN import Generator, Discriminator
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import utilities.lpips as lpips
from torch.utils.data import DataLoader
import visdom
import argparse

from utilities.utils import CSVLogger, Logger
from utilities.functions import SSIM, PSNR, generate_gan_loggers, drop_weights, clip_weights

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

# Parser
parser = argparse.ArgumentParser(
    prog='Train_GAN',
    description='Training GAN with different parameters',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--lambda_1', default=0.0, type=float)  # Identity loss G(y) approx y
parser.add_argument('--lambda_2', default=0.0, type=float)  # Supervised loss G(x) approx y
parser.add_argument('--training_csv_1', default='sidd_np_instances_064_128_1.csv', type=str)  # training samples to use
parser.add_argument('--training_csv_2', default='sidd_np_instances_064_128_2.csv', type=str)  # training samples to use
parser.add_argument('--device', default='cuda:0', type=str)  # Which device to use to generate .mat file
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
parser.add_argument('--model_size', default=6, type=int)  # Load previous models
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_gan.json'
config = json.load(open(config_path))
if not os.path.exists(dir_current + '/models/'):
    os.makedirs(dir_current + '/models/')

# Noise Dataset
if args.noise == 'SIDD':
    path_training_1 = dir_current + '/instances/' + args.training_csv_1
    path_training_2 = dir_current + '/instances/' + args.training_csv_2
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
Field_Names = ['Loss_D', 'Loss_GANG', 'Loss_IY', 'Loss_Sup_XY',
               'SSIM_Batch', 'SSIM_Original_Train', 'SSIM_Val', 'SSIM_Original_Val',
               'PSNR_Batch', 'PSNR_Original_Train', 'PSNR_Val', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(args.device)

# Load the Models:
D = Discriminator.NLayerDiscriminator(input_nc=3)
G = Generator.UnetGenerator(input_nc=3, output_nc=3, num_downs=args.model_size)  # G: X -> Y

D = D.to(device_0)
G = G.to(device_0)

if args.load_models:
    state_dict_D = torch.load(dir_current + config['Training']['Model_Path_D'], map_location=device_0)
    state_dict_G = torch.load(dir_current + config['Training']['Model_Path_G'], map_location=device_0)

    if args.drop > 0:
        state_dict_D = drop_weights(state_dict_D, p=args.drop, device=device_0)
        state_dict_G = drop_weights(state_dict_G, p=args.drop, device=device_0)

    D.load_state_dict(state_dict_D)
    G.load_state_dict(state_dict_G)

# Create the Visdom window:
# This window will show the SSIM and PSNR of the different networks.
vis = visdom.Visdom(
    server='eng-ml-01.utdallas.edu',
    port=8097,
    use_incoming_socket=False
)

# Display the data to the window:
vis.env = 'CycleGAN_' + str(args.noise)
vis_window = {'Loss_{date}'.format(date=d1): None,
              'SSIM_{date}'.format(date=d1): None,
              'PSNR_{date}'.format(date=d1): None}

# Define the optimizers:
optimizer_D = torch.optim.Adam(D.parameters(), lr=config['Training']['Learning_Rate'],
                               betas=(config['Training']['Beta_1'], config['Training']['Beta_2']))
optimizer_G = torch.optim.Adam(G.parameters(), lr=config['Training']['Learning_Rate'],
                               betas=(config['Training']['Beta_1'], config['Training']['Beta_2']))

# Define the Scheduling:
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, 3, 0.5, -1)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, 3, 0.5, -1)

# Define the Loss and evaluation metrics:
loss_0 = nn.L1Loss().to(device_0)
mse_0 = nn.MSELoss().to(device_0)
loss_fn_alex_0 = lpips.LPIPS(net='alex').to(device_0)  # best forward scores

# Now, let us define our loggers:
loggers = generate_gan_loggers()
ssim_meter_batch, ssim_original_meter_batch, psnr_meter_batch, psnr_original_meter_batch = loggers[0][0:4]
loss_D, loss_GANG, loss_IY, loss_sup_XY = loggers[0][4:]
ssim_meter_val, ssim_original_meter_val, psnr_meter_val, psnr_original_meter_val = loggers[1]

# Load the Noisy and GT Data (X and Y):
SIDD_training_1 = dataset.DatasetSIDD(csv_file=path_training_1, transform=dataset.RandomProcessing())
SIDD_training_2 = dataset.DatasetSIDD(csv_file=path_training_2, transform=dataset.RandomProcessing())
SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy, mat_gt_file=path_validation_gt)

dataloader_sidd_training_1 = DataLoader(dataset=SIDD_training_1, batch_size=config['Training']['Train_Batch_Size'],
                                        shuffle=True, num_workers=8)
dataloader_sidd_training_2 = DataLoader(dataset=SIDD_training_2, batch_size=config['Training']['Train_Batch_Size'],
                                        shuffle=True, num_workers=8)
dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=config['Training']['Validation_Batch_Size'],
                                        shuffle=False, num_workers=8)

for epoch in range(config['Training']['Epochs']):
    for i_batch, (sample_batch_1, sample_batch_2) in enumerate(zip(dataloader_sidd_training_1,
                                                                   dataloader_sidd_training_2)):
        x_1 = sample_batch_1['NOISY']
        y_1 = sample_batch_1['GT']
        x_2 = sample_batch_2['NOISY']
        y_2 = sample_batch_2['GT']

        # Generator Operations
        G_x = G(x_1.to(device_0))  # x -> G(x)

        # Calculate raw values between x and y
        with torch.no_grad():
            ssim_original_meter_batch.update(SSIM(y_1, x_1).item())
            ssim_meter_batch.update(SSIM(x_1, G_x.to('cpu')).item())

            psnr_original_meter_batch.update(PSNR(mse_0(y_1.to(device_0), x_1.to(device_0))).item())
            psnr_meter_batch.update(PSNR(mse_0(y_1.to(device_0), G_x.to(device_0))).item())

        # Discriminator Operators
        D_y = D(y_2.to(device_0))  # True clean images
        D_G_x = D(G_x)  # Denoised images
        Target_1 = torch.ones_like(D_y)

        # Calculate Losses (Discriminators):
        Loss_D_calc = mse_0(D_y, Target_1.to(device_0)) + mse_0(D_G_x, Target_1.to(device_0) * 0)

        # Update the Discriminators:
        optimizer_D.zero_grad()
        Loss_D_calc.backward()
        optimizer_D.step()

        del G_x, D_G_x, Target_1

        # Generator Operators
        G_x = G(x_1.to(device_0))  # x -> G(x)
        G_y = G(y_2.to(device_0))  # y -> G(y)

        # Updated Discriminator Operations
        D_G_x = D(G_x)
        Target_1 = torch.ones_like(D_G_x)

        # Calculate Losses (Generators):
        Loss_GANG_calc = mse_0(D_G_x.to(device_0), Target_1.to(device_0))

        Loss_IY_calc = loss_0(G_y, y_2.to(device_0))
        Loss_Sup_XY_calc = loss_0(G_x, y_1.to(device_0))

        Loss_G_calc = Loss_GANG_calc + args.lambda_1 * Loss_IY_calc + args.lambda_2 * Loss_Sup_XY_calc

        # Update the Generators:
        optimizer_G.zero_grad()
        Loss_G_calc.backward()
        optimizer_G.step()

        loss_D.update(Loss_D_calc.item())
        loss_GANG.update(Loss_GANG_calc.item())
        loss_IY.update(Loss_IY_calc.item())
        loss_sup_XY.update(Loss_Sup_XY_calc.item())

        if i_batch % 100 == 0:
            Display_Loss_D = "Loss_D: %.6f" % loss_D.val
            Display_Loss_G = "Loss_GANG: %.6f" % loss_GANG.val + "\tLoss_IY: %.6f" % loss_IY.val
            Display_Loss_Sup = "Loss_Sup: %.6f" % loss_sup_XY.val
            Display_SSIM = "SSIM_Batch: %.6f" % ssim_meter_batch.val + \
                           "\tSSIM_Original_Batch: %.6f" % ssim_original_meter_batch.val
            Display_PSNR = "PSNR_Batch: %.6f" % psnr_meter_batch.val + \
                           "\tPSNR_Original_Batch: %.6f" % psnr_original_meter_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Display_Loss_D + '\n' + Display_Loss_G + '\n' + Display_Loss_Sup + '\n' + Display_SSIM + '\n' +
                  Display_PSNR)

        # Free up space in GPU
        del x_1, y_1, x_2, y_2, G_x, G_y, D_y, D_G_x

    Display_Loss_D = "Loss_D: %.6f" % loss_D.avg
    Display_Loss_G = "Loss_GANG: %.6f" % loss_GANG.avg + "\tLoss_IY: %.6f" % loss_IY.avg
    Display_Loss_Sup = "Loss_Sup: %.6f" % loss_sup_XY.avg
    Display_SSIM = "SSIM_Batch: %.6f" % ssim_meter_batch.avg + \
                   "\tSSIM_Original_Batch: %.6f" % ssim_original_meter_batch.avg
    Display_PSNR = "PSNR_Batch: %.6f" % psnr_meter_batch.avg + \
                   "\tPSNR_Original_Batch: %.6f" % psnr_original_meter_batch.avg

    print("Training Data for Epoch: ", epoch)
    print(Display_Loss_D + '\n' + Display_Loss_G + '\n' + Display_Loss_Sup + '\n' + Display_SSIM + '\n' + Display_PSNR)

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        y_v = validation_batch['GT']

        with torch.no_grad():
            G_x_v = G(x_v.to(device_0))

            ssim_meter_val.update(SSIM(G_x_v.to('cpu'), y_v).item())
            ssim_original_meter_val.update(SSIM(x_v, y_v).item())

            psnr_meter_val.update(PSNR(mse_0(G_x_v, y_v.to(device_0))).item())
            psnr_original_meter_val.update(PSNR(mse_0(x_v.to(device_0), y_v.to(device_0))).item())

        # Free up space in GPU
        del x_v, y_v, G_x_v

    Display_SSIM = "SSIM: %.6f" % ssim_meter_val.avg + "\tSSIM_Original: %.6f" % ssim_original_meter_val.avg
    Display_PSNR = "PSNR: %.6f" % psnr_meter_val.avg + "\tPSNR_Original: %.6f" % psnr_original_meter_val.avg

    print('\n' + '-' * 160 + '\n')
    print("Validation Data for Epoch: ", epoch)
    print(Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_D': loss_D.avg,
        'Loss_GANG': loss_GANG.avg,
        'Loss_IY': loss_IY.avg,
        'Loss_Sup_XY': loss_sup_XY.avg,
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
    Legend_Loss = ['Loss_D', 'Loss_GANG', 'Loss_IY', 'Loss_Sup_XY']

    vis_window['Loss_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 4),
        Y=np.column_stack([loss_D.avg, loss_GANG.avg, loss_IY.avg, loss_sup_XY.avg]),
        win=vis_window['Loss_{date}'.format(date=d1)],
        opts=dict(title='Loss_{date}'.format(date=d1), xlabel='Epoch', ylabel='Loss', legend=Legend_Loss),
        update='append' if epoch > 0 else None)

    Legend = ['Train', 'Orig_Train', 'Val', 'Orig_Val']

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

    loss_D.reset()
    loss_GANG.reset()
    loss_IY.reset()
    loss_sup_XY.reset()
    ssim_meter_batch.reset()
    ssim_original_meter_batch.reset()
    psnr_meter_batch.reset()
    psnr_original_meter_batch.reset()
    ssim_meter_val.reset()
    ssim_original_meter_val.reset()
    psnr_meter_val.reset()
    psnr_original_meter_val.reset()

    scheduler_D.step()
    scheduler_G.step()

    if epoch > 0 and not epoch % 5:
        model_path_D = dir_current + '/models/{date}_D_{noise}_{epoch}.pth'.format(date=d1, noise=args.noise,
                                                                                   epoch=epoch)
        model_path_G = dir_current + '/models/{date}_G_{noise}_{epoch}.pth'.format(date=d1, noise=args.noise,
                                                                                   epoch=epoch)

        torch.save(D.state_dict(), model_path_D)
        torch.save(G.state_dict(), model_path_G)

        if not epoch % 10:
            state_dict_D = clip_weights(D.state_dict(), k=3, device=device_0)
            state_dict_G = clip_weights(G.state_dict(), k=3, device=device_0)

            D.load_state_dict(state_dict_D)
            G.load_state_dict(state_dict_G)

# Save final model
model_path_D = dir_current + '/models/{date}_D_{noise}.pth'.format(date=d1, noise=args.noise)
model_path_G = dir_current + '/models/{date}_G_{noise}.pth'.format(date=d1, noise=args.noise)

torch.save(D.state_dict(), model_path_D)
torch.save(G.state_dict(), model_path_G)
