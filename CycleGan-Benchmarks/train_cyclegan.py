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
from utilities.functions import SSIM, PSNR, generate_cyclegan_loggers, drop_weights, clip_weights

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

# Parser
parser = argparse.ArgumentParser(
    prog='Train_CycleGAN',
    description='Training cyclegan with different parameters',
)
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--cycle_loss', default='L1', type=str)  # type of Cycle Loss
parser.add_argument('--lambda_11', default=1.0, type=float)  # Cycle loss X -> Y -> X
parser.add_argument('--lambda_12', default=1.0, type=float)  # Cycle loss Y -> X -> Y
parser.add_argument('--lambda_21', default=0.0, type=float)  # Identity loss G(y) approx y
parser.add_argument('--lambda_22', default=0.0, type=float)  # Identity loss F(x) approx x
parser.add_argument('--lambda_31', default=0.0, type=float)  # Supervised loss G(x) approx y
parser.add_argument('--lambda_32', default=0.0, type=float)  # Supervised loss F(y) approx x
parser.add_argument('--training_csv_1', default='sidd_np_instances_128_128_1.csv', type=str)  # training samples to use
parser.add_argument('--training_csv_2', default='sidd_np_instances_128_128_2.csv', type=str)  # training samples to use
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--load_models', default=False, type=bool)  # Load previous models
parser.add_argument('--model_size', default=6, type=int)  # Load previous models
parser.add_argument('--epochs', default=40, type=int)  # number of epochs to train on
args = parser.parse_args()

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_cyclegan.json'
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
Field_Names = ['Loss_DX', 'Loss_DY', 'Loss_GANG', 'Loss_GANF', 'Loss_Cyc_XYX', 'Loss_Cyc_YXY', 'Loss_IX', 'Loss_IY',
               'Loss_Sup_XY', 'Loss_Sup_YX', 'SSIM_Batch', 'SSIM_Original_Train', 'SSIM_Val', 'SSIM_Original_Val',
               'PSNR_Batch', 'PSNR_Original_Train', 'PSNR_Val', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

# Define the devices:
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Load the Models:
DX = Discriminator.NLayerDiscriminator(input_nc=3)
DY = Discriminator.NLayerDiscriminator(input_nc=3)
G = Generator.UnetGenerator(input_nc=3, output_nc=3, num_downs=args.model_size)  # G: X -> Y
F = Generator.UnetGenerator(input_nc=3, output_nc=3, num_downs=args.model_size)  # F: Y -> X

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
vis.env = 'CycleGAN_' + str(args.noise)
vis_window = {'Loss_{date}'.format(date=d1): None,
              'SSIM_{date}'.format(date=d1): None,
              'PSNR_{date}'.format(date=d1): None}

# Define the optimizers:
optimizer_DX = torch.optim.Adam(DX.parameters(), lr=config['Training']['Learning_Rate'],
                                betas=(config['Training']['Beta_1'], config['Training']['Beta_2']))
optimizer_DY = torch.optim.Adam(DY.parameters(), lr=config['Training']['Learning_Rate'],
                                betas=(config['Training']['Beta_1'], config['Training']['Beta_2']))
optimizer_F = torch.optim.Adam(F.parameters(), lr=config['Training']['Learning_Rate'],
                               betas=(config['Training']['Beta_1'], config['Training']['Beta_2']))
optimizer_G = torch.optim.Adam(G.parameters(), lr=config['Training']['Learning_Rate'],
                               betas=(config['Training']['Beta_1'], config['Training']['Beta_2']))

# Define the Loss and evaluation metrics:
loss_0 = nn.L1Loss().to(device_0)
loss_1 = nn.L1Loss().to(device_1)
mse_0 = nn.MSELoss().to(device_0)
mse_1 = nn.MSELoss().to(device_1)
loss_fn_alex_0 = lpips.LPIPS(net='alex').to(device_0)  # best forward scores
loss_fn_alex_1 = lpips.LPIPS(net='alex').to(device_1)  # best forward scores

# Now, let us define our loggers:
loggers = generate_cyclegan_loggers()
ssim_meter_batch, ssim_original_meter_batch, psnr_meter_batch, psnr_original_meter_batch = loggers[0][0:4]
loss_DX, loss_DY, loss_GANG, loss_GANF, loss_Cyc_XYX, loss_Cyc_YXY, loss_IX, loss_IY = loggers[0][4:-2]
loss_sup_XY, loss_sup_YX = loggers[0][-2:]
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

for epoch in range(args.epochs):
    TP, FP, FN, TN = [], [], [], []
    for i_batch, (sample_batch_1, sample_batch_2) in enumerate(zip(dataloader_sidd_training_1,
                                                                   dataloader_sidd_training_2)):
        x_1 = sample_batch_1['NOISY']
        y_1 = sample_batch_1['GT']
        x_2 = sample_batch_2['NOISY']
        y_2 = sample_batch_2['GT']

        # Generator Operations
        F_y = F(y_2.to(device_1)).to(device_0)  # y -> F(y)
        G_x = G(x_1.to(device_0)).to(device_1)  # x -> G(x)

        # Calculate raw values between x and y
        with torch.no_grad():
            ssim_original_meter_batch.update(SSIM(y_1, x_1).item())
            ssim_meter_batch.update(SSIM(y_1, G_x.to('cpu')).item())

            psnr_original_meter_batch.update(PSNR(mse_0(y_1.to(device_0), x_1.to(device_0))).item())
            psnr_meter_batch.update(PSNR(mse_1(y_1.to(device_1), G_x.to(device_1))).item())

        # Discriminator Operators
        DX_x = DX(x_1.to(device_0))
        DY_y = DY(y_2.to(device_1))
        DX_F_y = DX(F_y)
        DY_G_x = DY(G_x)
        Target_1 = torch.ones_like(DX_x)

        # Calculate Losses (Discriminators):
        Loss_DX_calc = mse_0(DX_x, Target_1.to(device_0)) + mse_0(DX_F_y, Target_1.to(device_0) * 0)
        Loss_DY_calc = mse_1(DY_y, Target_1.to(device_1)) + mse_1(DY_G_x, Target_1.to(device_1) * 0)
        TP.append(((DY_y >= 0.5) * 1.).mean().item())
        FP.append(((DY_G_x >= 0.5) * 1.).mean().item())
        FN.append(((DY_y < 0.5) * 1.).mean().item())
        TN.append(((DY_G_x < 0.5) * 1.).mean().item())

        # Update the Discriminators:
        optimizer_DX.zero_grad()
        Loss_DX_calc.backward()
        optimizer_DX.step()

        optimizer_DY.zero_grad()
        Loss_DY_calc.backward()
        optimizer_DY.step()

        del F_y, G_x, DX_x, DY_y, DX_F_y, DY_G_x, Target_1

        # Generator Operators
        F_y = F(y_2.to(device_1)).to(device_0)  # y -> F(y)
        G_x = G(x_1.to(device_0)).to(device_1)  # x -> G(x)
        F_G_x__G = F(G_x).clone().detach().to(device_0)  # x -> G(x) -> F(G(x)) (G Model)
        F_G_x__F = F(G_x.clone().detach()).to(device_1)  # x -> G(x) -> F(G(x)) (F Model)
        G_F_y__G = G(F_y.clone().detach()).to(device_0)  # y -> F(y) -> G(F(y))
        G_F_y__F = G(F_y).clone().detach().to(device_1)  # y -> F(y) -> G(F(y))
        F_x = F(x_1.to(device_1)).to(device_0)  # x -> F(x)
        G_y = G(y_2.to(device_0)).to(device_1)  # y -> G(y)

        # Updated Discriminator Operations
        DX_F_y = DX(F_y)
        DY_G_x = DY(G_x)
        Target_1 = torch.ones_like(DX_F_y)

        # Calculate Losses (Generators):
        Loss_GANG_calc = mse_0(DY_G_x.to(device_0), Target_1.to(device_0))
        Loss_GANF_calc = mse_1(DX_F_y.to(device_1), Target_1.to(device_1))

        if args.cycle_loss == 'L2':
            Loss_Cyc_XYX_G_calc = mse_0(F_G_x__G, x_1.to(device_0))
            Loss_Cyc_XYX_F_calc = mse_1(F_G_x__F, x_1.to(device_1))
            Loss_Cyc_YXY_G_calc = mse_0(G_F_y__G, y_2.to(device_0))
            Loss_Cyc_YXY_F_calc = mse_1(G_F_y__F, y_2.to(device_1))
        elif args.cycle_loss == 'SSIM':
            Loss_Cyc_XYX_G_calc = 1 - SSIM(F_G_x__G, x_1.to(device_0))
            Loss_Cyc_XYX_F_calc = 1 - SSIM(F_G_x__F, x_1.to(device_1))
            Loss_Cyc_YXY_G_calc = 1 - SSIM(G_F_y__G, y_2.to(device_0))
            Loss_Cyc_YXY_F_calc = 1 - SSIM(G_F_y__F, y_2.to(device_1))
        elif args.cycle_loss == 'LPIPS':
            Loss_Cyc_XYX_G_calc = loss_fn_alex_0(F_G_x__G, x_1.to(device_0))
            Loss_Cyc_XYX_F_calc = loss_fn_alex_1(F_G_x__F, x_1.to(device_1))
            Loss_Cyc_YXY_G_calc = loss_fn_alex_0(G_F_y__G, y_2.to(device_0))
            Loss_Cyc_YXY_F_calc = loss_fn_alex_1(G_F_y__F, y_2.to(device_1))
        elif args.cycle_loss == 'Custom':
            Loss_Cyc_XYX_G_calc = loss_0(F_G_x__G, x_1.to(device_0)) + (1 - SSIM(F_G_x__G, x_1.to(device_0))) + \
                                  loss_fn_alex_0(F_G_x__G, x_1.to(device_0))
            Loss_Cyc_XYX_F_calc = loss_1(F_G_x__F, x_1.to(device_1)) + (1 - SSIM(F_G_x__F, x_1.to(device_1))) + \
                                  loss_fn_alex_1(F_G_x__F, x_1.to(device_1))
            Loss_Cyc_YXY_G_calc = loss_0(G_F_y__G, y_2.to(device_0)) + (1 - SSIM(G_F_y__G, y_2.to(device_0))) + \
                                  loss_fn_alex_0(G_F_y__G, y_2.to(device_0))
            Loss_Cyc_YXY_F_calc = loss_1(G_F_y__F, y_2.to(device_1)) + (1 - SSIM(G_F_y__F, y_2.to(device_1))) + \
                                  loss_fn_alex_1(G_F_y__F, y_2.to(device_1))
        else:  # L1 Loss (default)
            Loss_Cyc_XYX_G_calc = loss_0(F_G_x__G, x_1.to(device_0))
            Loss_Cyc_XYX_F_calc = loss_1(F_G_x__F, x_1.to(device_1))
            Loss_Cyc_YXY_G_calc = loss_0(G_F_y__G, y_2.to(device_0))
            Loss_Cyc_YXY_F_calc = loss_1(G_F_y__F, y_2.to(device_1))

        Loss_IX_calc = loss_0(F_x, x_1.to(device_0)).to(device_1)
        Loss_IY_calc = loss_1(G_y, y_2.to(device_1)).to(device_0)
        Loss_Sup_XY_calc = loss_1(G_x, y_1.to(device_1)).to(device_0)
        Loss_Sup_YX_calc = loss_0(F_y, x_2.to(device_0)).to(device_1)

        Loss_G_calc = Loss_GANG_calc + args.lambda_11 * Loss_Cyc_XYX_G_calc + args.lambda_12 * Loss_Cyc_YXY_G_calc + \
                      args.lambda_21 * Loss_IY_calc + args.lambda_31 * Loss_Sup_XY_calc
        Loss_F_calc = Loss_GANF_calc + args.lambda_11 * Loss_Cyc_XYX_F_calc + args.lambda_12 * Loss_Cyc_YXY_F_calc + \
                      args.lambda_22 * Loss_IX_calc + args.lambda_32 * Loss_Sup_YX_calc

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
        loss_Cyc_XYX.update(Loss_Cyc_XYX_G_calc.item())
        loss_Cyc_YXY.update(Loss_Cyc_YXY_F_calc.item())
        loss_IX.update(Loss_IX_calc.item())
        loss_IY.update(Loss_IY_calc.item())
        loss_sup_XY.update(Loss_Sup_XY_calc.item())
        loss_sup_YX.update(Loss_Sup_YX_calc.item())

        if i_batch % 100 == 0:
            Confusion_Matrix = "True Positive: %.6f" % TP[-1] + \
                               "\tFalse Positive: %.6f" % FP[-1] + \
                               "\nFalse Negative: %.6f" % FN[-1] + \
                               "\tTrue Negative: %.6f" % TN[-1]
            Display_Loss_D = "Loss_DX: %.6f" % loss_DX.val + "\tLoss_DY: %.6f" % loss_DY.val
            Display_Loss_Cyc = "loss_Cyc_XYX: %.6f" % loss_Cyc_XYX.val + "\tLoss_Cyc_YXY: %.6f" % loss_Cyc_YXY.val
            Display_Loss_G = "Loss_GANG: %.6f" % loss_GANG.val + "\tLoss_IY: %.6f" % loss_IY.val
            Display_Loss_F = "Loss_GANF: %.6f" % loss_GANF.val + "\tLoss_IX: %.6f" % loss_IX.val
            Display_Loss_Sup = "Loss_Sup_XY: %.6f" % loss_sup_XY.val + "\tLoss_Sup_YX: %.6f" % loss_sup_YX.val
            Display_SSIM = "SSIM_Batch: %.6f" % ssim_meter_batch.val + \
                           "\tSSIM_Original_Batch: %.6f" % ssim_original_meter_batch.val
            Display_PSNR = "PSNR_Batch: %.6f" % psnr_meter_batch.val + \
                           "\tPSNR_Original_Batch: %.6f" % psnr_original_meter_batch.val

            print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
            print(Confusion_Matrix + '\n' + Display_Loss_D + '\n' + Display_Loss_Cyc + '\n' + Display_Loss_G + '\n' +
                  Display_Loss_F + '\n' + Display_Loss_Sup + '\n' + Display_SSIM + '\n' + Display_PSNR)

        # Free up space in GPU
        del x_1, y_1, x_2, y_2, F_y, G_x, F_G_x__G, F_G_x__F, G_F_y__G, G_F_y__F, F_x, G_y, DX_F_y, DY_G_x

    Confusion_Matrix = "True Positive: %.6f" % np.array(TP).mean() + \
                       "\tFalse Positive: %.6f" % np.array(FP).mean() + \
                       "\nFalse Negative: %.6f" % np.array(FN).mean() + \
                       "\tTrue Negative: %.6f" % np.array(TN).mean()
    Display_Loss_D = "Loss_DX: %.6f" % loss_DX.avg + "\tLoss_DY: %.6f" % loss_DY.avg
    Display_Loss_Cyc = "loss_Cyc_XYX: %.6f" % loss_Cyc_XYX.avg + "\tLoss_Cyc_YXY: %.6f" % loss_Cyc_YXY.avg
    Display_Loss_G = "Loss_GANG: %.6f" % loss_GANG.avg + "\tLoss_IY: %.6f" % loss_IY.avg
    Display_Loss_F = "Loss_GANF: %.6f" % loss_GANF.avg + "\tLoss_IX: %.6f" % loss_IX.avg
    Display_Loss_Sup = "Loss_Sup_XY: %.6f" % loss_sup_XY.avg + "\tLoss_Sup_YX: %.6f" % loss_sup_YX.avg
    Display_SSIM = "SSIM_Batch: %.6f" % ssim_meter_batch.avg + \
                   "\tSSIM_Original_Batch: %.6f" % ssim_original_meter_batch.avg
    Display_PSNR = "PSNR_Batch: %.6f" % psnr_meter_batch.avg + \
                   "\tPSNR_Original_Batch: %.6f" % psnr_original_meter_batch.avg

    print("Training Data for Epoch: ", epoch)
    print(Confusion_Matrix + '\n' + Display_Loss_D + '\n' + Display_Loss_Cyc + '\n' + Display_Loss_G + '\n' +
          Display_Loss_F + '\n' + Display_Loss_Sup + '\n' + Display_SSIM + '\n' + Display_PSNR)

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        y_v = validation_batch['GT']

        with torch.no_grad():
            G_x_v = G(x_v.to(device_0))

            ssim_meter_val.update(SSIM(G_x_v, y_v.to(device_0)).item())
            ssim_original_meter_val.update(SSIM(x_v.to(device_1), y_v.to(device_1)).item())

            psnr_meter_val.update(PSNR(mse_0(G_x_v, y_v.to(device_0))).item())
            psnr_original_meter_val.update(PSNR(mse_1(x_v.to(device_1), y_v.to(device_1))).item())

        # Free up space in GPU
        del x_v, y_v, G_x_v

    Display_SSIM = "SSIM: %.6f" % ssim_meter_val.avg + "\tSSIM_Original: %.6f" % ssim_original_meter_val.avg
    Display_PSNR = "PSNR: %.6f" % psnr_meter_val.avg + "\tPSNR_Original: %.6f" % psnr_original_meter_val.avg

    print('\n' + '-' * 160 + '\n')
    print("Validation Data for Epoch: ", epoch)
    print(Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 160 + '\n')

    Logger.writerow({
        'Loss_DX': loss_DX.avg,
        'Loss_DY': loss_DY.avg,
        'Loss_GANG': loss_GANG.avg,
        'Loss_GANF': loss_GANF.avg,
        'Loss_Cyc_XYX': loss_Cyc_XYX.avg,
        'Loss_Cyc_YXY': loss_Cyc_YXY.avg,
        'Loss_IX': loss_IX.avg,
        'Loss_IY': loss_IY.avg,
        'Loss_Sup_XY': loss_sup_XY.avg,
        'Loss_Sup_YX': loss_sup_YX.avg,
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
    Legend_Loss = ['Loss_DX', 'Loss_DY', 'Loss_GANG', 'Loss_GANF', 'Loss_Cyc_XYX', 'Loss_Cyc_YXY', 'Loss_IX', 'Loss_IY',
                   'Loss_Sup_XY', 'Loss_Sup_YX']

    vis_window['Loss_{date}'.format(date=d1)] = vis.line(
        X=np.column_stack([epoch] * 10),
        Y=np.column_stack([loss_DX.avg, loss_DY.avg, loss_GANG.avg, loss_GANF.avg,
                           loss_Cyc_XYX.avg, loss_Cyc_YXY.avg, loss_IX.avg, loss_IY.avg,
                           loss_sup_XY.avg, loss_sup_YX.avg]),
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

    loss_DX.reset()
    loss_DY.reset()
    loss_GANG.reset()
    loss_GANF.reset()
    loss_Cyc_XYX.reset()
    loss_Cyc_YXY.reset()
    loss_IX.reset()
    loss_IY.reset()
    loss_sup_XY.reset()
    loss_sup_YX.reset()
    ssim_meter_batch.reset()
    ssim_original_meter_batch.reset()
    psnr_meter_batch.reset()
    psnr_original_meter_batch.reset()
    ssim_meter_val.reset()
    ssim_original_meter_val.reset()
    psnr_meter_val.reset()
    psnr_original_meter_val.reset()

    if epoch > 0 and not epoch % 5:
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

        if not epoch % 10:
            state_dict_DX = clip_weights(DX.state_dict(), k=3, device=device_0)
            state_dict_DX = drop_weights(state_dict_DX, p=0.95, device=device_0)
            state_dict_DY = clip_weights(DY.state_dict(), k=3, device=device_1)
            state_dict_DY = drop_weights(state_dict_DY, p=0.95, device=device_1)
            state_dict_G = clip_weights(G.state_dict(), k=3, device=device_0)
            state_dict_G = drop_weights(state_dict_G, p=0.95, device=device_0)
            state_dict_F = clip_weights(F.state_dict(), k=3, device=device_1)
            state_dict_F = drop_weights(state_dict_F, p=0.95, device=device_1)

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
