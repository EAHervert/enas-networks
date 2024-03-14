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
import time

from utilities.utils import CSVLogger
from utilities.utils import Logger as logger_util
from utilities.functions import SSIM, PSNR, generate_loggers, drop_weights, clip_weights, display_time

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

# Parser
parser = argparse.ArgumentParser(
    prog='Train_DHDN_{date}'.format(date=d1),
    description='Trains Vanilla DHDN',
)
parser.add_argument('--name', default='Default', type=str)  # Name to save Models
parser.add_argument('--noise', default='SIDD', type=str)  # Which dataset to train on
parser.add_argument('--size', type=int, default=3)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--drop', default='-1', type=float)  # Drop weights for model weight initialization
parser.add_argument('--device', default='cuda:0', type=str)  # GPU to use
parser.add_argument('--clip_weights', default=False, type=lambda s: (str(s).lower() == 'true'))  # Load previous models
parser.add_argument('--load_model', default=False, type=lambda s: (str(s).lower() == 'true'))  # Load previous models
parser.add_argument('--model_path_dhdn', default='dhdn_SIDD.pth', type=str)  # Model path dhdn
parser.add_argument('--training_path_csv', default='sidd_np_instances_064_0016.csv', type=str)  # Model path dhdn
parser.add_argument('--save_every', default='5', type=float)  # Save model every x epochs
args = parser.parse_args()


def main():
    global args

    # Hyperparameters
    dir_current = os.getcwd()
    config_path = dir_current + '/configs/config_dhdn.json'
    config = json.load(open(config_path))
    if not os.path.exists(dir_current + '/models/'):
        os.makedirs(dir_current + '/models/')

    # Noise Dataset
    if args.noise not in config['Locations'].keys():
        print('Incorrect Noise Selection!')
        exit()

    path_training = dir_current + '/instances/' + args.training_path_csv
    path_validation_noisy = dir_current + config['Locations'][args.noise]['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations'][args.noise]['Validation_GT']
    Result_Path = dir_current + '/{noise}/{date}_{name}/'.format(date=d1, name=args.name, noise=args.noise)
    Output_Path = config['Locations'][args.noise]['Output_File']
    Log_Path = Result_Path + '/' + Output_Path

    if not os.path.isdir(Result_Path):
        os.mkdir(Result_Path)

    if not os.path.isdir(Log_Path):
        os.mkdir(Log_Path)
    sys.stdout = logger_util(Log_Path + '/log.log')

    # Create the CSV Logger:
    File_Name = Log_Path + '/data.csv'
    Field_Names = ['Loss_Batch', 'Loss_Val', 'Loss_Original_Train', 'Loss_Original_Val',
                   'SSIM_Batch', 'SSIM_Val', 'SSIM_Original_Train', 'SSIM_Original_Val',
                   'PSNR_Batch', 'PSNR_Val', 'PSNR_Original_Train', 'PSNR_Original_Val']
    Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

    # Define the devices:
    device = torch.device(args.device)

    # Load the models:
    encoder = [i * 0 for i in range(3 * args.size)]
    decoder = [i * 0 for i in range(3 * args.size)]
    bottleneck = [0, 0]
    dhdn_architecture = encoder + bottleneck + decoder

    dhdn = DHDN.SharedDHDN(k_value=args.size, architecture=dhdn_architecture)
    dhdn.to(device)

    if args.load_model:
        state_dict_dhdn = torch.load(dir_current + args.model_path_dhdn, map_location=device)
        if args.drop > 0:
            state_dict_dhdn = drop_weights(state_dict_dhdn, p=args.drop, device=device)

        dhdn.load_state_dict(state_dict_dhdn)

    # Create the Visdom window:
    # This window will show the SSIM and PSNR of the different networks.
    vis = visdom.Visdom(
        server='eng-ml-01.utdallas.edu',
        port=8097,
        use_incoming_socket=False
    )

    # Display the data to the window:
    vis.env = Output_Path
    vis_window = {'SSIM_{date}'.format(date=d1): None, 'PSNR_{date}'.format(date=d1): None}

    t_init = time.time()
    # Training Optimization and Scheduling:
    optimizer = torch.optim.Adam(dhdn.parameters(), config['Training']['Learning_Rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5, -1)

    # Define the Loss and evaluation metrics:
    loss = nn.L1Loss().to(device)
    MSE = nn.MSELoss().to(device)

    # Now, let us define our loggers:
    loggers0 = generate_loggers()

    # Training Batches
    loss_batch, loss_original_batch, ssim_batch, ssim_original_batch, psnr_batch, psnr_original_batch = loggers0[0]

    # Validation Batches
    loss_batch_val, loss_original_batch_val, ssim_batch_val, ssim_original_batch_val = loggers0[1][0:4]
    psnr_batch_val, psnr_original_batch_val = loggers0[1][4:]

    # Load the Training and Validation Data:
    dataset_training = dataset.DatasetNoise(csv_file=path_training,
                                            transform=dataset.RandomProcessing(),
                                            device=device)
    dataset_validation = dataset.DatasetMAT(mat_noisy_file=path_validation_noisy,
                                            mat_gt_file=path_validation_gt,
                                            device=device)

    dataloader_training = DataLoader(dataset=dataset_training,
                                     batch_size=config['Training']['Train_Batch_Size'],
                                     shuffle=True)
    dataloader_validation = DataLoader(dataset=dataset_validation,
                                       batch_size=config['Training']['Validation_Batch_Size'],
                                       shuffle=False)

    for epoch in range(args.epochs):
        for i_batch, sample_batch in enumerate(dataloader_training):
            x = sample_batch['NOISY']
            y0 = dhdn(x.to(device))
            t = sample_batch['GT']

            loss_value = loss(y0, t)
            loss_batch.update(loss_value.item())

            # Calculate values not needing to be backpropagated
            with torch.no_grad():
                loss_original_batch.update(loss(x, t).item())
                ssim_batch.update(SSIM(y0, t).item())
                ssim_original_batch.update(SSIM(x, t).item())
                psnr_batch.update(PSNR(MSE(y0, t)).item())
                psnr_original_batch.update(PSNR(MSE(x, t)).item())

            # Backpropagate to train model
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            if i_batch % 100 == 0:
                Display_Loss = "Loss_DHDN: %.6f" % loss_batch.val + "\tLoss_Original: %.6f" % loss_original_batch.val
                Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
                Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

                print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
                print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

            # Free up space in GPU
            del x, y0, t

        Display_Loss = "Loss_DHDN: %.6f" % loss_batch.avg + "\tLoss_Original: %.6f" % loss_original_batch.avg
        Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch.avg + "\tSSIM_Original: %.6f" % ssim_original_batch.avg
        Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch.avg + "\tPSNR_Original: %.6f" % psnr_original_batch.avg

        print("\nTotal Training Data for Epoch: ", epoch)
        print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')

        for i_validation, validation_batch in enumerate(dataloader_validation):
            with torch.no_grad():
                x_v = validation_batch['NOISY']
                t_v = validation_batch['GT']
                y_v0 = dhdn(x_v)

                loss_batch_val.update(loss(y_v0, t_v).item())
                loss_original_batch_val.update(loss(x_v, t_v).item())
                ssim_batch_val.update(SSIM(y_v0, t_v).item())
                ssim_original_batch_val.update(SSIM(x_v, t_v).item())
                psnr_batch_val.update(PSNR(MSE(y_v0, t_v)).item())
                psnr_original_batch_val.update(PSNR(MSE(x_v, t_v)).item())

            # Free up space in GPU
            del x_v, y_v0, t_v

        Display_Loss = "Loss_DHDN: %.6f" % loss_batch_val.avg + "\tLoss_Original: %.6f" % loss_original_batch_val.avg
        Display_SSIM = "SSIM_DHDN: %.6f" % ssim_batch_val.avg + "\tSSIM_Original: %.6f" % ssim_original_batch_val.avg
        Display_PSNR = "PSNR_DHDN: %.6f" % psnr_batch_val.avg + "\tPSNR_Original: %.6f" % psnr_original_batch_val.avg

        print("Validation Data for Epoch: ", epoch)
        print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
        print('-' * 160 + '\n')

        Logger.writerow({
            'Loss_Batch': loss_batch.avg,
            'Loss_Val': loss_batch_val.avg,
            'Loss_Original_Train': loss_original_batch.avg,
            'Loss_Original_Val': loss_original_batch_val.avg,
            'SSIM_Batch': ssim_batch.avg,
            'SSIM_Val': ssim_batch_val.avg,
            'SSIM_Original_Train': ssim_original_batch.avg,
            'SSIM_Original_Val': ssim_original_batch_val.avg,
            'PSNR_Batch': psnr_batch.avg,
            'PSNR_Val': psnr_batch_val.avg,
            'PSNR_Original_Train': psnr_original_batch.avg,
            'PSNR_Original_Val': psnr_original_batch_val.avg
        })

        Legend = ['DHDN_Train', 'Orig_Train', 'DHDN_Val', 'Orig_Val']

        vis_window['SSIM_{date}'.format(date=d1)] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack(
                [ssim_batch.avg, ssim_original_batch.avg, ssim_batch_val.avg, ssim_original_batch_val.avg]),
            win=vis_window['SSIM_{date}'.format(date=d1)],
            opts=dict(title='SSIM_{date}'.format(date=d1), xlabel='Epoch', ylabel='SSIM', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window['PSNR_{date}'.format(date=d1)] = vis.line(
            X=np.column_stack([epoch] * 4),
            Y=np.column_stack(
                [psnr_batch.avg, psnr_original_batch.avg, psnr_batch_val.avg, psnr_original_batch_val.avg]),
            win=vis_window['PSNR_{date}'.format(date=d1)],
            opts=dict(title='PSNR_{date}'.format(date=d1), xlabel='Epoch', ylabel='PSNR', legend=Legend),
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

        scheduler.step()

        # Save every validation instance
        if epoch > 0 and not epoch % args.save_every:
            model_path_0 = dir_current + '/models/{date}_dhdn_{noise}_{name}_{epoch}.pth'.format(date=d1, epoch=epoch,
                                                                                                 name=args.name,
                                                                                                 noise=args.noise)
            torch.save(dhdn.state_dict(), model_path_0)
            # modify weights to avoid overfitting
            if args.clip_weights:
                state_dict_dhdn = clip_weights(dhdn.state_dict(), k=3, device=device)
                dhdn.load_state_dict(state_dict_dhdn)

    t_final = time.time()
    display_time(t_final - t_init)

    # Save final model
    model_path_0 = dir_current + '/models/{date}_dhdn_{noise}_{name}.pth'.format(date=d1, name=args.name,
                                                                                 noise=args.noise)
    torch.save(dhdn.state_dict(), model_path_0)


if __name__ == "__main__":
    main()
