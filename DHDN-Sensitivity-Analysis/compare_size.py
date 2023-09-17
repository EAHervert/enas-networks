import os
import sys
from utilities import dataset
from ENAS_DHDN import SHARED_DHDN as DHDN
import time
from datetime import date
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import visdom
import argparse

from utilities.utils import CSVLogger, Logger
from utilities.functions import SSIM, PSNR, generate_loggers
#
# Parser
parser = argparse.ArgumentParser(
                    prog='DHDN_Compare_Size',
                    description='Compares 3 sizes of models based on the DHDN architecture',
                    )
parser.add_argument('Noise')  # positional argument
args = parser.parse_args()
#
# Hyperparameters
config_path = os.getcwd() + '/configs/config_compare_size.json'
config = json.load(open(config_path))
#
config['Locations']['Output_File'] += '_' + str(args.Noise)

today = date.today()  # Date to label the models
if args.Noise == 'SSID':
    path_training = os.getcwd() + '/instances/sidd_np_instances_064.csv'
    path_validation = os.getcwd() + '/instances/sidd_np_instances_256.csv'
    Result_Path = os.getcwd() + '/SIDD/'
else:
    path_training = os.getcwd() + '/instances/davis_np_instances_128.csv'
    path_validation = os.getcwd() + '/instances/davis_np_instances_256.csv'
    Result_Path = os.getcwd() + '/{noise}/'.format(noise=args.Noise)

if not os.path.isdir(Result_Path):
    os.mkdir(Result_Path)

if not os.path.isdir(Result_Path + '/' + config['Locations']['Output_File']):
    os.mkdir(Result_Path + '/' + config['Locations']['Output_File'])
sys.stdout = Logger(Result_Path + '/' + config['Locations']['Output_File'] + 'log.log')
#
# Create the CSV Logger:
File_Name = Result_Path + '/' + config['Locations']['Output_File'] + '/data.csv'
Field_Names = ['Loss_Batch0', 'Loss_Batch1', 'Loss_Val0', 'Loss_Val1', 'Loss_Original_Train', 'Loss_Original_Val',
               'SSIM_Batch0', 'SSIM_Batch1', 'SSIM_Val0', 'SSIM_Val1', 'SSIM_Original_Train', 'SSIM_Original_Val',
               'PSNR_Batch0', 'PSNR_Batch1', 'PSNR_Val0', 'PSNR_Val1', 'PSNR_Original_Train', 'PSNR_Original_Val']
Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)
#
# Define the devices:
if config['CUDA']['Device'] != 'None':
    device = torch.device(config['CUDA']['Device'])
else:
    device = torch.device("cpu")

# Load the Models:
# Size 5 - Two steps Down, Two steps Up
encoder_5, bottleneck_5, decoder_5 = [0, 0, 0, 0, 0, 0], [0, 0], [0, 0, 0, 0, 0, 0]
architecture_5 = encoder_5 + bottleneck_5 + decoder_5
DHDN_5 = DHDN.SharedDHDN(k_value=2, channels=128, architecture=architecture_5)

# Cast to GPU(s)
if (config['CUDA']['DataParallel']):
    DHDN_5 = nn.DataParallel(DHDN_5)

DHDN_5 = DHDN_5.to(device0)

# Size 7
encoder_7 = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Three Steps Down
bottleneck_7 = [0, 0]
decoder_7 = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Three Steps Up
architecture_7 = encoder_7 + bottleneck_7 + decoder_7

DHDN_7 = DHDN.SharedDHDN(k_value=3, channels=128, architecture=architecture_7)

## Cast to GPU(s)
if (args.DataParallel == 1):
    DHDN_7 = nn.DataParallel(DHDN_7)

DHDN_7 = DHDN_7.to(Device)

# Size 9
encoder_9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Four Steps Down
bottleneck_9 = [0, 0]
# decoder_9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Four steps Up
# architecture_9 = encoder_9 + bottleneck_9 + decoder_9
#
# DHDN_9 = DHDN.SharedDHDN(k_value=4, channels=128, architecture=architecture_9)
#
# ## Cast to GPU(s)
# if (args.DataParallel == 1):
#     DHDN_9 = nn.DataParallel(DHDN_9)
#
# DHDN_9 = DHDN_9.to(Device)
#
# ## Create the Visdom window:
# ## This window will show the SSIM and PSNR of the different networks.
# vis = visdom.Visdom(
#     server='eng-ml-01.utdallas.edu',
#     port=8097,
#     use_incoming_socket=False
# )
#
# ## Display the data to the window:
# vis.env = 'DHDN_Compare_Size'
# vis_window = {
#     'DHDN_5_SSIM': None, 'DHDN_7_SSIM': None, 'DHDN_9_SSIM': None,
#     'DHDN_5_PSNR': None, 'DHDN_7_PSNR': None, 'DHDN_9_PSNR': None,
#     'DHDN_Compare_SSIM_Train': None, 'DHDN_Compare_SSIM_Val': None,
#     'DHDN_Compare_PSNR_Train': None, 'DHDN_Compare_PSNR_Val': None
# }
#
# ## Load the Validation Data:
# if (args.Instances == 1):
#     with open(args.Instances_Location + 'Training_List_1.json', 'r') as f:
#         Training_List = json.load(f)
#
#     with open(args.Instances_Location + 'Validation_List_1.json', 'r') as f:
#         Validation_List = json.load(f)
#
# elif (args.Instances == 2):
#     with open(args.Instances_Location + 'Training_List_2.json', 'r') as f:
#         Training_List = json.load(f)
#
#     with open(args.Instances_Location + 'Validation_List_2.json', 'r') as f:
#         Validation_List = json.load(f)
#
# elif (args.Instances == 3):
#     with open(args.Instances_Location + 'Training_List_3.json', 'r') as f:
#         Training_List = json.load(f)
#
#     with open(args.Instances_Location + 'Validation_List_3.json', 'r') as f:
#         Validation_List = json.load(f)
#
# else:
#     print("Incorrect Instance Number")
#     exit()
#
# ## Define the optimizers:
# Optimizer_5 = torch.optim.Adam(DHDN_5.parameters(), args.Learning_Rate)
# Optimizer_7 = torch.optim.Adam(DHDN_7.parameters(), args.Learning_Rate)
# Optimizer_9 = torch.optim.Adam(DHDN_9.parameters(), args.Learning_Rate)
#
# ## Define the Scheduling:
# Scheduler_5 = torch.optim.lr_scheduler.StepLR(Optimizer_5, 3, 0.5, -1)
# Scheduler_7 = torch.optim.lr_scheduler.StepLR(Optimizer_7, 3, 0.5, -1)
# Scheduler_9 = torch.optim.lr_scheduler.StepLR(Optimizer_9, 3, 0.5, -1)
#
# ## Define the Loss and evaluation metrics:
# Loss = nn.L1Loss().to(Device)
# MSE = nn.MSELoss().to(Device)
#
# ## Now, let us define our loggers:
#
# ## Image Batches
# Loss_5_Meter = AverageMeter()
# Loss_7_Meter = AverageMeter()
# Loss_9_Meter = AverageMeter()
# Loss_Original_Meter = AverageMeter()
#
# SSIM_5_Meter = AverageMeter()
# SSIM_7_Meter = AverageMeter()
# SSIM_9_Meter = AverageMeter()
# SSIM_Original_Meter = AverageMeter()
#
# PSNR_5_Meter = AverageMeter()
# PSNR_7_Meter = AverageMeter()
# PSNR_9_Meter = AverageMeter()
# PSNR_Original_Meter = AverageMeter()
#
# ## Total Training
# Loss_5_Meter_Train = AverageMeter()
# Loss_7_Meter_Train = AverageMeter()
# Loss_9_Meter_Train = AverageMeter()
# Loss_Original_Meter_Train = AverageMeter()
#
# SSIM_5_Meter_Train = AverageMeter()
# SSIM_7_Meter_Train = AverageMeter()
# SSIM_9_Meter_Train = AverageMeter()
# SSIM_Original_Meter_Train = AverageMeter()
#
# PSNR_5_Meter_Train = AverageMeter()
# PSNR_7_Meter_Train = AverageMeter()
# PSNR_9_Meter_Train = AverageMeter()
# PSNR_Original_Meter_Train = AverageMeter()
#
# ## Validation
# Loss_5_Meter_Val = AverageMeter()
# Loss_7_Meter_Val = AverageMeter()
# Loss_9_Meter_Val = AverageMeter()
# Loss_Original_Meter_Val = AverageMeter()
#
# SSIM_5_Meter_Val = AverageMeter()
# SSIM_7_Meter_Val = AverageMeter()
# SSIM_9_Meter_Val = AverageMeter()
# SSIM_Original_Meter_Val = AverageMeter()
#
# PSNR_5_Meter_Val = AverageMeter()
# PSNR_7_Meter_Val = AverageMeter()
# PSNR_9_Meter_Val = AverageMeter()
# PSNR_Original_Meter_Val = AverageMeter()
#
# t_init = time.time()
#
# ## Now, let us loop through the training Data:
# for Epoch in range(args.Num_Epochs):
#
#     t0 = time.time()
#
#     Batches_Num = 0
#
#     Batches_Training = Create_Batches(Training_List, args.Train_Image_Batch)
#
#     ## Loop through the training batches:
#     for i in Batches_Training:
#
#         t00 = time.time()
#
#         Batches_Num += 1
#
#         Loader = Load_Dataset_Images(i, Size_N=args.Train_N, Size_M=args.Train_M)
#
#         ## Loop through the batches:
#         for i, (Input, Target) in enumerate(Loader):
#             ## Randomly Flip and rotate the tensors:
#             Input, Target = Rand_Mod(Input, Target)
#
#             ## Cast to Cuda:
#             Input = Input.to(Device)
#             Target = Target.to(Device)
#
#             ## Get the Outputs of each network:
#             Output_5 = DHDN_5(Input, Architecture_5)
#             Output_7 = DHDN_7(Input, Architecture_7)
#             Output_9 = DHDN_9(Input, Architecture_9)
#
#             ## Calculate the losses:
#             Loss_5 = Loss(Output_5, Target)
#             Loss_7 = Loss(Output_7, Target)
#             Loss_9 = Loss(Output_9, Target)
#             with torch.no_grad():
#                 Loss_Original = Loss(Input, Target)
#
#             Loss_5_Meter_Train.update(Loss_5.item())
#             Loss_7_Meter_Train.update(Loss_7.item())
#             Loss_9_Meter_Train.update(Loss_9.item())
#             Loss_Original_Meter_Train.update(Loss_Original.item())
#
#             ## Calculate SSIM:
#             with torch.no_grad():
#                 SSIM_5 = SSIM(Output_5, Target)
#                 SSIM_7 = SSIM(Output_7, Target)
#                 SSIM_9 = SSIM(Output_9, Target)
#                 SSIM_Original = SSIM(Input, Target)
#
#             SSIM_5_Meter_Train.update(SSIM_5.item())
#             SSIM_7_Meter_Train.update(SSIM_7.item())
#             SSIM_9_Meter_Train.update(SSIM_9.item())
#             SSIM_Original_Meter_Train.update(SSIM_Original.item())
#
#             ## Calculate PSNR:
#             with torch.no_grad():
#                 MSE_5 = MSE(Output_5, Target)
#                 MSE_7 = MSE(Output_7, Target)
#                 MSE_9 = MSE(Output_9, Target)
#                 MSE_Original = MSE(Input, Target)
#
#             PSNR_5 = PSNR(MSE_5)
#             PSNR_7 = PSNR(MSE_7)
#             PSNR_9 = PSNR(MSE_9)
#             PSNR_Original = PSNR(MSE_Original)
#
#             PSNR_5_Meter_Train.update(PSNR_5.item())
#             PSNR_7_Meter_Train.update(PSNR_7.item())
#             PSNR_9_Meter_Train.update(PSNR_9.item())
#             PSNR_Original_Meter_Train.update(PSNR_Original.item())
#
#             ## Back Progagate through the batch:
#             Optimizer_5.zero_grad()
#             Loss_5.backward()
#             Optimizer_5.step()
#
#             Optimizer_7.zero_grad()
#             Loss_7.backward()
#             Optimizer_7.step()
#
#             Optimizer_9.zero_grad()
#             Loss_9.backward()
#             Optimizer_9.step()
#
#             ## Update Image Batches logger:
#             Loss_5_Meter.update(Loss_5.item())
#             Loss_7_Meter.update(Loss_7.item())
#             Loss_9_Meter.update(Loss_9.item())
#             Loss_Original_Meter.update(Loss_Original.item())
#             SSIM_5_Meter.update(SSIM_5.item())
#             SSIM_7_Meter.update(SSIM_7.item())
#             SSIM_9_Meter.update(SSIM_9.item())
#             SSIM_Original_Meter.update(SSIM_Original.item())
#             PSNR_5_Meter.update(PSNR_5.item())
#             PSNR_7_Meter.update(PSNR_7.item())
#             PSNR_9_Meter.update(PSNR_9.item())
#             PSNR_Original_Meter.update(PSNR_Original.item())
#
#         t01 = time.time()
#
#         print("Training Data for Epoch: ", Epoch, "Image Batch: ", Batches_Num)
#
#         Display_Loss = "Loss_DHDN_5: %.6f" % (Loss_5_Meter.avg) + \
#                        "\tLoss_DHDN_7: %.6f" % (Loss_7_Meter.avg) + \
#                        "\tLoss_DHDN_9: %.6f" % (Loss_9_Meter.avg) + \
#                        "\tLoss_Original: %.6f" % (Loss_Original_Meter.avg)
#         print(Display_Loss)
#
#         Display_SSIM = "SSIM_DHDN_5: %.6f" % (SSIM_5_Meter.avg) + \
#                        "\tSSIM_DHDN_7: %.6f" % (SSIM_7_Meter.avg) + \
#                        "\tSSIM_DHDN_9: %.6f" % (SSIM_9_Meter.avg) + \
#                        "\tSSIM_Original: %.6f" % (SSIM_Original_Meter.avg)
#         print(Display_SSIM)
#
#         Display_PSNR = "PSNR_DHDN_5: %.6f" % (PSNR_5_Meter.avg) + \
#                        "\tPSNR_DHDN_7: %.6f" % (PSNR_7_Meter.avg) + \
#                        "\tPSNR_DHDN_9: %.6f" % (PSNR_9_Meter.avg) + \
#                        "\tPSNR_Original: %.6f" % (PSNR_Original_Meter.avg)
#         print(Display_PSNR)
#
#         Display_Time(t01 - t00)
#
#         ## Reset the loggers:
#         Loss_5_Meter.reset()
#         Loss_7_Meter.reset()
#         Loss_9_Meter.reset()
#         Loss_Original_Meter.reset()
#
#         SSIM_5_Meter.reset()
#         SSIM_7_Meter.reset()
#         SSIM_9_Meter.reset()
#         SSIM_Original_Meter.reset()
#
#         PSNR_5_Meter.reset()
#         PSNR_7_Meter.reset()
#         PSNR_9_Meter.reset()
#         PSNR_Original_Meter.reset()
#
#     t1 = time.time()
#
#     print('-' * 160)
#
#     print("Total Training Data for Epoch: ", Epoch)
#
#     Display_Loss = "Loss_DHDN_5: %.6f" % (Loss_5_Meter_Train.avg) + \
#                    "\tLoss_DHDN_7: %.6f" % (Loss_7_Meter_Train.avg) + \
#                    "\tLoss_DHDN_9: %.6f" % (Loss_9_Meter_Train.avg) + \
#                    "\tLoss_Original: %.6f" % (Loss_Original_Meter_Train.avg)
#     print(Display_Loss)
#
#     Display_SSIM = "SSIM_DHDN_5: %.6f" % (SSIM_5_Meter_Train.avg) + \
#                    "\tSSIM_DHDN_7: %.6f" % (SSIM_7_Meter_Train.avg) + \
#                    "\tSSIM_DHDN_9: %.6f" % (SSIM_9_Meter_Train.avg) + \
#                    "\tSSIM_Original: %.6f" % (SSIM_Original_Meter_Train.avg)
#     print(Display_SSIM)
#
#     Display_PSNR = "PSNR_DHDN_5: %.6f" % (PSNR_5_Meter_Train.avg) + \
#                    "\tPSNR_DHDN_7: %.6f" % (PSNR_7_Meter_Train.avg) + \
#                    "\tPSNR_DHDN_9: %.6f" % (PSNR_9_Meter_Train.avg) + \
#                    "\tPSNR_Original: %.6f" % (PSNR_Original_Meter_Train.avg)
#     print(Display_PSNR)
#
#     for param_group in Optimizer_5.param_groups:
#         LR = param_group['lr']
#         print("Learning Rate: ", LR)
#
#     Display_Time(t1 - t0)
#
#     Batches_Validation = Create_Batches(Validation_List, args.Validation_Image_Batch)
#
#     t2 = time.time()
#
#     ## Loop through the training batches:
#     for i in Batches_Validation:
#
#         Loader = Load_Dataset_Images(i, Size_Crop=256, Size_N=args.Validation_N, Size_M=args.Validation_M)
#
#         ## Loop through the batches:
#         for i, (Input, Target) in enumerate(Loader):
#             ## Randomly Flip and rotate the tensors:
#             Input, Target = Rand_Mod(Input, Target)
#
#             ## Cast to Cuda:
#             Input = Input.to(Device).detach()
#             Target = Target.to(Device).detach()
#
#             ## Get the Outputs of each network:
#             with torch.no_grad():
#                 Output_5 = DHDN_5(Input, Architecture_5)
#                 Output_7 = DHDN_7(Input, Architecture_7)
#                 Output_9 = DHDN_9(Input, Architecture_9)
#
#             ## Calculate the losses:
#             with torch.no_grad():
#                 Loss_5 = Loss(Output_5, Target)
#                 Loss_7 = Loss(Output_7, Target)
#                 Loss_9 = Loss(Output_9, Target)
#                 Loss_Original = Loss(Input, Target)
#
#             Loss_5_Meter_Val.update(Loss_5.item())
#             Loss_7_Meter_Val.update(Loss_7.item())
#             Loss_9_Meter_Val.update(Loss_9.item())
#             Loss_Original_Meter_Val.update(Loss_Original.item())
#
#             ## Calculate SSIM:
#             with torch.no_grad():
#                 SSIM_5 = SSIM(Output_5, Target)
#                 SSIM_7 = SSIM(Output_7, Target)
#                 SSIM_9 = SSIM(Output_9, Target)
#                 SSIM_Original = SSIM(Input, Target)
#
#             SSIM_5_Meter_Val.update(SSIM_5.item())
#             SSIM_7_Meter_Val.update(SSIM_7.item())
#             SSIM_9_Meter_Val.update(SSIM_9.item())
#             SSIM_Original_Meter_Val.update(SSIM_Original.item())
#
#             ## Calculate PSNR:
#             with torch.no_grad():
#                 MSE_5 = MSE(Output_5, Target)
#                 MSE_7 = MSE(Output_7, Target)
#                 MSE_9 = MSE(Output_9, Target)
#                 MSE_Original = MSE(Input, Target)
#
#             PSNR_5 = PSNR(MSE_5)
#             PSNR_7 = PSNR(MSE_7)
#             PSNR_9 = PSNR(MSE_9)
#             PSNR_Original = PSNR(MSE_Original)
#
#             PSNR_5_Meter_Val.update(PSNR_5.item())
#             PSNR_7_Meter_Val.update(PSNR_7.item())
#             PSNR_9_Meter_Val.update(PSNR_9.item())
#             PSNR_Original_Meter_Val.update(PSNR_Original.item())
#
#     t3 = time.time()
#
#     print('-' * 160)
#
#     print("Validation Data for Epoch: ", Epoch)
#     Display_Loss = "Loss_DHDN_5: %.6f" % (Loss_5_Meter_Val.avg) + \
#                    "\tLoss_DHDN_7: %.6f" % (Loss_7_Meter_Val.avg) + \
#                    "\tLoss_DHDN_9: %.6f" % (Loss_9_Meter_Val.avg) + \
#                    "\tLoss_Original: %.6f" % (Loss_Original_Meter_Val.avg)
#     print(Display_Loss)
#
#     Display_SSIM = "SSIM_DHDN_5: %.6f" % (SSIM_5_Meter_Val.avg) + \
#                    "\tSSIM_DHDN_7: %.6f" % (SSIM_7_Meter_Val.avg) + \
#                    "\tSSIM_DHDN_9: %.6f" % (SSIM_9_Meter_Val.avg) + \
#                    "\tSSIM_Original: %.6f" % (SSIM_Original_Meter_Val.avg)
#     print(Display_SSIM)
#
#     Display_PSNR = "PSNR_DHDN_5: %.6f" % (PSNR_5_Meter_Val.avg) + \
#                    "\tPSNR_DHDN_7: %.6f" % (PSNR_7_Meter_Val.avg) + \
#                    "\tPSNR_DHDN_9: %.6f" % (PSNR_9_Meter_Val.avg) + \
#                    "\tPSNR_Original: %.6f" % (PSNR_Original_Meter_Val.avg)
#     print(Display_PSNR)
#
#     Display_Time(t3 - t2)
#
#     print('-' * 160)
#     print()
#
#     Logger.writerow({
#         'Loss_5_Train': Loss_5_Meter_Train.avg,
#         'Loss_5_Val': Loss_5_Meter_Val.avg,
#         'Loss_7_Train': Loss_7_Meter_Train.avg,
#         'Loss_7_Val': Loss_7_Meter_Val.avg,
#         'Loss_9_Train': Loss_9_Meter_Train.avg,
#         'Loss_9_Val': Loss_9_Meter_Val.avg,
#         'Loss_Original_Train': Loss_Original_Meter_Train.avg,
#         'Loss_Original_Val': Loss_Original_Meter_Val.avg,
#         'SSIM_5_Train': SSIM_5_Meter_Train.avg,
#         'SSIM_5_Val': SSIM_5_Meter_Val.avg,
#         'SSIM_7_Train': SSIM_7_Meter_Train.avg,
#         'SSIM_7_Val': SSIM_7_Meter_Val.avg,
#         'SSIM_9_Train': SSIM_9_Meter_Train.avg,
#         'SSIM_9_Val': SSIM_9_Meter_Val.avg,
#         'SSIM_Original_Train': SSIM_Original_Meter_Train.avg,
#         'SSIM_Original_Val': SSIM_Original_Meter_Val.avg,
#         'PSNR_5_Train': PSNR_5_Meter_Train.avg,
#         'PSNR_5_Val': PSNR_5_Meter_Val.avg,
#         'PSNR_7_Train': PSNR_7_Meter_Train.avg,
#         'PSNR_7_Val': PSNR_7_Meter_Val.avg,
#         'PSNR_9_Train': PSNR_9_Meter_Train.avg,
#         'PSNR_9_Val': PSNR_9_Meter_Val.avg,
#         'PSNR_Original_Train': PSNR_Original_Meter_Train.avg,
#         'PSNR_Original_Val': PSNR_Original_Meter_Val.avg
#     })
#
#     Legend = ['Train', 'Val', 'Original_Train', 'Original_Val']
#
#     vis_window['DHDN_5_SSIM'] = vis.line(
#         X=np.column_stack([Epoch] * 4),
#         Y=np.column_stack([SSIM_5_Meter_Train.avg, SSIM_5_Meter_Val.avg,
#                            SSIM_Original_Meter_Train.avg, SSIM_Original_Meter_Val.avg]),
#         win=vis_window['DHDN_5_SSIM'],
#         opts=dict(title='DHDN_5_SSIM', xlabel='Epoch', ylabel='SSIM', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     vis_window['DHDN_7_SSIM'] = vis.line(
#         X=np.column_stack([Epoch] * 4),
#         Y=np.column_stack([SSIM_7_Meter_Train.avg, SSIM_7_Meter_Val.avg,
#                            SSIM_Original_Meter_Train.avg, SSIM_Original_Meter_Val.avg]),
#         win=vis_window['DHDN_7_SSIM'],
#         opts=dict(title='DHDN_7_SSIM', xlabel='Epoch', ylabel='SSIM', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     vis_window['DHDN_9_SSIM'] = vis.line(
#         X=np.column_stack([Epoch] * 4),
#         Y=np.column_stack([SSIM_9_Meter_Train.avg, SSIM_9_Meter_Val.avg,
#                            SSIM_Original_Meter_Train.avg, SSIM_Original_Meter_Val.avg]),
#         win=vis_window['DHDN_9_SSIM'],
#         opts=dict(title='DHDN_9_SSIM', xlabel='Epoch', ylabel='SSIM', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     vis_window['DHDN_5_PSNR'] = vis.line(
#         X=np.column_stack([Epoch] * 4),
#         Y=np.column_stack([PSNR_5_Meter_Train.avg, PSNR_5_Meter_Val.avg,
#                            PSNR_Original_Meter_Train.avg, PSNR_Original_Meter_Val.avg]),
#         win=vis_window['DHDN_5_PSNR'],
#         opts=dict(title='DHDN_5_PSNR', xlabel='Epoch', ylabel='PSNR', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     vis_window['DHDN_7_PSNR'] = vis.line(
#         X=np.column_stack([Epoch] * 4),
#         Y=np.column_stack([PSNR_7_Meter_Train.avg, PSNR_7_Meter_Val.avg,
#                            PSNR_Original_Meter_Train.avg, PSNR_Original_Meter_Val.avg]),
#         win=vis_window['DHDN_7_PSNR'],
#         opts=dict(title='DHDN_7_PSNR', xlabel='Epoch', ylabel='PSNR', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     vis_window['DHDN_9_PSNR'] = vis.line(
#         X=np.column_stack([Epoch] * 4),
#         Y=np.column_stack([PSNR_9_Meter_Train.avg, PSNR_9_Meter_Val.avg,
#                            PSNR_Original_Meter_Train.avg, PSNR_Original_Meter_Val.avg]),
#         win=vis_window['DHDN_9_PSNR'],
#         opts=dict(title='DHDN_9_PSNR', xlabel='Epoch', ylabel='PSNR', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     Legend = ['Size_5', 'Size_7', 'Size_9']
#
#     vis_window['DHDN_Compare_SSIM_Train'] = vis.line(
#         X=np.column_stack([Epoch] * 3),
#         Y=np.column_stack([SSIM_5_Meter_Train.avg, SSIM_7_Meter_Train.avg,
#                            SSIM_9_Meter_Train.avg]),
#         win=vis_window['DHDN_Compare_SSIM_Train'],
#         opts=dict(title='DHDN_Compare_SSIM_Train', xlabel='Epoch', ylabel='SSIM', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     vis_window['DHDN_Compare_SSIM_Val'] = vis.line(
#         X=np.column_stack([Epoch] * 3),
#         Y=np.column_stack([SSIM_5_Meter_Val.avg, SSIM_7_Meter_Val.avg, SSIM_9_Meter_Val.avg]),
#         win=vis_window['DHDN_Compare_SSIM_Val'],
#         opts=dict(title='DHDN_Compare_SSIM_Val', xlabel='Epoch', ylabel='SSIM', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     vis_window['DHDN_Compare_PSNR_Train'] = vis.line(
#         X=np.column_stack([Epoch] * 3),
#         Y=np.column_stack([PSNR_5_Meter_Train.avg, PSNR_7_Meter_Train.avg, PSNR_9_Meter_Train.avg]),
#         win=vis_window['DHDN_Compare_PSNR_Train'],
#         opts=dict(title='DHDN_Compare_PSNR_Train', xlabel='Epoch', ylabel='PSNR', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     vis_window['DHDN_Compare_PSNR_Val'] = vis.line(
#         X=np.column_stack([Epoch] * 3),
#         Y=np.column_stack([PSNR_5_Meter_Val.avg, PSNR_7_Meter_Val.avg, PSNR_9_Meter_Val.avg]),
#         win=vis_window['DHDN_Compare_PSNR_Val'],
#         opts=dict(title='DHDN_Compare_PSNR_Val', xlabel='Epoch', ylabel='PSNR', legend=Legend),
#         update='append' if Epoch > 0 else None)
#
#     ## Now, let us reset our loggers:
#     ## Training
#     Loss_5_Meter_Train.reset()
#     Loss_7_Meter_Train.reset()
#     Loss_9_Meter_Train.reset()
#     Loss_Original_Meter_Train.reset()
#
#     SSIM_5_Meter_Train.reset()
#     SSIM_7_Meter_Train.reset()
#     SSIM_9_Meter_Train.reset()
#     SSIM_Original_Meter_Train.reset()
#
#     PSNR_5_Meter_Train.reset()
#     PSNR_7_Meter_Train.reset()
#     PSNR_9_Meter_Train.reset()
#     PSNR_Original_Meter_Train.reset()
#
#     ## Validation
#     Loss_5_Meter_Val.reset()
#     Loss_7_Meter_Val.reset()
#     Loss_9_Meter_Val.reset()
#     Loss_Original_Meter_Val.reset()
#
#     SSIM_5_Meter_Val.reset()
#     SSIM_7_Meter_Val.reset()
#     SSIM_9_Meter_Val.reset()
#     SSIM_Original_Meter_Val.reset()
#
#     PSNR_5_Meter_Val.reset()
#     PSNR_7_Meter_Val.reset()
#     PSNR_9_Meter_Val.reset()
#     PSNR_Original_Meter_Val.reset()
#
#     ## Adjust Learning Rate:
#     Scheduler_5.step()
#     Scheduler_7.step()
#     Scheduler_9.step()
#
# t_fin = time.time()
#
# Display_Time(t_fin - t_init)
#
# ## Save the parameters:
# Path_5 = Result_Path + '/' + args.Output_File + '/DHDN_5_Parameters'
# Path_7 = Result_Path + '/' + args.Output_File + '/DHDN_7_Parameters'
# Path_9 = Result_Path + '/' + args.Output_File + '/DHDN_9_Parameters'
#
# torch.save(DHDN_5.state_dict(), Path_5)
# torch.save(DHDN_7.state_dict(), Path_7)
# torch.save(DHDN_9.state_dict(), Path_9)
