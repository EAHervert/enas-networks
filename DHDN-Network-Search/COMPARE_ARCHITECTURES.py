import os
import sys
from utilities import dataset
from ENAS_DHDN import SHARED_DHDN
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from utilities.utils import CSVLogger, Logger
from utilities.functions import list_of_ints
from ENAS_DHDN.TRAINING_FUNCTIONS import get_eval_accuracy

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

default_arc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
parser = argparse.ArgumentParser(description='ENAS_SEARCH_DHDN')

parser.add_argument('--output_file', default='Evaluate_Shared_DHDN', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--arc_1', default=default_arc, type=list_of_ints)  # Encoder of the DHDN
parser.add_argument('--arc_2', default=default_arc, type=list_of_ints)  # Encoder of the DHDN
parser.add_argument('--sample_size', type=int, default=-1)  # How many samples from validation to evaluate
parser.add_argument('--load_shared', default=False, type=lambda x: (str(x).lower() == 'true'))  # Load shared model(s)
parser.add_argument('--model_shared_path', default='2023_12_15__16_25_17/shared_network_parameters.pth', type=str)
parser.add_argument('--device', default='cuda:0', type=str)  # GPU to use
# Put shared network on two devices instead of one
parser.add_argument('--data_parallel', default=True, type=lambda x: (str(x).lower() == 'true'))
# To do outer sums for models
parser.add_argument('--outer_sum', default=False, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()


# Now, let us run all these pieces and have out program train the controller.
def main():
    global args

    current_time = datetime.datetime.now()
    d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

    dir_current = os.getcwd()  # Hyperparameters
    config_path = dir_current + '/configs/config_compare.json'
    config = json.load(open(config_path))
    model_shared_path = '/models/' + args.model_shared_path

    device_0 = torch.device(args.device)  # Define the devices
    samples = None if args.sample_size == -1 else args.sample_size
    # Create the CSV Logger:
    Result_Path = 'results/' + args.output_file + '/' + d1
    if not os.path.isdir(Result_Path):
        os.mkdir(Result_Path)

    File_Name = Result_Path + '/data.csv'
    Field_Names = ['Shared_Network', 'Architecture_1', 'Architecture_2',
                   'Loss_1', 'Loss_2', 'Loss_Original',
                   'SSIM_1', 'SSIM_2', 'SSIM_Original',
                   'PSNR_1', 'PSNR_2', 'PSNR_Original']
    CSV_Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

    np.random.seed(args.seed)
    sys.stdout = Logger(filename=Result_Path + '/log.log')

    print(args)
    print()
    print(config)
    print()

    Shared_Autoencoder = SHARED_DHDN.SharedDHDN(
        k_value=config['Shared']['K_Value'],
        channels=config['Shared']['Channels'],
        outer_sum=args.outer_sum
    )

    if args.data_parallel:
        Shared_Autoencoder = nn.DataParallel(Shared_Autoencoder, device_ids=[0, 1]).cuda()
    else:
        Shared_Autoencoder = Shared_Autoencoder.to(device_0)

    if args.load_shared:
        state_dict_shared = torch.load(dir_current + model_shared_path, map_location=device_0)
        Shared_Autoencoder.load_state_dict(state_dict_shared)

    # Noise Dataset
    path_validation_noisy = dir_current + config['Locations']['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations']['Validation_GT']

    # Todo: Make function that returns these datasets.
    SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy,
                                             mat_gt_file=path_validation_gt,
                                             device=device_0)

    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                            batch_size=config['Training']['Validation_Batch_Size'],
                                            shuffle=False)

    print('Architecture 1:', args.arc_1)
    print('Architecture 2:', args.arc_2)
    results_1 = get_eval_accuracy(shared=Shared_Autoencoder,
                                  sample_arc=args.arc_1,
                                  dataloader_sidd_validation=dataloader_sidd_validation,
                                  samples=samples,
                                  device=device_0)
    results_2 = get_eval_accuracy(shared=Shared_Autoencoder,
                                  sample_arc=args.arc_2,
                                  dataloader_sidd_validation=dataloader_sidd_validation,
                                  samples=samples,
                                  device=device_0)

    Display_Loss = ("Loss_1: %.6f" % results_1['Loss'] + "\tLoss_2: %.6f" % results_2['Loss'] +
                    "\tLoss_Original: %.6f" % results_1['Loss_Original'])
    Display_SSIM = ("SSIM_1: %.6f" % results_1['SSIM'] + "\tSSIM_2: %.6f" % results_2['SSIM'] +
                    "\tSSIM_Original: %.6f" % results_1['SSIM_Original'])
    Display_PSNR = ("PSNR_1: %.6f" % results_1['PSNR'] + "\tPSNR_2: %.6f" % results_2['PSNR'] +
                    "\tSSIM_Original: %.6f" % results_1['PSNR_Original'])
    print('\n' + '-' * 120)
    print("Validation Data for Architectures: ")
    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR + '\n')
    print('-' * 120 + '\n')

    CSV_Logger.writerow({'Shared_Network': args.model_shared_path,
                         'Architecture_1': args.arc_1,
                         'Architecture_2': args.arc_2,
                         'Loss_1': results_1['Loss'],
                         'Loss_2': results_2['Loss'],
                         'Loss_Original': results_1['Loss_Original'],
                         'SSIM_1': results_1['SSIM'],
                         'SSIM_2': results_2['SSIM'],
                         'SSIM_Original': results_1['SSIM_Original'],
                         'PSNR_1': results_1['PSNR'],
                         'PSNR_2': results_2['PSNR'],
                         'PSNR_Original': results_1['PSNR_Original']})

    CSV_Logger.close()


if __name__ == "__main__":
    main()
