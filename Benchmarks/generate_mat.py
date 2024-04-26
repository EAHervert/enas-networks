import os
import copy
import datetime
import json
import pandas as pd
from scipy.io import loadmat, savemat
import numpy as np
import torch
from ENAS_DHDN import SHARED_DHDN as DHDN
import srgb_conc_unet
import utilities.dataset as dataset
from torch.utils.data import DataLoader
from utilities.functions import list_of_ints, get_out, SSIM
from utilities.functions import True_PSNR as PSNR
import argparse

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

parser = argparse.ArgumentParser(
    prog='Mat_Generation_'.format(date=d1),
    description='Generates .mat file for testing model performance',
)
parser.add_argument('--name', default='tests', type=str)  # Name of the folder to save
parser.add_argument('--type', default='validation', type=str)  # Name of the folder to save
parser.add_argument('--dataset', default='SIDD', type=str)  # Name of the folder to save
parser.add_argument('--device', default='cuda:0', type=str)  # Which device to use to generate .mat file
parser.add_argument('--architecture', default='DHDN', type=str)  # DHDN, EDHDN, or DHDN_Color
parser.add_argument('--encoder', default=[0, 0, 0, 0, 0, 0, 0, 0, 0], type=list_of_ints)  # Encoder of the DHDN
parser.add_argument('--bottleneck', default=[0, 0], type=list_of_ints)  # Bottleneck of the Encoder
parser.add_argument('--decoder', default=[0, 0, 0, 0, 0, 0, 0, 0, 0], type=list_of_ints)  # Decoder of the DHDN
parser.add_argument('--channels', default=128, type=int)  # Channel Number for DHDN
parser.add_argument('--k_value', default=3, type=int)  # Size of encoder and decoder
parser.add_argument('--model_file', default='models/model.pth', type=str)  # Model path to weights
parser.add_argument('--generate_metrics', default=False, type=lambda s: (str(s).lower() == 'true'))  # Generate Metrics
parser.add_argument('--generate_mat', default=True, type=lambda s: (str(s).lower() == 'true'))  # Generate Metrics
# To use self-ensembling
args = parser.parse_args()


def main():
    global args

    dir_current = os.getcwd()
    config_path = dir_current + '/configs/config_dhdn.json'
    config = json.load(open(config_path))

    # Load benchmark data for processing
    if args.type == 'validation':
        mat_gt_file = dir_current + config['Locations'][args.dataset]['Validation_GT']
        mat_noisy_file = dir_current + config['Locations'][args.dataset]['Validation_Noisy']
    else:
        if args.dataset == 'SIDD':
            mat_gt_file = None
            mat_noisy_file = dir_current + config['Locations'][args.dataset]['Benchmark']
        else:
            mat_gt_file = dir_current + config['Locations'][args.dataset]['Testing_GT']
            mat_noisy_file = dir_current + config['Locations'][args.dataset]['Testing_Noisy']

    mat_noisy = loadmat(mat_noisy_file)

    # Get the model paths
    model_dhdn = dir_current + '/' + args.model_file
    device_0 = torch.device(args.device)

    # Model architectures and parameters
    if args.architecture == 'DHDN':
        architecture = args.encoder + args.bottleneck + args.decoder
        dhdn = DHDN.SharedDHDN(architecture=architecture, channels=args.channels, k_value=args.k_value)

        # Cast to relevant device
        dhdn.to(device_0)
        state_dict_dhdn = torch.load(model_dhdn, map_location=device_0)

    elif args.architecture == 'EDHDN':
        architecture = [0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        dhdn = DHDN.SharedDHDN(architecture=architecture)

        # Cast to relevant device
        dhdn.to(device_0)
        state_dict_dhdn = torch.load(model_dhdn, map_location=device_0)

    elif args.architecture == 'DHDN_Color':
        # Model architectures and parameters
        dhdn = srgb_conc_unet.Net2()
        state_dict_dhdn = dhdn.state_dict()  # state dict for dhdn in repo

        # Cast to relevant device
        dhdn.to(device_0)
        state_dict_dhdn_weights = torch.load(model_dhdn, map_location=device_0)['model'].state_dict()  # weights

        # Transfer weights correctly
        for i in state_dict_dhdn_weights.keys():
            key = i[7:]
            state_dict_dhdn[key] = state_dict_dhdn_weights[i].clone()

    else:
        print('Invalid Architecture!')
        exit()

    dhdn.load_state_dict(state_dict_dhdn)

    SIDD_validation = dataset.DatasetMAT(mat_noisy_file=mat_noisy_file, mat_gt_file=mat_gt_file, device=device_0)
    if args.dataset == 'SIDD':
        dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                                batch_size=config['Training']['Testing_Batch_Size'],
                                                shuffle=False)
    else:
        dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                                batch_size=config['Training']['Testing_Batch_Size_DIV2K'],
                                                shuffle=False)

    y_dhdn_final = []
    dict_out = []
    ssim_base = 0
    psnr_base = 0
    ssim_denoise_val = 0
    psnr_denoise_val = 0
    count = 0
    for i_batch, sample_batch in enumerate(dataloader_sidd_validation):
        x_noisy_pt = sample_batch['NOISY']

        with torch.no_grad():
            y_out_pt = dhdn(x_noisy_pt.to(device_0))

            print('Batch {i} processed.'.format(i=i_batch))
            y_dhdn_final.append(get_out(y_out_pt))

        if mat_gt_file is not None and args.generate_metrics:
            x_gt_pt = sample_batch['GT']

            ssim_denoise = SSIM(x_gt_pt, y_out_pt)
            psnr_denoise = PSNR(x_gt_pt, y_out_pt)

            ssim = SSIM(x_gt_pt, x_noisy_pt)
            psnr = PSNR(x_gt_pt, x_noisy_pt)

            dict_item = {'Image': i_batch, 'Denoised_PSNR': round(ssim_denoise.item(), 6),
                         'Denoised_SSIM': round(psnr_denoise.item(), 6),
                         'Base_PSNR': round(ssim.item(), 6), 'Base_SSIM': round(psnr.item(), 6)}
            print(dict_item)
            dict_out.append(dict_item)

            ssim_base += ssim.item()
            psnr_base += psnr.item()
            ssim_denoise_val += ssim_denoise.item()
            psnr_denoise_val += psnr_denoise.item()
            count += 1
            del x_gt_pt

        del x_noisy_pt, y_out_pt

    if args.generate_metrics:
        ssim_base /= count
        psnr_base /= count
        ssim_denoise_val /= count
        psnr_denoise_val /= count

        dict_item = {'Image': 'ALL', 'Denoised_PSNR': round(ssim_denoise_val, 6),
                     'Denoised_SSIM': round(psnr_denoise_val, 6),
                     'Base_PSNR': round(ssim_base, 6), 'Base_SSIM': round(psnr_base, 6)}

        print(dict_item)
        dict_out.append(dict_item)

        dataframe_out = pd.DataFrame(dict_out)
        dataframe_out.to_csv('csv_out/' + args.name + '.csv', index=False)

    if args.generate_mat:
        if not os.path.exists(
                dir_current + '/results/{type}/{name}/'.format(name=args.name, type=args.type)):
            os.makedirs(dir_current + '/results/{type}/{name}/'.format(name=args.name, type=args.type))

        if dataset == 'SIDD':
            validation_tag = 'ValidationNoisyBlocksSrgb'
            benchmark_tag = 'BenchmarkNoisyBlocksSrgb'
        else:
            validation_tag = 'val_ng'
            benchmark_tag = 'test_ng'

        y_dhdn_final = np.array(y_dhdn_final, dtype=np.uint8)
        file_dhdn = 'results/{type}/{name}/SubmitSrgb.mat'.format(name=args.name, type=args.type)
        mat_dhdn = copy.deepcopy(mat_noisy)
        if args.type == 'validation':
            mat_dhdn[validation_tag] = y_dhdn_final
        else:
            mat_dhdn[benchmark_tag] = y_dhdn_final
        savemat(file_dhdn, mat_dhdn)


if __name__ == "__main__":
    main()
