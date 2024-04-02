import os
import datetime
import json
import pandas as pd
from scipy.io import loadmat
import torch
import argparse

import utilities.dataset as dataset
from torch.utils.data import DataLoader

from utilities.functions import SSIM, PSNR

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_dhdn.json'
config = json.load(open(config_path))

parser = argparse.ArgumentParser(
    prog='Metrics'.format(date=d1),
    description='Generates metrics file for model performance',
)
parser.add_argument('--name', default='tests', type=str)  # Name of the results
parser.add_argument('--type', default='validation', type=str)  # Name of the folder to save
parser.add_argument('--dataset', default='SIDD', type=str)  # Name of the folder to save
parser.add_argument('--path', default='data/ValidationNoisyBlocksSrgb.mat', type=str)  # path to denoised .mat file
parser.add_argument('--device', default='cuda:0', type=str)  # Which device to use to generate .mat file
args = parser.parse_args()

# Load data for processing
if args.type == 'validation':
    mat_gt_file = dir_current + config['Locations'][args.dataset]['Validation_GT']
    mat_noisy_file = dir_current + config['Locations'][args.dataset]['Validation_Noisy']
else:
    mat_gt_file = dir_current + config['Locations'][args.dataset]['Testing_GT']
    mat_noisy_file = dir_current + config['Locations'][args.dataset]['Testing']

mat_denoise_file = dir_current + '/' + args.path
mat_gt = loadmat(mat_gt_file)
mat_noisy = loadmat(mat_noisy_file)
mat_denoise = loadmat(mat_denoise_file)

if args.dataset == 'SIDD':
    size = mat_gt['ValidationGtBlocksSrgb'].shape
    images_gt = mat_gt['ValidationGtBlocksSrgb']
    images_noisy = mat_noisy['ValidationNoisyBlocksSrgb']
    images_denoise = mat_denoise['ValidationNoisyBlocksSrgb']
else:
    if args.type == 'validation':
        size = mat_gt['val_gt'].shape
        images_gt = mat_gt['val_gt']
        images_noisy = mat_noisy['val_ng']
        images_denoise = mat_denoise['val_ng']
    else:
        size = mat_gt['test_gt'].shape
        images_gt = mat_gt['test_gt']
        images_noisy = mat_noisy['test_ng']
        images_denoise = mat_denoise['test_ng']

device = torch.device(args.device)

validation_dataset = dataset.DatasetMAT(mat_noisy_file=mat_noisy_file, mat_gt_file=mat_gt_file, device=device)
denoised_dataset = dataset.DatasetMAT(mat_noisy_file=mat_denoise_file, mat_gt_file=None, device=device)
if args.dataset == 'SIDD':
    dataloader_validation = DataLoader(dataset=validation_dataset,
                                       batch_size=config['Training']['Validation_Batch_Size'] * 2,
                                       shuffle=False)
    dataloader_denoised = DataLoader(dataset=denoised_dataset,
                                     batch_size=config['Training']['Validation_Batch_Size'] * 2,
                                     shuffle=False)

else:
    dataloader_denoised = DataLoader(dataset=denoised_dataset,
                                     batch_size=config['Training']['Validation_Batch_Size_DIV2K'] * 2,
                                     shuffle=False)
    dataloader_validation = DataLoader(dataset=validation_dataset,
                                       batch_size=config['Training']['Validation_Batch_Size_DIV2K'] * 2,
                                       shuffle=False)

dict_out = []
ssim_base = 0
psnr_base = 0
ssim_denoise_val = 0
psnr_denoise_val = 0
count = 0
with torch.no_grad():
    for i_batch, (sample_batch_val, sample_batch_den) in enumerate(zip(dataloader_validation, dataloader_denoised)):
        images_gt_pt = sample_batch_val['GT']
        images_noisy_pt = sample_batch_val['NOISY']
        images_denoise_pt = sample_batch_den['NOISY']

        mse = torch.square(images_gt_pt - images_noisy_pt).mean()
        mse_denoise = torch.square(images_gt_pt - images_denoise_pt).mean()
        ssim = SSIM(images_gt_pt, images_noisy_pt)
        ssim_denoise = SSIM(images_gt_pt, images_denoise_pt)
        psnr = PSNR(mse)
        psnr_denoise = PSNR(mse_denoise)
        dict_item = {'Image': i_batch, 'Denoised_PSNR': round(ssim_denoise.item(), 6),
                     'Denoised_SSIM': round(psnr_denoise.item(), 6),
                     'Base_PSNR': round(ssim.item(), 6), 'Base_SSIM': round(psnr.item(), 6)}
        print(dict_item)
        dict_out.append(dict_item)
        ssim_base += ssim.item()
        psnr_base += psnr.item()
        ssim_denoise_val += ssim_denoise.item()
        psnr_denoise_val += psnr_denoise.item()

        del images_gt_pt, images_noisy_pt, images_denoise_pt
        count += 1

ssim_base /= count
psnr_base /= count
ssim_denoise_val /= count
psnr_denoise_val /= count

dict_item = {'Image': 'ALL', 'Denoised_PSNR': round(ssim_denoise_val, 6), 'Denoised_SSIM': round(psnr_denoise_val, 6),
             'Base_PSNR': round(ssim_base, 6), 'Base_SSIM': round(psnr_base, 6)}

print(dict_item)
dict_out.append(dict_item)

dataframe_out = pd.DataFrame(dict_out)
dataframe_out.to_csv('csv_out/' + args.name + '.csv', index=False)
