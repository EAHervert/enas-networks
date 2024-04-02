import os
import datetime
import json
from scipy.io import loadmat
import torch
import argparse

from utilities.functions import SSIM, PSNR

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_dhdn.json'
config = json.load(open(config_path))

parser = argparse.ArgumentParser(
    prog='Validation_Metrics'.format(date=d1),
    description='Generates Validation metrics file for testing model performance',
)
parser.add_argument('--name', default='tests', type=str)  # Name of the results
parser.add_argument('--path', default='results/denoise_test.mat', type=str)  # path to denoised .mat file
args = parser.parse_args()

# Load benchmark data for processing
mat_gt_file = dir_current + '/data/ValidationGtBlocksSrgb.mat'
mat_noisy_file = dir_current + '/data/ValidationNoisyBlocksSrgb.mat'
mat_denoise_file = dir_current + args.path
mat_gt = loadmat(mat_gt_file)
mat_noisy = loadmat(mat_noisy_file)
mat_denoise = loadmat(mat_noisy_file)

size = mat_gt['ValidationGtBlocksSrgb'].shape

images_gt = mat_gt['ValidationGtBlocksSrgb']
images_noisy = mat_noisy['ValidationNoisyBlocksSrgb']
images_denoise = mat_denoise['ValidationNoisyBlocksSrgb']

ssim_base = 0
psnr_base = 0
ssim_denoise_val = 0
psnr_denoise_val = 0
for i in range(size[0]):
    for j in range(2):
        start_j = int(j * size[1] / 2)
        end_j = int((j + 1) * size[1] / 2)
        images_gt_pt = torch.tensor(images_gt[i, start_j:end_j, :, :, :] / 255.,
                                    dtype=torch.float).permute(0, 3, 1, 2)
        images_noisy_pt = torch.tensor(images_noisy[i, start_j:end_j, :, :, :] / 255.,
                                       dtype=torch.float).permute(0, 3, 1, 2)
        images_denoise_pt = torch.tensor(images_denoise[i, start_j:end_j, :, :, :] / 255.,
                                         dtype=torch.float).permute(0, 3, 1, 2)

        mse = torch.square(images_gt_pt - images_noisy_pt).mean()
        mse_denoise = torch.square(images_gt_pt - images_denoise_pt).mean()
        ssim = SSIM(images_gt_pt, images_noisy_pt)
        ssim_denoise = SSIM(images_gt_pt, images_denoise_pt)
        psnr = PSNR(mse)
        psnr_denoise = PSNR(mse_denoise)
        print('Image', i, '-', j, '-- Denoised:', ssim_denoise.item(), '-', psnr_denoise.item(),
              '\tBase:', ssim.item(), '-', psnr.item())
        ssim_base += ssim.item()
        psnr_base += psnr.item()
        ssim_denoise_val += ssim_denoise.item()
        psnr_denoise_val += psnr_denoise.item()
    print()

ssim_base /= size[0] * 2
psnr_base /= size[0] * 2
ssim_denoise_val /= size[0] * 2
psnr_denoise_val /= size[0] * 2

print('TOTAL -- Denoised:', ssim_denoise_val, '-', psnr_denoise_val, '\tBase:', ssim_base, '-', psnr_base)
