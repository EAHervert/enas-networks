import os
from scipy.io import loadmat
import torch

from utilities.functions import SSIM, PSNR

# Load benchmark data for processing
mat_gt_file = os.getcwd() + '/data/ValidationGtBlocksSrgb.mat'
mat_noisy_file = os.getcwd() + '/data/ValidationNoisyBlocksSrgb.mat'
mat_gt = loadmat(mat_gt_file)
mat_noisy = loadmat(mat_noisy_file)

size = mat_gt['ValidationGtBlocksSrgb'].shape

images_gt = mat_gt['ValidationGtBlocksSrgb']
images_noisy = mat_noisy['ValidationNoisyBlocksSrgb']

total_ssim = 0
total_psnr = 0
for i in range(size[0]):
    for j in range(2):
        start_j = int(j * size[1] / 2)
        end_j = int((j + 1) * size[1] / 2)
        images_gt_pt = torch.tensor(images_gt[i, start_j:end_j, :, :, :] / 255.,
                                    dtype=torch.float).permute(0, 3, 1, 2)
        images_noisy_pt = torch.tensor(images_noisy[i, start_j:end_j, :, :, :] / 255.,
                                       dtype=torch.float).permute(0, 3, 1, 2)

        mse = torch.square(images_gt_pt - images_noisy_pt).mean()
        ssim = SSIM(images_gt_pt, images_noisy_pt)
        psnr = PSNR(mse)
        print('Image', i, '-', j, ':', ssim.item(), '-', psnr.item())
        total_ssim += ssim.item()
        total_psnr += psnr.item()
    print()

total_ssim /= size[0] * 2
total_psnr /= size[0] * 2

print('TOTAL:', total_ssim, '-', total_psnr)
