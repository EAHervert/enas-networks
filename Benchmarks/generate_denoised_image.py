import os
import datetime
import json
import numpy as np
import torch
import pandas as pd
import time
from ENAS_DHDN import SHARED_DHDN as DHDN
from utilities.functions import image_np_to_tensor, tensor_to_np_image, SSIM
from utilities.functions import True_PSNR as PSNR

import srgb_conc_unet
import cv2
import argparse

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_dhdn.json'
config = json.load(open(config_path))


# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))


parser = argparse.ArgumentParser(
    prog='Image_Generation_'.format(date=d1),
    description='Generates image for testing model performance',
)
parser.add_argument('--name', default='tests', type=str)  # Name of the folder to save
parser.add_argument('--image_path', default='sample_images/sample_1_ns.png', type=str)  # path to image
parser.add_argument('--crop_size', default=256, type=int)  # path to image
parser.add_argument('--device', default='cuda:0', type=str)  # Which device to use
parser.add_argument('--architecture', default='DHDN', type=str)  # DHDN, EDHDN, or DHDN_Color
parser.add_argument('--encoder', default=[0, 0, 0, 0, 0, 0, 0, 0, 0], type=list_of_ints)  # Encoder of the DHDN
parser.add_argument('--bottleneck', default=[0, 0], type=list_of_ints)  # Bottleneck of the Encoder
parser.add_argument('--decoder', default=[0, 0, 0, 0, 0, 0, 0, 0, 0], type=list_of_ints)  # Decoder of the DHDN
parser.add_argument('--channels', default=128, type=int)  # Channel Number for DHDN
parser.add_argument('--k_value', default=3, type=int)  # Size of encoder and decoder
parser.add_argument('--model_file', default='models/model.pth', type=str)  # Model path to weights
args = parser.parse_args()

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

dhdn.load_state_dict(state_dict_dhdn)  # Load weights to the model

# Load the images as np array (make values float)
noisy = np.array(cv2.imread(dir_current + '/' + args.image_path, cv2.IMREAD_COLOR), dtype=np.single)[:, :, ::-1]
image_path_gt = args.image_path[:-6] + 'gt.png'
gt = np.array(cv2.imread(dir_current + '/' + image_path_gt, cv2.IMREAD_COLOR), dtype=np.single)[:, :, ::-1]

# Image Size
np_size = noisy.shape
height, width = noisy.shape[0:2]
print('Image Numpy Size:', np_size)

# Transformed to Tensor
noisy_pt = image_np_to_tensor(noisy, crop_size=args.crop_size).detach()
gt_pt = image_np_to_tensor(gt, crop_size=args.crop_size).detach()
denoised_pt = torch.zeros_like(noisy_pt).detach()

# Tensor Size
tensor_size = noisy_pt.size()
n, m = tensor_size[0:2]
print('Image Tensor Size:', tensor_size)

# Metrics dictionary that will hold the SSIM and PSNR values comparing against ground truth
metrics = []
t_final = 0
for i in range(n):
    t_init = time.time()
    for j in range(m):
        gt_i_j = gt_pt[i, j, :, :, :].unsqueeze(0)

        # Process one image crop
        t_before = time.time()
        image_i_j = noisy_pt[i, j, :, :, :].to(device_0).detach().unsqueeze(0)
        with torch.no_grad():
            denoised_i_j = dhdn(image_i_j).clone()
            t_after = time.time()

            denoised_pt[i, j, :, :, :] = denoised_i_j  # Save the denoised sample in the output tensor

        print('Batch {i} - Image {j} processed.'.format(i=i, j=j),
              ' \tTime to process batch: {:.2f}s'.format(t_after - t_before))
        t_final += t_after - t_before

        row = {'index_i': i, 'index_j': j, 'time_to_process': t_after - t_init,
               'ssim_denoised': SSIM(denoised_i_j, gt_i_j), 'ssim_original': SSIM(image_i_j, gt_i_j),
               'psnr_denoised': PSNR(denoised_i_j, gt_i_j), 'psnr_original': PSNR(image_i_j, gt_i_j)}

        metrics.append(row)

print('Time to process Whole Image: {:.2f}s'.format(t_final))

# transform back to np array
denoised = tensor_to_np_image(denoised_pt)

# Save the denoised image
output_file = args.image_path[:-6] + 'dn_' + args.name + '.png'
cv2.imwrite(dir_current + '/' + output_file, denoised[:, :, ::-1])

# Save the csv file with the metrics
csv_file = args.image_path[:-6] + args.name + '.csv'
metrics = pd.DataFrame(metrics)
metrics.to_csv(csv_file, index=False)
