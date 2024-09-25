import os
import datetime
import json
import numpy as np
import torch
import csv
import time
import math
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
parser.add_argument('--architecture_type', default='DHDN', type=str)  # DHDN, EDHDN, or DHDN_Color
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
architecture = None

# Model architectures and parameters
if args.architecture_type == 'DHDN':
    architecture = args.encoder + args.bottleneck + args.decoder
    dhdn = DHDN.SharedDHDN(architecture=architecture, channels=args.channels, k_value=args.k_value).to(
        device_0)  # Cast to relevant device
    state_dict_dhdn = torch.load(model_dhdn, map_location=device_0)

elif args.architecture_type == 'Shared_DHDN':
    architecture = args.encoder + args.bottleneck + args.decoder
    dhdn = DHDN.SharedDHDN(channels=args.channels, k_value=args.k_value).to(device_0)  # Cast to relevant device
    state_dict_dhdn = torch.load(model_dhdn, map_location=device_0)

elif args.architecture_type == 'DHDN_Color':
    # Model architectures and parameters
    dhdn = srgb_conc_unet.Net2().to(device_0)
    state_dict_dhdn = dhdn.state_dict()  # state dict for dhdn in repo  # Cast to relevant device
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
noisy = np.array(cv2.cvtColor(cv2.imread(dir_current + '/' + args.image_path, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR),
                 dtype=np.single)
image_path_gt = args.image_path[:-6] + 'gt.png'
gt = np.array(cv2.cvtColor(cv2.imread(dir_current + '/' + image_path_gt, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR),
              dtype=np.single)

# Image Size
np_size = noisy.shape
height, width = noisy.shape[0:2]
i_range = math.ceil(height / args.crop_size)
j_range = math.ceil(width / args.crop_size)
print('Image Numpy Size:', np_size)

# Transformed to Tensor
noisy_pt = image_np_to_tensor(noisy, i_range=i_range, j_range=j_range, crop_size=args.crop_size).detach()
gt_pt = image_np_to_tensor(gt, i_range=i_range, j_range=j_range, crop_size=args.crop_size).detach()
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
        # Process one image crop
        gt_i_j = gt_pt[i, j, :, :, :].to(device_0).detach().unsqueeze(0)
        t_before = time.time()
        image_i_j = noisy_pt[i, j, :, :, :].to(device_0).detach().unsqueeze(0)
        with torch.no_grad():
            if args.architecture_type == 'Shared_DHDN':
                denoised_i_j = dhdn(image_i_j, architecture=architecture)
            else:
                denoised_i_j = dhdn(image_i_j)
            t_after = time.time()

            denoised_pt[i, j, :, :, :] = denoised_i_j.squeeze(0)  # Save the denoised sample in the output tensor

        print('Sample {i:02} - {j:02} processed.'.format(i=i, j=j),
              ' Time to process batch: {:.6f}s'.format(t_after - t_before),
              '\nssim_denoised: {:.6f}'.format(SSIM(denoised_i_j, gt_i_j).item()),
              '\tssim_original: {:.6f}'.format(SSIM(image_i_j, gt_i_j).item()),
              '\tpsnr_denoised: {:.6f}'.format(PSNR(denoised_i_j, gt_i_j).item()),
              '\tpsnr_original: {:.6f}'.format(PSNR(image_i_j, gt_i_j).item()))
        t_final += t_after - t_before

        row = {'index_i': i, 'index_j': j, 'time_to_process': t_after - t_init,
               'ssim_denoised': SSIM(denoised_i_j, gt_i_j).item(), 'ssim_original': SSIM(image_i_j, gt_i_j).item(),
               'psnr_denoised': PSNR(denoised_i_j, gt_i_j).item(), 'psnr_original': PSNR(image_i_j, gt_i_j).item()}

        metrics.append(row)
        del image_i_j, gt_i_j, denoised_i_j

print('Time to process whole image: {:.6f}s'.format(t_final))

# transform back to np array
denoised = tensor_to_np_image(denoised_pt, crop_size=args.crop_size)

# Save the denoised image
output_file = args.image_path[:-6] + 'dn_' + args.name + '.png'
cv2.imwrite(dir_current + '/' + output_file, denoised[:, :, ::-1])

# Save the csv file with the metrics
csv_file = args.image_path[:-6] + args.name + '.csv'
keys = metrics[0].keys()

with open(csv_file, 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(metrics)
