import os
import datetime
import json
import numpy as np
import torch
from ENAS_DHDN import SHARED_DHDN as DHDN
from utilities.functions import image_np_to_tensor, tensor_to_np_image
import utilities.models as models
import cv2
import argparse

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')
dir_current = os.getcwd()
config_path = dir_current + '/configs/config.json'
config = json.load(open(config_path))


# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))


parser = argparse.ArgumentParser(
    prog='Image_Generation_Compare'.format(date=d1),
    description='Generates image for testing different models',
)
parser.add_argument('--name', default='test', type=str)  # Name of the folder to save
parser.add_argument('--image_path', default='sample_images/sample_1_ns.png', type=str)  # path to image
parser.add_argument('--device', default='cuda:0', type=str)  # Which device to use
parser.add_argument('--model', default='RDUNet', type=str)  # Model path to weights
parser.add_argument('--model_file', default='models/model.pth', type=str)  # Model path to weights
args = parser.parse_args()

# Get the model paths
model_weights = dir_current + '/' + args.model_file
device_0 = torch.device(args.device)

# Model architectures and parameters
if args.model == 'RDUNet':
    model = models.RDUNet(**config['Model'])

else:
    print('Invalid Architecture!')
    exit()

# Cast to relevant device
model.to(device_0)
state_dict_model = torch.load(model_weights, map_location=device_0)
model.load_state_dict(state_dict_model)

noisy = np.array(cv2.imread(dir_current + '/' + args.image_path, cv2.IMREAD_COLOR), dtype=np.single)[:, :, ::-1]
image_path_gt = args.image_path[:-6] + 'gt.png'
gt = np.array(cv2.imread(dir_current + '/' + image_path_gt, cv2.IMREAD_COLOR), dtype=np.single)[:, :, ::-1]

# Transformed to Tensor
noisy_pt = image_np_to_tensor(noisy)
gt_pt = image_np_to_tensor(gt)
denoised_pt = torch.zeros_like(noisy_pt)

for i in range(noisy_pt.size()[0]):
    with torch.no_grad():
        denoised_pt[i, :, :, :, :] = model(noisy_pt[i, :, :, :, :].to(device_0)).clone()

    print('Batch {i} processed.'.format(i=i))

# transform back to np array
denoised = tensor_to_np_image(denoised_pt)

output_file = args.image_path[:-6] + 'dn_' + args.name + '.png'
cv2.imwrite(dir_current + '/' + output_file, denoised[:, :, ::-1])
