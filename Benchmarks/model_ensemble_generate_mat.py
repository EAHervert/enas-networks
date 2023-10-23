import os
import copy
import datetime
import json
from scipy.io import loadmat, savemat
import numpy as np
import torch
from ENAS_DHDN import SHARED_DHDN as DHDN
import srgb_conc_unet
from utilities.functions import transform_tensor, get_out
import utilities.dataset as dataset
from torch.utils.data import DataLoader
import argparse

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_dhdn.json'
config = json.load(open(config_path))

parser = argparse.ArgumentParser(
    prog='Model_Ensemble_Mat_Generation_'.format(date=d1),
    description='Generates .mat file for testing model performance',
)
parser.add_argument('--name', default='test', type=str)  # Name of the folder to save
parser.add_argument('--type', default='validation', type=str)  # Name of the folder to save
parser.add_argument('--architecture', default='DHDN', type=str)  # DHDN or EDHDN
parser.add_argument('--model_file_0', default='models/model_0.pth', type=str)  # Model path to weights
parser.add_argument('--model_file_1', default='models/model_1.pth', type=str)  # Model path to weights
args = parser.parse_args()

# Load benchmark data for processing
if args.type == 'validation':
    mat_file = os.getcwd() + config['Locations']['Validation_Noisy']
else:
    mat_file = os.getcwd() + config['Locations']['Benchmark']
mat = loadmat(mat_file)

# Get the model paths
model_dhdn_0 = os.getcwd() + '/' + args.model_file_0
model_dhdn_1 = os.getcwd() + '/' + args.model_file_1
# Cast to relevant device
device_0 = torch.device(config['CUDA']['Device0'])
device_1 = torch.device(config['CUDA']['Device1'])

# Model architectures and parameters
if args.architecture == 'DHDN':
    architecture = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dhdn_0 = DHDN.SharedDHDN(architecture=architecture)
    dhdn_1 = DHDN.SharedDHDN(architecture=architecture)

    # Cast to relevant device
    dhdn_0.to(device_0)
    dhdn_1.to(device_1)
    state_dict_dhdn_0 = torch.load(model_dhdn_0, map_location=device_0)
    state_dict_dhdn_1 = torch.load(model_dhdn_1, map_location=device_1)

elif args.architecture == 'EDHDN':
    architecture = [0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dhdn_0 = DHDN.SharedDHDN(architecture=architecture)
    dhdn_1 = DHDN.SharedDHDN(architecture=architecture)

    # Cast to relevant device
    dhdn_0.to(device_0)
    dhdn_1.to(device_1)
    state_dict_dhdn_0 = torch.load(model_dhdn_0, map_location=device_0)
    state_dict_dhdn_1 = torch.load(model_dhdn_1, map_location=device_1)

elif args.architecture == 'DHDN_Color':
    # Model architectures and parameters
    dhdn_0 = srgb_conc_unet.Net2()
    dhdn_1 = srgb_conc_unet.Net2()
    state_dict_dhdn_0 = dhdn_0.state_dict()  # state dict for dhdn in repo
    state_dict_dhdn_1 = dhdn_1.state_dict()  # state dict for dhdn in repo

    # Cast to relevant device
    dhdn_0.to(device_0)
    dhdn_1.to(device_1)
    state_dict_dhdn_0_weights = torch.load(model_dhdn_0, map_location=device_0)['model'].state_dict()  # weights
    state_dict_dhdn_1_weights = torch.load(model_dhdn_1, map_location=device_0)['model'].state_dict()  # weights

    # Transfer weights correctly
    for i in state_dict_dhdn_0_weights.keys():
        key = i[7:]
        state_dict_dhdn_0[key] = state_dict_dhdn_0_weights[i].clone()
        state_dict_dhdn_1[key] = state_dict_dhdn_1_weights[i].clone()

else:
    print('Invalid Architecture!')
    exit()

dhdn_0.load_state_dict(state_dict_dhdn_0)
dhdn_1.load_state_dict(state_dict_dhdn_1)

SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=mat_file, mat_gt_file=None)
dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=config['Training']['Validation_Batch_Size'],
                                        shuffle=False, num_workers=8)

y_dhdn_final, y_dhdn_final_plus = [], []
transforms = [[1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]]
out_temp, out_temp_plus = [], []
for i_batch, sample_batch in enumerate(dataloader_sidd_validation):
    x_sample_pt = sample_batch['NOISY']
    with torch.no_grad():
        y_dhdn_0 = dhdn_0(x_sample_pt.to(device_0))
        y_dhdn_0_plus = y_dhdn_0.detach().clone()
        for transform in transforms:
            y_dhdn_0_plus += transform_tensor(dhdn_0(transform_tensor(x_sample_pt,
                                                                      r=transform[0], s=transform[1]).to(device_0)),
                                              r=4 - transform[0], s=transform[1])
        y_dhdn_0_plus /= 8

        y_dhdn_1 = dhdn_1(x_sample_pt.to(device_1))
        y_dhdn_1_plus = y_dhdn_1.detach().clone()
        for transform in transforms:
            y_dhdn_1_plus += transform_tensor(dhdn_1(transform_tensor(x_sample_pt,
                                                                      r=transform[0], s=transform[1]).to(device_1)),
                                              r=4 - transform[0], s=transform[1])
        y_dhdn_1_plus /= 8

        y_dhdn = (y_dhdn_0.cpu() + y_dhdn_1.cpu()) / 2
        y_dhdn_plus = (y_dhdn_0_plus.cpu() + y_dhdn_1_plus.cpu()) / 2

    i_image, i_split = i_batch // 2, i_batch % 2
    print('Batch {i}-{j} processed.'.format(i=i_image, j=i_split))

    y_dhdn_out = get_out(y_dhdn)
    y_dhdn_out_plus = get_out(y_dhdn_plus)

    if len(out_temp) == 0:
        out_temp.append(y_dhdn_out)
        out_temp_plus.append(y_dhdn_out_plus)
    else:
        out_temp.append(y_dhdn_out)
        out_temp_plus.append(y_dhdn_out_plus)
        y_dhdn_final.append(np.concatenate(out_temp))
        y_dhdn_final_plus.append(np.concatenate(out_temp_plus))
        out_temp = []
        out_temp_plus = []

    del x_sample_pt, y_dhdn_0, y_dhdn_0_plus, y_dhdn_1, y_dhdn_1_plus, y_dhdn, y_dhdn_plus

if not os.path.exists(dir_current + '/results/model-ensemble/{type}/single/{name}/'.format(name=args.name,
                                                                                           type=args.type)):
    os.makedirs(dir_current + '/results/model-ensemble/{type}/single/{name}/'.format(name=args.name, type=args.type))
if not os.path.exists(dir_current + '/results/model-ensemble/{type}/self-ensemble/{name}/'.format(name=args.name,
                                                                                                  type=args.type)):
    os.makedirs(dir_current + '/results/model-ensemble/{type}/self-ensemble/{name}/'.format(name=args.name,
                                                                                            type=args.type))

y_dhdn_final = np.array(y_dhdn_final, dtype=np.uint8)
file_dhdn = 'results/model-ensemble/{type}/single/{name}/SubmitSrgb.mat'.format(name=args.name, type=args.type)
mat_dhdn = copy.deepcopy(mat)
if args.type == 'validation':
    mat_dhdn['ValidationNoisyBlocksSrgb'] = y_dhdn_final
else:
    mat_dhdn['BenchmarkNoisyBlocksSrgb'] = y_dhdn_final
savemat(file_dhdn, mat_dhdn)

y_dhdn_final_plus = np.array(y_dhdn_final_plus, dtype=np.uint8)
file_dhdn_plus = 'results/model-ensemble/{type}/self-ensemble/{name}/SubmitSrgb.mat'.format(name=args.name,
                                                                                            type=args.type)
mat_dhdn_plus = copy.deepcopy(mat)
if args.type == 'validation':
    mat_dhdn_plus['ValidationNoisyBlocksSrgb'] = y_dhdn_final_plus
else:
    mat_dhdn_plus['BenchmarkNoisyBlocksSrgb'] = y_dhdn_final_plus
savemat(file_dhdn_plus, mat_dhdn_plus)
