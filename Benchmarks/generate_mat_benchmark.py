import os
import copy
import datetime
import json
from scipy.io import loadmat, savemat
import numpy as np
import torch
from ENAS_DHDN import SHARED_DHDN as DHDN
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
    prog='Validation_Mat'.format(date=d1),
    description='Generates Validation .mat file for testing model performance',
)
parser.add_argument('--name', default='test', type=str)  # Name of the folder to save
parser.add_argument('--device', default='cuda:0', type=str)  # Which device to use to generate .mat file
parser.add_argument('--architecture', default='DHDN', type=str)  # DHDN or EDHDN
parser.add_argument('--model_file', default='model.pth', type=str)  # Model path to weights
args = parser.parse_args()

# Load benchmark data for processing
mat_file = os.getcwd() + config['Locations']['Benchmark']
mat = loadmat(mat_file)

# Get the model paths
model_dhdn = os.getcwd() + '/models/' + args.model_file
# Cast to relevant device
device0 = torch.device(args.device)

# Model architectures and parameters
if args.architecture == 'DHDN':
    architecture = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif args.architecture == 'EDHDN':
    architecture = [0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
else:
    print('Invalid Architecture!')
    exit()
dhdn = DHDN.SharedDHDN(architecture=architecture)
dhdn.to(device0)

dhdn.load_state_dict(torch.load(model_dhdn, map_location=device0))

SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=mat_file, mat_gt_file=None)
dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=config['Training']['Validation_Batch_Size'],
                                        shuffle=False, num_workers=8)

y_dhdn_final = []
y_dhdn_final_plus = []
transforms = [[1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]]
for i_batch, sample_batch in enumerate(dataloader_sidd_validation):
    x_sample_pt = sample_batch['NOISY']
    with torch.no_grad():
        y_dhdn = dhdn(x_sample_pt.to(device0))

        y_dhdn_plus = y_dhdn.detach().clone()
        for transform in transforms:
            y_dhdn_plus += transform_tensor(dhdn(transform_tensor(x_sample_pt,
                                                                  r=transform[0], s=transform[1]).to(device0)),
                                            r=4 - transform[0], s=transform[1])
        y_dhdn_plus /= 8

        print('Batch {i} processed.'.format(i=i_batch))

    y_dhdn_out = get_out(y_dhdn)
    y_dhdn_out_plus = get_out(y_dhdn_plus)

    y_dhdn_final.append(y_dhdn_out)
    y_dhdn_final_plus.append(y_dhdn_out_plus)

    del x_sample_pt
    del y_dhdn

if not os.path.exists(dir_current + '/results/single/{name}/'.format(name=args.name)):
    os.makedirs(dir_current + '/results/single/{name}/'.format(name=args.name))
if not os.path.exists(dir_current + '/results/self-ensemble/{name}/'.format(name=args.name)):
    os.makedirs(dir_current + '/results/self-ensemble/{name}/'.format(name=args.name))

y_dhdn_final = np.array(y_dhdn_final, dtype=np.uint8)
file_dhdn = 'results/single/{name}/SubmitSrgb.mat'.format(name=args.name)
mat_dhdn = copy.deepcopy(mat)
mat_dhdn['BenchmarkNoisyBlocksSrgb'] = y_dhdn_final
savemat(file_dhdn, mat_dhdn)

y_dhdn_final_plus = np.array(y_dhdn_final_plus, dtype=np.uint8)
file_dhdn_plus = 'results/self-ensemble/{name}/SubmitSrgb.mat'.format(name=args.name)
mat_dhdn_plus = copy.deepcopy(mat)
mat_dhdn_plus['BenchmarkNoisyBlocksSrgb'] = y_dhdn_final_plus
savemat(file_dhdn_plus, mat_dhdn_plus)
