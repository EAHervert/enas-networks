import os
import sys
from ENAS_DHDN import CONTROLLER
import datetime
import json
import torch
import argparse

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Generate Architectures')

# Training:
parser.add_argument('--number', type=int, default=30)
parser.add_argument('--device', default='cuda:0', type=str)  # Which device to use to generate .mat file
parser.add_argument('--Kernel_Bool', type=bool, default=True)
parser.add_argument('--Down_Bool', type=bool, default=True)
parser.add_argument('--Up_Bool', type=bool, default=True)
parser.add_argument('--model_controller_path', default='2023_12_15__16_25_17/controller_parameters.pth',
                    type=str)

args = parser.parse_args()

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

# Hyperparameters
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_search.json'
config = json.load(open(config_path))
if not os.path.exists(dir_current + '/models/'):
    os.makedirs(dir_current + '/models/')

# Define the devices:
device_0 = torch.device(args.device)

Controller = CONTROLLER.Controller(
    k_value=config['Shared']['K_Value'],
    kernel_bool=args.Kernel_Bool,
    down_bool=args.Down_Bool,
    up_bool=args.Up_Bool,
    LSTM_size=config['Controller']['Controller_LSTM_Size'],
    LSTM_num_layers=config['Controller']['Controller_LSTM_Num_Layers']
)

Controller = Controller.to(device_0)
state_dict_controller = torch.load(dir_current + '/models/' + args.model_controller_path, map_location=device_0)
Controller.load_state_dict(state_dict_controller)

Controller.eval()
for i in range(args.number):
    with torch.no_grad():
        Controller()
    print(Controller.sample_arc)
