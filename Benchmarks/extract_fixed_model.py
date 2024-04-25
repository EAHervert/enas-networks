import os
from ENAS_DHDN import SHARED_DHDN as DHDN
import datetime
import torch
import argparse
from utilities.functions import SSIM, generate_loggers, drop_weights, clip_weights, display_time, list_of_ints
from utilities.functions import shared_weights_to_fixed_weights

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

# Parser
parser = argparse.ArgumentParser(
    prog='Extract_Fixed_Model',
    description='Takes a shared model and extracts fixed model given a fixed architecture.',
)
parser.add_argument('--name', default='Default', type=str)  # Name to save Models
parser.add_argument('--encoder', default=[0, 0, 0, 0, 0, 0, 0, 0, 0], type=list_of_ints)  # Encoder of the DHDN
parser.add_argument('--bottleneck', default=[0, 0], type=list_of_ints)  # Bottleneck of the Encoder
parser.add_argument('--decoder', default=[0, 0, 0, 0, 0, 0, 0, 0, 0], type=list_of_ints)  # Decoder of the DHDN
parser.add_argument('--size', default=3, type=int)  # Size of the DHDN
parser.add_argument('--device', default='cuda:0', type=str)  # GPU to use
parser.add_argument('--shared_model_path_dhdn', default='dhdn_SIDD.pth', type=str)  # Model path dhdn
args = parser.parse_args()


def main():
    global args

    # Hyperparameters
    dir_current = os.getcwd()

    print('Parser Arguments:')
    print(args)

    # Define the devices:
    device = torch.device(args.device)

    # Load the models:
    dhdn_architecture = args.encoder + args.bottleneck + args.decoder
    print('Architecture Being Extracted:')
    print(dhdn_architecture)

    dhdn_shared = DHDN.SharedDHDN(k_value=args.size)
    dhdn_fixed = DHDN.SharedDHDN(k_value=args.size, architecture=dhdn_architecture)
    dhdn_shared.to(device)
    dhdn_fixed.to(device)

    # Load shared weights:
    state_dict_dhdn_shared = torch.load(dir_current + '/' + args.shared_model_path_dhdn, map_location=device)
    dhdn_shared.load_state_dict(state_dict_dhdn_shared)

    # Extract Model Dict
    dict_Fixed_Autoencoder = shared_weights_to_fixed_weights(dhdn_shared, dhdn_fixed, dhdn_architecture)

    # Save fixed model
    model_path_0 = dir_current + '/models/{date}_dhdn_{name}.pth'.format(date=d1, name=args.name)
    torch.save(dict_Fixed_Autoencoder, model_path_0)


if __name__ == "__main__":
    main()
