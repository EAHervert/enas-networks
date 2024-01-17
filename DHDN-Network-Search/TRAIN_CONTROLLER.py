import os
import sys
from utilities import dataset
from ENAS_DHDN import SHARED_DHDN
from ENAS_DHDN import CONTROLLER
import datetime
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import visdom
import random

from utilities.functions import SSIM, display_time
from utilities.utils import CSVLogger, Logger
from ENAS_DHDN.TRAINING_FUNCTIONS import evaluate_model, AverageMeter

# To supress warnings:
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='ENAS_SEARCH_DHDN')

parser.add_argument('--output_file', default='Controller_DHDN', type=str)
parser.add_argument('--number', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sample_size', type=int, default=-1)  # How many samples from validation to evaluate
parser.add_argument('--validation_samples', type=int, default=5)  # How many samples from validation to evaluate
parser.add_argument('--controller_num_aggregate', type=int, default=8)  # Steps in same samples
parser.add_argument('--controller_train_steps', type=int, default=35)  # Total different sample sets
parser.add_argument('--controller_lr', type=float, default=5e-4)  # Total different sample sets
parser.add_argument('--load_shared', default=False, type=lambda x: (str(x).lower() == 'true'))  # Load shared model(s)
parser.add_argument('--model_shared_path', default='2023_12_15__16_25_17/shared_network_parameters.pth', type=str)
parser.add_argument('--load_controller', default=False, type=lambda x: (str(x).lower() == 'true'))  # Load controller
parser.add_argument('--model_controller_path', default='2023_12_15__16_25_17/controller_parameters.pth', type=str)
parser.add_argument('--device', default='cuda:0', type=str)  # GPU to use
# Put shared network on two devices instead of one
parser.add_argument('--data_parallel', default=True, type=lambda x: (str(x).lower() == 'true'))
# To do outer sums for models
parser.add_argument('--outer_sum', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--kernel_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--down_bool', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--up_bool', default=True, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()


# Now, let us run all these pieces and have out program train the controller.
def main():
    global args

    current_time = datetime.datetime.now()
    d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')

    dir_current = os.getcwd()  # Hyperparameters
    config_path = dir_current + '/configs/config_controller.json'
    config = json.load(open(config_path))
    model_controller_path = '/models/' + args.model_controller_path
    model_shared_path = '/models/' + args.model_shared_path

    Model_Path = 'models/' + d1
    if not os.path.isdir(Model_Path):
        os.mkdir(Model_Path)

    device_0 = torch.device(args.device)  # Define the devices
    samples = None if args.sample_size == -1 else args.sample_size
    # Create the CSV Logger:
    Result_Path = 'results/' + args.output_file + '/' + d1
    if not os.path.isdir(Result_Path):
        os.mkdir(Result_Path)

    # Create the CSV Logger:
    File_Name = Result_Path + '/data.csv'
    Field_Names = ['Loss', 'Loss_Original', 'SSIM', 'SSIM_Original', 'PSNR', 'PSNR_Original']
    CSV_Logger = CSVLogger(fieldnames=Field_Names, filename=File_Name)

    File_Name_Ctrl = Result_Path + '/controller.log'
    Field_Names_Ctrl = ['Controller_Reward', 'Controller_Accuracy', 'Controller_Loss']
    Ctrl_Logger = CSVLogger(fieldnames=Field_Names_Ctrl, filename=File_Name_Ctrl)

    # This window will show the SSIM and PSNR of the different networks.
    vis = visdom.Visdom(
        server='eng-ml-01.utdallas.edu',
        port=8097,
        use_incoming_socket=False
    )

    vis.env = args.output_file
    vis_window = {
        'SN_Loss_{d1}'.format(d1=d1): None, 'SN_SSIM_{d1}'.format(d1=d1): None,
        'SN_PSNR_{d1}'.format(d1=d1): None, 'Ctrl_Loss_{d1}'.format(d1=d1): None,
        'Ctrl_Accuracy_{d1}'.format(d1=d1): None, 'Ctrl_Reward_{d1}'.format(d1=d1): None
    }

    t_init = time.time()
    np.random.seed(args.seed)
    sys.stdout = Logger(filename=Result_Path + '/log.log')

    print(args)
    print()
    print(config)
    print()

    Shared_Autoencoder = SHARED_DHDN.SharedDHDN(
        k_value=config['Shared']['K_Value'],
        channels=config['Shared']['Channels'],
        outer_sum=args.outer_sum
    )

    if args.data_parallel:
        Shared_Autoencoder = nn.DataParallel(Shared_Autoencoder, device_ids=[0, 1]).cuda()
    else:
        Shared_Autoencoder = Shared_Autoencoder.to(device_0)

    if args.load_shared:
        state_dict_shared = torch.load(dir_current + model_shared_path, map_location=device_0)
        Shared_Autoencoder.load_state_dict(state_dict_shared)

    Controller = CONTROLLER.Controller(
        k_value=config['Shared']['K_Value'],
        kernel_bool=args.kernel_bool,
        down_bool=args.down_bool,
        up_bool=args.up_bool,
        lstm_size=config['Controller']['Controller_LSTM_Size'],
        lstm_num_layers=config['Controller']['Controller_LSTM_Num_Layers']
    )
    Controller = Controller.to(device_0)

    if args.load_controller:
        state_dict_controller = torch.load(dir_current + model_controller_path, map_location=device_0)
        Controller.load_state_dict(state_dict_controller)

    # We will use the ADAM optimizer for the controller.
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
    Controller_Optimizer = torch.optim.Adam(params=Controller.parameters(),
                                            lr=args.controller_lr,
                                            betas=(0.9, 0.999))

    # Noise Dataset
    path_validation_noisy = dir_current + config['Locations']['Validation_Noisy']
    path_validation_gt = dir_current + config['Locations']['Validation_GT']

    # Todo: Make function that returns these datasets.
    SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=path_validation_noisy,
                                             mat_gt_file=path_validation_gt)

    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                            batch_size=config['Training']['Validation_Batch_Size'],
                                            shuffle=False,
                                            num_workers=8)
    Shared_Autoencoder.eval()

    # Here we will have the following meters for the metrics
    reward_meter = AverageMeter()  # Reward R
    baseline_meter = AverageMeter()  # Baseline b, which controls variance
    val_acc_meter = AverageMeter()  # Validation Accuracy
    loss_meter = AverageMeter()  # Loss
    SSIM_Meter = AverageMeter()
    SSIM_Original_Meter = AverageMeter()

    Controller.zero_grad()
    choices = random.sample(range(80), k=args.validation_samples)
    baseline = None
    for epoch in range(args.epochs):
        Controller.train()  # Train Controller
        for i in range(
                args.controller_train_steps * args.controller_num_aggregate):
            # Randomly selects "validation_samples" batches to run the validation for each controller_num_aggregate
            if i % args.controller_num_aggregate == 0:
                choices = random.sample(range(80), k=args.validation_samples)
            Controller()  # perform forward pass to generate a new architecture
            architecture = Controller.sample_arc

            for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
                if i_validation in choices:
                    x_v = validation_batch['NOISY']
                    t_v = validation_batch['GT']

                    with torch.no_grad():
                        y_v = Shared_Autoencoder(x_v.to(device_0), architecture)
                        SSIM_Meter.update(SSIM(y_v, t_v.to(device_0)).item())
                        SSIM_Original_Meter.update(SSIM(x_v, t_v).item())

            # Use the Normalized SSIM improvement as the accuracy
            normalized_accuracy = (SSIM_Meter.avg - SSIM_Original_Meter.avg) / (1 - SSIM_Original_Meter.avg)

            # make sure that gradients aren't backpropped through the reward or baseline
            reward = normalized_accuracy
            reward += config['Controller']['Controller_Entropy_Weight'] * Controller.sample_entropy.item()
            if baseline is None:
                baseline = normalized_accuracy
            else:
                baseline -= (1 - config['Controller']['Controller_Bl_Dec']) * (baseline - reward)

            loss = - Controller.sample_log_prob * (reward - baseline)

            reward_meter.update(reward)
            baseline_meter.update(baseline)
            val_acc_meter.update(normalized_accuracy)
            loss_meter.update(loss.item())

            # Controller Update:
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(Controller.parameters(), config['Controller']['Controller_Grad_Bound'])
            Controller_Optimizer.step()
            Controller.zero_grad()

            # Aggregate gradients for controller_num_aggregate iteration, then update weights
            if (i + 1) % args.controller_num_aggregate == 0:
                display = 'Epoch_Number=' + str(epoch) + '-' + \
                          str(i // args.controller_num_aggregate) + \
                          '\tController_log_probs=%+.6f' % Controller.sample_log_prob.item() + \
                          '\tController_loss=%+.6f' % loss_meter.val + \
                          '\tEntropy=%.6f' % Controller.sample_entropy.item() + \
                          '\tAccuracy (Normalized SSIM)=%.6f' % val_acc_meter.val + \
                          '\tReward=%.6f' % reward_meter.val + \
                          '\tBaseline=%.6f' % baseline_meter.val
                print(display)
                baseline = None

        print('\n' + '-' * 120)
        print("Controller Average Loss: ", loss_meter.avg)
        print("Controller Average Accuracy (Normalized SSIM): ", val_acc_meter.avg)
        print("Controller Average Reward: ", reward_meter.avg)
        print("Controller Learning Rate:", Controller_Optimizer.param_groups[0]['lr'])
        print('\n' + '-' * 120)

        Ctrl_Logger.writerow({'Controller_Reward': reward_meter.avg, 'Controller_Accuracy': val_acc_meter.avg,
                              'Controller_Loss': loss_meter.avg})

        vis_window[list(vis_window)[3]] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([loss_meter.avg]),
            win=vis_window[list(vis_window)[3]],
            opts=dict(title=list(vis_window)[3], xlabel='Epoch', ylabel='Loss'),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[4]] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([val_acc_meter.avg]),
            win=vis_window[list(vis_window)[4]],
            opts=dict(title=list(vis_window)[4], xlabel='Epoch', ylabel='Accuracy'),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[5]] = vis.line(
            X=np.column_stack([epoch]),
            Y=np.column_stack([reward_meter.avg]),
            win=vis_window[list(vis_window)[5]],
            opts=dict(title=list(vis_window)[5], xlabel='Epoch', ylabel='Reward'),
            update='append' if epoch > 0 else None)

        # Controller in eval mode called in evaluate_model
        validation_results = evaluate_model(epoch=epoch,
                                            controller=Controller,
                                            shared=Shared_Autoencoder,
                                            dataloader_sidd_validation=dataloader_sidd_validation,
                                            config=config,
                                            arc_bools=[args.kernel_bool, args.down_bool, args.up_bool],
                                            sample_size=samples,
                                            device=device_0)

        Legend = ['Shared_Val', 'Orig_Val']

        vis_window[list(vis_window)[0]] = vis.line(
            X=np.column_stack([epoch] * 2),
            Y=np.column_stack([validation_results['Validation_Loss'], validation_results['Validation_Loss_Original']]),
            win=vis_window[list(vis_window)[0]],
            opts=dict(title=list(vis_window)[0], xlabel='Epoch', ylabel='Loss', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[1]] = vis.line(
            X=np.column_stack([epoch] * 2),
            Y=np.column_stack([validation_results['Validation_SSIM'], validation_results['Validation_SSIM_Original']]),
            win=vis_window[list(vis_window)[1]],
            opts=dict(title=list(vis_window)[1], xlabel='Epoch', ylabel='SSIM', legend=Legend),
            update='append' if epoch > 0 else None)

        vis_window[list(vis_window)[2]] = vis.line(
            X=np.column_stack([epoch] * 2),
            Y=np.column_stack([validation_results['Validation_PSNR'], validation_results['Validation_PSNR_Original']]),
            win=vis_window[list(vis_window)[2]],
            opts=dict(title=list(vis_window)[2], xlabel='Epoch', ylabel='PSNR', legend=Legend),
            update='append' if epoch > 0 else None)

        CSV_Logger.writerow({'Loss': validation_results['Validation_Loss'],
                             'Loss_Original': validation_results['Validation_Loss_Original'],
                             'SSIM': validation_results['Validation_SSIM'],
                             'SSIM_Original': validation_results['Validation_SSIM_Original'],
                             'PSNR': validation_results['Validation_PSNR'],
                             'PSNR_Original': validation_results['Validation_PSNR_Original']})
        architectures = []
        for i in range(args.number):
            with torch.no_grad():
                Controller()
            architectures.append(Controller.sample_arc)
        architectures_array = np.array(architectures)
        dict_arc = {}
        for i in range(len(architectures_array[-1])):
            dict_arc[i] = {}
            if not (i + 1) % 3:
                for j in range(3):
                    dict_arc[i][j] = np.count_nonzero(architectures_array[:, i] == j) / args.number
            else:
                for j in range(8):
                    dict_arc[i][j] = np.count_nonzero(architectures_array[:, i] == j) / args.number
        argmax_arc = []
        print('\n' + '-' * 120)
        print('Controller Distribution:')
        for key in dict_arc.keys():
            print(key, dict_arc[key])
            argmax_arc.append(np.argmax(list(dict_arc[key].values())))
        print('Architecture argmax:', argmax_arc)
        print('\n' + '-' * 120)

    CSV_Logger.close()
    Ctrl_Logger.close()

    t_final = time.time()
    display_time(t_final - t_init)

    Controller_Path = Model_Path + '/pre_trained_controller_parameters.pth'
    torch.save(Controller.state_dict(), Controller_Path)


if __name__ == "__main__":
    main()
