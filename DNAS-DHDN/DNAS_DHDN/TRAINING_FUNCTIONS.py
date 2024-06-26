import torch
import torch.nn as nn

from utilities.utils import AverageMeter  # Helps with keeping track of performance
from utilities.functions import SSIM, PSNR, random_architecture_generation


def train_loop(epoch: int,
               weights,
               shared,
               shared_optimizer,
               config,
               dataloader_sidd_training,
               passes=1,
               device=None,
               pre_train=False,
               verbose=True):
    """Trains the shared network based on outputs of controller (if passed).

    Args:
        epoch: Current epoch.
        weights: weights to generate alphas.
        shared: Network that contains all possible architectures, with shared weights.
        shared_optimizer: Optimizer for the Shared Network.
        dataloader_sidd_training: Training dataset.
        config: config for the hyperparameters.
        passes: Number of passes through the training data.
        device: The GPU that we will use.
        pre_train: If we are pre-training or doing standard training.
        verbose: If we print training information.
        ...

    Returns: Nothing.
    """

    # Keep track of the accuracy and loss through the process.
    loss_batch = AverageMeter()
    loss_original_batch = AverageMeter()
    ssim_batch = AverageMeter()  # Doubles as the accuracy.
    ssim_original_batch = AverageMeter()
    psnr_batch = AverageMeter()
    psnr_original_batch = AverageMeter()

    loss = nn.L1Loss()
    mse = nn.MSELoss()

    if device is not None:
        loss = loss.to(device)
        mse = mse.to(device)

    # Start the timer for the epoch.
    for pass_ in range(passes):
        for i_batch, sample_batch in enumerate(dataloader_sidd_training):
            x = sample_batch['NOISY']
            y = shared(x, weights)  # Net is the output of the network
            t = sample_batch['GT']

            loss_value = loss(y, t)
            loss_batch.update(loss_value.item())

            # Calculate values not needing to be backpropagated
            with torch.no_grad():
                loss_original_batch.update(loss(x, t).item())

                ssim_batch.update(SSIM(y, t).item())
                ssim_original_batch.update(SSIM(x, t).item())

                psnr_batch.update(PSNR(mse(y, t)).item())
                psnr_original_batch.update(PSNR(mse(x, t)).item())

            # Backpropagate to train model
            shared_optimizer.zero_grad()
            loss_value.backward()
            nn.utils.clip_grad_norm_(shared.parameters(), config['Differential']['Child_Grad_Bound'])
            shared_optimizer.step()

            if verbose:
                if i_batch % 100 == 0:
                    Display_Loss = "Loss_Shared: %.6f" % loss_batch.val + "\tLoss_Original: %.6f" % loss_original_batch.val
                    Display_SSIM = "SSIM_Shared: %.6f" % ssim_batch.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
                    Display_PSNR = "PSNR_Shared: %.6f" % psnr_batch.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

                    if pre_train:
                        print("Pre-Training Data for Epoch: ", epoch, "Pass:", pass_, "Image Batch: ", i_batch)
                    else:
                        print("Training Data for Epoch: ", epoch, "Pass:", pass_, "Image Batch: ", i_batch)
                    print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

            # Free up space in GPU
            del x, y, t

    dict_train = {'Loss': loss_batch.avg, 'Loss_Original': loss_original_batch.avg, 'SSIM': ssim_batch.avg,
                  'SSIM_Original': ssim_original_batch.avg, 'PSNR': psnr_batch.avg,
                  'PSNR_Original': psnr_original_batch.avg}

    return dict_train


def train_alphas_loop(epoch: int,
                      weights,
                      shared,
                      epsilon: float,
                      lr_w_alpha: float,
                      eta,
                      config,
                      dataloader_sidd_training,
                      dataloader_sidd_validation,
                      device=None,
                      verbose=True):
    """Trains the architecture weights of the DNAS-DHDN model.

    Args:
        epoch: Current epoch.
        weights: weights to generate alphas.
        shared: Network that contains all possible architectures, with shared weights.
        epsilon: Parameter for approximating second order derivative.
        lr_w_alpha: Parameter for updating weights.
        eta: Parameter for modifying [w'].
        dataloader_sidd_training: Training dataset.
        dataloader_sidd_validation: Validation dataset.
        config: config for the hyperparameters.
        device: The GPU that we will use.
        verbose: If we print training information.
        ...

    Returns: Nothing.
    """

    # Keep track of the accuracy and loss through the process.
    loss_batch = AverageMeter()
    loss_original_batch = AverageMeter()
    ssim_batch = AverageMeter()  # Doubles as the accuracy.
    ssim_original_batch = AverageMeter()
    psnr_batch = AverageMeter()
    psnr_original_batch = AverageMeter()

    loss = nn.L1Loss()
    mse = nn.MSELoss()

    if device is not None:
        loss = loss.to(device)
        mse = mse.to(device)

    # Start the timer for the epoch.
    for i_batch, sample_batch in enumerate(dataloader_sidd_validation):
        x = sample_batch['NOISY']
        t = sample_batch['GT']
        y = shared(x, weights)  # Net is the output of the network

        if eta > 0:
            second_order = calculate_second_order(weights=weights,
                                                  shared=shared,
                                                  epsilon=epsilon,
                                                  dataloader_sidd_training=dataloader_sidd_training,
                                                  device=device)
        else:
            second_order = None

        loss_value_f = loss(y, t)
        loss_batch.update(loss_value_f.item())

        # Calculate values not needing to be backpropagated
        with torch.no_grad():
            loss_original_batch.update(loss(x, t).item())

            ssim_batch.update(SSIM(y, t).item())
            ssim_original_batch.update(SSIM(x, t).item())

            psnr_batch.update(PSNR(mse(y, t)).item())
            psnr_original_batch.update(PSNR(mse(x, t)).item())

        # Backpropagate to train model
        loss_value_f.backward()
        nn.utils.clip_grad_norm_(weights, config['Differential']['Child_Grad_Bound'])
        weights += lr_w_alpha * weights.grad

        if verbose:
            if i_batch % 100 == 0:
                Display_Loss = "Loss_Shared: %.6f" % loss_batch.val + "\tLoss_Original: %.6f" % loss_original_batch.val
                Display_SSIM = "SSIM_Shared: %.6f" % ssim_batch.val + "\tSSIM_Original: %.6f" % ssim_original_batch.val
                Display_PSNR = "PSNR_Shared: %.6f" % psnr_batch.val + "\tPSNR_Original: %.6f" % psnr_original_batch.val

                print("Training Data for Epoch: ", epoch, "Image Batch: ", i_batch)
                print(Display_Loss + '\n' + Display_SSIM + '\n' + Display_PSNR)

        # Free up space in GPU
        del x, y, t

    dict_train = {'Loss': loss_batch.avg, 'Loss_Original': loss_original_batch.avg, 'SSIM': ssim_batch.avg,
                  'SSIM_Original': ssim_original_batch.avg, 'PSNR': psnr_batch.avg,
                  'PSNR_Original': psnr_original_batch.avg}

    return dict_train


def calculate_w_prime(weights,
                      shared,
                      eta,
                      loss,
                      dataloader_sidd_training):
    weights_prime = weights.clone().detach().requires_grad_(True)
    shared_prime = shared.clone().detach()
    for i_batch, sample_batch in enumerate(dataloader_sidd_training):
        x = sample_batch['NOISY']
        t = sample_batch['GT']
        y = shared_prime(x, weights.clone().detach())  # Net is the output of the network

        loss_params = loss(y, t)
        loss_params.backward()
        grad_params = shared_prime.parameters().grad

        return weights_prime - eta * grad_params


def calculate_w_pm(weights,
                   weights_prime,
                   shared,
                   eta,
                   loss,
                   dataloader_sidd_validation):
    shared_prime = shared.clone().detach()
    for i_batch, sample_batch in enumerate(dataloader_sidd_validation):
        x = sample_batch['NOISY']
        t = sample_batch['GT']
        y = shared_prime(x, weights.clone().detach())  # Net is the output of the network

        loss_params = loss(y, t)
        loss_params.backward()
        grad_params = shared_prime.parameters().grad

        return weights_prime - eta * grad_params


def calculate_second_order(weights,
                           shared,
                           epsilon,
                           loss,
                           dataloader_sidd_training,
                           dataloader_sidd_validation,
                           device=None):
    shared.eval()
    for i_batch, sample_batch in enumerate(dataloader_sidd_training):
        x = sample_batch['NOISY']
        t = sample_batch['GT']
        y = shared(x, weights)  # Net is the output of the network

        loss_params = loss(y, t)
        loss_params.backward()
    grad_params = shared.parameters().grad

    w_plus, w_minus = shared.parameters() + epsilon * grad_params, shared.parameters() - epsilon * grad_params

    weights_prime = weights.clone().detach().requires_grad_(True)

    return 0


def evaluate_model(
        epoch,
        shared,
        dataloader_sidd_validation,
        config,
        n_samples=10,
        sample_size=-1,
        device=None):
    """Print the validation and test accuracy for a Controller and Shared.

    Args:
        epoch: Current epoch.
        shared: Network that contains all possible architectures, with shared weights.
        dataloader_sidd_validation: Validation dataset.
        config: config for the hyperparameters.
        n_samples: Number of architectures to test when looking for the best one.
        sample_size: Number of the validation samples we will use for evaluation, -1 for all samples.
        device: The GPU that we will use.
        ...

    Returns: Nothing.
    """

    print('\n' + '-' * 120)
    results = get_eval_accuracy(shared=shared,
                                dataloader_sidd_validation=dataloader_sidd_validation,
                                device=device)

    accuracy = (results['SSIM'] - results['SSIM_Original']) / (1 - results['SSIM_Original'])
    display = 'Epoch ' + str(epoch) + ': Eval' + \
              '\nAccuracy=%.6f' % accuracy + \
              '\nValidation_Loss=%.6f' % results['Loss'] + \
              '\tValidation_Loss_Original=%.6f' % results['Loss_Original'] + \
              '\nValidation_SSIM=%.6f' % results['SSIM'] + \
              '\tValidation_SSIM_Original=%.6f' % results['SSIM_Original'] + \
              '\nValidation_PSNR=%.6f' % results['PSNR'] + \
              '\tValidation_PSNR_Original=%.6f' % results['PSNR_Original'] + '\n' + '-' * 120
    print(display)

    final_dict = {'Validation_Loss': results['Loss'], 'Validation_Loss_Original': results['Loss_Original'],
                  'Validation_SSIM': results['SSIM'], 'Validation_SSIM_Original': results['SSIM_Original'],
                  'Validation_PSNR': results['PSNR'], 'Validation_PSNR_Original': results['PSNR_Original']}

    return final_dict


def get_eval_accuracy(
        shared,
        dataloader_sidd_validation,
        device=None
):
    """Evaluate a given architecture.

    Args:
        shared: Network that contains all possible architectures, with shared weights.
        dataloader_sidd_validation: Validation dataset.
        device: The GPU that we will use.

    Returns:
        acc: Average accuracy.
    """

    loss = nn.L1Loss()
    mse = nn.MSELoss()

    # Pass to GPU if needed.
    if device is not None:
        loss = loss.to(device)
        mse = mse.to(device)

    # Use Meters to keep track of the averages:
    Loss_Meter = AverageMeter()
    Loss_Meter_Original = AverageMeter()
    SSIM_Meter = AverageMeter()
    SSIM_Meter_Original = AverageMeter()
    PSNR_Meter = AverageMeter()
    PSNR_Meter_Original = AverageMeter()

    # Loop through the entire validation set
    # samples = list(range(80))

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']

        with torch.no_grad():
            y_v = shared(x_v)
            Loss_Meter.update(loss(y_v, t_v).item())
            Loss_Meter_Original.update(loss(x_v, t_v).item())
            SSIM_Meter.update(SSIM(y_v, t_v).item())
            SSIM_Meter_Original.update(SSIM(x_v, t_v).item())
            PSNR_Meter.update(PSNR(mse(y_v, t_v)).item())
            PSNR_Meter_Original.update(PSNR(mse(x_v, t_v)).item())

    dict_metrics = {'Loss': Loss_Meter.avg, 'Loss_Original': Loss_Meter_Original.avg, 'SSIM': SSIM_Meter.avg,
                    'SSIM_Original': SSIM_Meter_Original.avg, 'PSNR': PSNR_Meter.avg,
                    'PSNR_Original': PSNR_Meter_Original.avg}

    return dict_metrics
