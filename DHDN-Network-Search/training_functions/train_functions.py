import torch
import torch.nn as nn
import numpy as np

from utilities.utils import AverageMeter  # Helps with keeping track of performance
from utilities.functions import SSIM, PSNR


def evaluate_model(
        epoch,
        controller,
        shared,
        dataloader_sidd_validation,
        n_samples=10,
        device=None):
    """Print the validation and test accuracy for a Controller and Shared.

    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        dataloader_sidd_validation: Validation dataset.
        n_samples: Number of architectures to test when looking for the best one.

    Returns: Nothing.
    """

    controller.eval()
    shared.eval()

    print('Here are ' + str(n_samples) + ' architectures:')
    best_arc, _ = get_best_arc(
        controller=controller,
        shared=shared,
        dataloader_sidd_validation=dataloader_sidd_validation,
        n_samples=10,
        verbose=True,
        device=device
    )

    valid_SSIM, valid_SSIM_Original, valid_PSNR, valid_PSNR_Original = get_eval_accuracy(
        shared=shared,
        sample_arc=best_arc,
        dataloader_sidd_validation=dataloader_sidd_validation,
        device=device
    )

    print('Epoch ' + str(epoch) + ': Eval')
    print('valid_SSIM: %.6f' % valid_SSIM)
    print('valid_SSIM_Original: %.6f' % valid_SSIM_Original)
    print('valid_PSNR: %.6f' % valid_PSNR)
    print('valid_PSNR_Original: %.6f' % valid_PSNR_Original)

    controller.train()
    shared.train()


def get_best_arc(
        controller,
        shared,
        dataloader_sidd_validation,
        n_samples=10,
        verbose=False,
        device=None
):
    """Evaluate several architectures and return the best performing one.

    Args:
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        dataloader_sidd_validation: Validation dataset.
        n_samples: Number of architectures to test when looking for the best one.
        verbose: If True, display the architecture and resulting validation accuracy.

    Returns:
        best_arc: The best performing architecture.
        best_val_acc: Accuracy achieved on the best performing architecture.

    All architectures are evaluated on the same minibatch from the validation set.
    """

    controller.eval()
    shared.eval()

    # Define the MSE loss for PSNR value
    mse = nn.MSELoss()

    # Pass to GPU if needed.
    if device is not None:
        mse = mse.to(device)

    arcs = []
    val_accs = []

    # We loop through the number of samples generating architectures.
    # From these architectures we find the best ones.
    for i in range(n_samples):

        # Generate an architecture.
        with torch.no_grad():
            controller()  # perform forward pass to generate a new architecture.
        architecture = controller.sample_arc
        arcs.append(architecture)

        SSIM_val = 0
        SSIM_original = 0
        mse_val = 0
        mse_original = 0
        for i_validation, validation_batch in enumerate(dataloader_sidd_validation, start=1):
            x_v = validation_batch['NOISY']
            t_v = validation_batch['GT']

            with torch.no_grad():
                y_v = shared(x_v.to(device), architecture)

            SSIM_val += (SSIM(y_v, t_v.to(device)) + SSIM_val * (i_validation - 1)) / i_validation
            SSIM_original += (SSIM(x_v, t_v) + SSIM_val * (i_validation - 1)) / i_validation
            mse_val += (mse(y_v, t_v.to(device)) + mse_val * (i_validation - 1)) / i_validation
            mse_original += (mse(x_v, t_v) + mse_original * (i_validation - 1)) / i_validation

        # We use the SSIM to get our accuracy.
        val_accs.append(SSIM_val)
        PSNR_val = PSNR(mse_val)
        PSNR_Original = PSNR(mse_original)

        if verbose:
            print_arc(architecture)
            print('SSIM=' + str(SSIM_val))
            print('SSIM_Original=' + str(SSIM_original))
            print('PSNR=' + str(PSNR_val))
            print('PSNR_Original=' + str(PSNR_Original))
            print('-' * 120)

    best_iter = np.argmax(val_accs)
    best_arc = arcs[best_iter]
    best_val_acc = val_accs[best_iter]

    controller.train()
    shared.train()

    return best_arc, best_val_acc


def get_eval_accuracy(
        shared,
        sample_arc,
        dataloader_sidd_validation,
        device=None
):
    """Evaluate a given architecture.

    Args:
        shared: Network that contains all possible architectures, with shared weights.
        dataloader_sidd_validation: Validation dataset.
        sample_arc: The architecture to use for the evaluation.

    Returns:
        acc: Average accuracy.
    """

    mse = nn.MSELoss()

    # Pass to GPU if needed.
    if device is not None:
        mse = mse.to(device)

    # Use Meters to keep track of the averages:
    SSIM_Meter = AverageMeter()
    SSIM_Meter_Original = AverageMeter()
    PSNR_Meter = AverageMeter()
    PSNR_Original_Meter = AverageMeter()

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation, start=1):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']

        with torch.no_grad():
            y_v = shared(x_v.to(device), sample_arc)
            SSIM_Meter.update(SSIM(y_v, t_v.to(device)).item())
            SSIM_Meter_Original.update(SSIM(x_v, t_v).item())
            PSNR_Meter.update(PSNR(mse(y_v, t_v.to(device))).item())
            PSNR_Original_Meter.update(PSNR(mse(x_v, t_v)).item())

    return SSIM_Meter.avg, SSIM_Meter_Original.avg, PSNR_Meter.avg, PSNR_Original_Meter.avg


# This function will display the architectures that were generated by the Controller.
def print_arc(sample_arc):
    """Display a sample architecture in a readable format.

    Args:
        sample_arc: The architecture to display.

    Returns: Nothing.
    """

    # First the Encoder:
    Kernels = ['3, 3, 3', '5, 3, 3', '3, 5, 3', '5, 5, 3', '3, 3, 5', '5, 3, 5', '3, 5, 5', '5, 5, 5']
    Down = ['Max', 'Average', 'Convolution']
    Up = ['Pixel Shuffle', 'Transpose Convolution', 'Bilinear Interpolation']

    Array_Len = len(sample_arc)

    for i in range((Array_Len // 2) - 1):
        if (i + 1) % 3 == 0:
            print(Down[sample_arc[i]])
        else:
            print(Kernels[sample_arc[i]])

    for i in range((Array_Len // 2) - 1, (Array_Len // 2) + 1):
        print(Kernels[sample_arc[i]])

    j = 0
    for i in range((Array_Len // 2) + 1, Array_Len):
        if j % 3 == 0:
            print(Up[sample_arc[i]])
        else:
            print(Kernels[sample_arc[i]])

        j += 1

    print()
