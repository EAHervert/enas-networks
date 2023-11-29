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
        ...

    Returns: Nothing.
    """

    controller.eval()
    shared.eval()

    print('Here are ' + str(n_samples) + ' architectures:')
    best_arc, valid_SSIM, valid_SSIM_Original, valid_PSNR, valid_PSNR_Original = get_best_arc(
        controller=controller,
        shared=shared,
        dataloader_sidd_validation=dataloader_sidd_validation,
        n_samples=10,
        verbose=True,
        device=device
    )

    print('Epoch ' + str(epoch) + ': Eval')
    print('Best_Architecture:', best_arc)
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

    arcs = []
    val_accs = []
    val_orig = []
    val_psnr = []
    psnr_ori = []

    # We loop through the number of samples generating architectures.
    # From these architectures we find the best ones.
    for i in range(n_samples):

        # Generate an architecture.
        with torch.no_grad():
            controller()  # perform forward pass to generate a new architecture.
        architecture = controller.sample_arc
        arcs.append(architecture)

        valid_SSIM, valid_SSIM_Original, valid_PSNR, valid_PSNR_Original = get_eval_accuracy(
            shared=shared,
            sample_arc=architecture,
            dataloader_sidd_validation=dataloader_sidd_validation,
            device=device
        )
        # We use the SSIM to get our accuracy.
        val_accs.append(valid_SSIM)
        val_orig.append(valid_SSIM_Original)
        val_psnr.append(valid_PSNR)
        psnr_ori.append(valid_PSNR_Original)
        if verbose:
            print_arc(architecture)
            print('SSIM=' + str(valid_SSIM))
            print('SSIM_Original=' + str(valid_SSIM_Original))
            print('PSNR=' + str(valid_PSNR))
            print('PSNR_Original=' + str(valid_PSNR_Original))
            print('-' * 120)

    best_iter = np.argmax(val_accs)
    best_arc = arcs[best_iter]
    best_val_acc = val_accs[best_iter]
    best_val_orig = val_orig[best_iter]
    best_val_psnr = val_psnr[best_iter]
    best_psnr_ori = psnr_ori[best_iter]

    controller.train()
    shared.train()

    return best_arc, best_val_acc, best_val_orig, best_val_psnr, best_psnr_ori


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
    PSNR_Meter_Original = AverageMeter()

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation, start=1):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']

        with torch.no_grad():
            y_v = shared(x_v.to(device), sample_arc)
            SSIM_Meter.update(SSIM(y_v, t_v.to(device)).item())
            SSIM_Meter_Original.update(SSIM(x_v, t_v).item())
            PSNR_Meter.update(PSNR(mse(y_v, t_v.to(device))).item())
            PSNR_Meter_Original.update(PSNR(mse(x_v, t_v)).item())

    return SSIM_Meter.avg, SSIM_Meter_Original.avg, PSNR_Meter.avg, PSNR_Meter_Original.avg


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
