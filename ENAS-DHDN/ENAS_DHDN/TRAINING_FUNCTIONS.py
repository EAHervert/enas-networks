import torch
import torch.nn as nn
import numpy as np

from utilities.utils import AverageMeter  # Helps with keeping track of performance
from utilities.functions import SSIM, PSNR, random_architecture_generation


def evaluate_model(
        epoch,
        use_random,
        controller,
        shared,
        dataloader_sidd_validation,
        config,
        arc_bools=None,
        n_samples=10,
        device=None):
    """Print the validation and test accuracy for a Controller and Shared.

    Args:
        epoch: Current epoch.
        use_random: Whether to use a controller or randomly select architectures.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        dataloader_sidd_validation: Validation dataset.
        config: config for the hyperparameters.
        arc_bools: Booleans for architecture selection
        n_samples: Number of architectures to test when looking for the best one.
        ...

    Returns: Nothing.
    """

    if arc_bools is None:
        arc_bools = [True, True, True]

    print('Here are ' + str(n_samples) + ' architectures:')
    results = get_best_arc(
        use_random=use_random,
        controller=controller,
        shared=shared,
        dataloader_sidd_validation=dataloader_sidd_validation,
        config=config,
        arc_bools=arc_bools,
        n_samples=10,
        verbose=True,
        device=device
    )

    print('Best Architecture:')
    print(results['Best_Arc'])
    display = 'Epoch ' + str(epoch) + ': Eval' + \
              '\nAccuracy=%.6f' % results['Best_Accuracy'] + \
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


def get_best_arc(
        use_random,
        controller,
        shared,
        dataloader_sidd_validation,
        config,
        arc_bools=None,
        n_samples=10,
        verbose=False,
        device=None
):
    """Evaluate several architectures and return the best performing one.

    Args:
        use_random: Whether to use a controller or randomly select architectures.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        dataloader_sidd_validation: Validation dataset.
        config: config for the hyperparameters.
        arc_bools: Booleans for architecture selection
        n_samples: Number of architectures to test when looking for the best one.
        verbose: If True, display the architecture and resulting validation accuracy.

    Returns:
        best_arc: The best performing architecture.
        best_val_acc: Accuracy achieved on the best performing architecture.

    All architectures are evaluated on the same minibatch from the validation set.
    """
    if arc_bools is None:
        arc_bools = [True, True, True]

    if not use_random:
        controller.eval()
    shared.eval()

    arcs = []
    val_loss = []
    val_loss_orig = []
    val_ssim = []
    val_ssim_orig = []
    val_accs = []
    val_psnr = []
    val_psnr_orig = []

    # We loop through the number of samples generating architectures.
    # From these architectures we find the best ones.
    for i in range(n_samples):
        # Generate an architecture.
        if not use_random:
            with torch.no_grad():
                controller()  # perform forward pass to generate a new architecture.
            architecture = controller.sample_arc
        else:
            architecture = random_architecture_generation(k_value=config['Shared']['K_Value'],
                                                          kernel_bool=arc_bools[0],
                                                          down_bool=arc_bools[1],
                                                          up_bool=arc_bools[2])
        arcs.append(architecture)

        results = get_eval_accuracy(shared=shared, sample_arc=architecture,
                                    dataloader_sidd_validation=dataloader_sidd_validation, device=device)

        # We use the SSIM to get our accuracy.
        accuracy = (results['SSIM'] - results['SSIM_Original']) / (1 - results['SSIM_Original'])
        val_loss.append(results['Loss'])
        val_loss_orig.append(results['Loss_Original'])
        val_accs.append(accuracy)
        val_ssim.append(results['SSIM'])
        val_ssim_orig.append(results['SSIM_Original'])
        val_psnr.append(results['PSNR'])
        val_psnr_orig.append(results['PSNR_Original'])
        if verbose:
            # print_arc(architecture)
            print(architecture)
            display = 'Accuracy=%+.6f' % accuracy + \
                      '\nLoss=%+.6f' % results['Loss'] + \
                      '\tLoss_Original=%+.6f' % results['Loss_Original'] + \
                      '\nSSIM=%+.6f' % results['SSIM'] + \
                      '\tSSIM_Original=%+.6f' % results['SSIM_Original'] + \
                      '\nPSNR=%.6f' % results['PSNR'] + \
                      '\tPSNR_Original=%.6f' % results['PSNR_Original'] + '\n' + '-' * 120
            print(display)

    best_iter = np.argmax(val_accs)
    best_arc = arcs[best_iter]
    best_val_acc = val_accs[best_iter]
    best_val_loss = val_loss[best_iter]
    best_val_loss_orig = val_loss_orig[best_iter]
    best_val_ssim = val_ssim[best_iter]
    best_val_ssim_orig = val_ssim_orig[best_iter]
    best_val_psnr = val_psnr[best_iter]
    best_val_psnr_orig = val_psnr_orig[best_iter]

    dict_metrics = {'Best_Arc': best_arc, 'Best_Accuracy': best_val_acc, 'Loss': best_val_loss,
                    'Loss_Original': best_val_loss_orig, 'SSIM': best_val_ssim, 'SSIM_Original': best_val_ssim_orig,
                    'PSNR': best_val_psnr, 'PSNR_Original': best_val_psnr_orig}

    return dict_metrics


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

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x_v = validation_batch['NOISY']
        t_v = validation_batch['GT']

        with torch.no_grad():
            y_v = shared(x_v.to(device), sample_arc)
            Loss_Meter.update(loss(y_v, t_v.to(device)).item())
            Loss_Meter_Original.update(loss(x_v, t_v).item())
            SSIM_Meter.update(SSIM(y_v, t_v.to(device)).item())
            SSIM_Meter_Original.update(SSIM(x_v, t_v).item())
            PSNR_Meter.update(PSNR(mse(y_v, t_v.to(device))).item())
            PSNR_Meter_Original.update(PSNR(mse(x_v, t_v)).item())

    dict_metrics = {'Loss': Loss_Meter.avg, 'Loss_Original': Loss_Meter_Original.avg, 'SSIM': SSIM_Meter.avg,
                    'SSIM_Original': SSIM_Meter_Original.avg, 'PSNR': PSNR_Meter.avg,
                    'PSNR_Original': PSNR_Meter_Original.avg}

    return dict_metrics


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
