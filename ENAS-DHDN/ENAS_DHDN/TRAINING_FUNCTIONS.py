import torch
import torch.nn as nn
import numpy as np

from utilities.utils import AverageMeter  # Helps with keeping track of performance
from utilities.functions import SSIM, PSNR, random_architecture_generation


def train_loop(epoch,
               controller,
               shared,
               shared_optimizer,
               config,
               dataloader_sidd_training,
               fixed_arc,
               arc_bools=None,
               passes=1,
               device=None,
               pre_train=False):
    """Trains the shared network based on outputs of controller (if passed).

    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        shared_optimizer: Optimizer for the Shared Network.
        dataloader_sidd_training: Training dataset.
        config: config for the hyperparameters.
        fixed_arc: Architecture to train, overrides the controller sample.
        arc_bools: Booleans for architecture selection
        passes: Number of passes through the training data.
        device: The GPU that we will use.
        pre_train: If we are pre-training or doing standard training.
        ...

    Returns: Nothing.
    """

    # Keep track of the accuracy and loss through the process.
    if arc_bools is None:
        arc_bools = [True, True, True]

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
            # Pick an architecture to work with from the Graph Network (Shared)
            if fixed_arc is None:
                if controller is None:
                    architecture = random_architecture_generation(k_value=config['Shared']['K_Value'],
                                                                  kernel_bool=arc_bools[0],
                                                                  down_bool=arc_bools[1],
                                                                  up_bool=arc_bools[2])
                else:
                    with torch.no_grad():
                        controller()  # perform forward pass to generate a new architecture.
                    architecture = controller.sample_arc
            else:
                architecture = fixed_arc

            x = sample_batch['NOISY']
            y = shared(x, architecture)  # Net is the output of the network
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
            nn.utils.clip_grad_norm_(shared.parameters(), config['Shared']['Child_Grad_Bound'])
            shared_optimizer.step()

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


def evaluate_model(
        epoch,
        controller,
        shared,
        dataloader_sidd_validation,
        config,
        arc_bools=None,
        n_samples=10,
        sample_size=-1,
        device=None):
    """Print the validation and test accuracy for a Controller and Shared.

    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        dataloader_sidd_validation: Validation dataset.
        config: config for the hyperparameters.
        arc_bools: Booleans for architecture selection
        n_samples: Number of architectures to test when looking for the best one.
        sample_size: Number of the validation samples we will use for evaluation, -1 for all samples.
        device: The GPU that we will use.
        ...

    Returns: Nothing.
    """

    if arc_bools is None:
        arc_bools = [True, True, True]

    print('Here are ' + str(n_samples) + ' architectures:')
    results = get_best_arc(
        controller=controller,
        shared=shared,
        dataloader_sidd_validation=dataloader_sidd_validation,
        config=config,
        arc_bools=arc_bools,
        n_samples=10,
        sample_size=sample_size,
        verbose=True,
        device=device
    )

    print('\n' + '-' * 120)
    print('Best Architecture:', results['Best_Arc'])
    accuracy = results['Best_Accuracy']
    if sample_size > 0:
        # Get the resulting accuracy from evaluating on all the validation data
        results = get_eval_accuracy(shared=shared,
                                    sample_arc=results['Best_Arc'],
                                    dataloader_sidd_validation=dataloader_sidd_validation,
                                    samples=None,
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


def get_best_arc(
        controller,
        shared,
        dataloader_sidd_validation,
        config,
        arc_bools=None,
        n_samples=10,
        sample_size=-1,
        verbose=False,
        device=None
):
    """Evaluate several architectures and return the best performing one.

    Args:
        controller: Controller module that generates architectures to be trained.
        shared: Network that contains all possible architectures, with shared weights.
        dataloader_sidd_validation: Validation dataset.
        config: config for the hyperparameters.
        arc_bools: Booleans for architecture selection
        n_samples: Number of architectures to test when looking for the best one.
        sample_size: Number of the validation samples we will use for evaluation, -1 for all samples.
        verbose: If True, display the architecture and resulting validation accuracy.
        device: The GPU that we will use.

    Returns:
        best_arc: The best performing architecture.
        best_val_acc: Accuracy achieved on the best performing architecture.

    All architectures are evaluated on the same minibatch from the validation set.
    """
    if arc_bools is None:
        arc_bools = [True, True, True]

    if controller is not None:
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
        if controller is None:
            architecture = random_architecture_generation(k_value=config['Shared']['K_Value'],
                                                          kernel_bool=arc_bools[0],
                                                          down_bool=arc_bools[1],
                                                          up_bool=arc_bools[2])
        else:
            with torch.no_grad():
                controller()  # perform forward pass to generate a new architecture.
            architecture = controller.sample_arc
        arcs.append(architecture)

        if sample_size > 0:
            samples = np.random.choice(80, sample_size, replace=False)
        else:
            samples = None
        results = get_eval_accuracy(shared=shared,
                                    sample_arc=architecture,
                                    dataloader_sidd_validation=dataloader_sidd_validation,
                                    samples=samples,
                                    device=device)

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
        samples=None,
        device=None
):
    """Evaluate a given architecture.

    Args:
        shared: Network that contains all possible architectures, with shared weights.
        sample_arc: The architecture to use for the evaluation.
        dataloader_sidd_validation: Validation dataset.
        samples: Samples to use from dataloader, or None to use all samples.
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

    # Loop through the entire training set if not to use samples
    if samples is None:
        samples = list(range(80))

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        if i_validation in samples:
            x_v = validation_batch['NOISY']
            t_v = validation_batch['GT']

            with torch.no_grad():
                y_v = shared(x_v, sample_arc)
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


