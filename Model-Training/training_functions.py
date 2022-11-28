import torch
from utilities.functions import list_instances, display_time, create_batches, rand_mod, SSIM, PSNR
from utilities.dataloader_sidd_medium import load_dataset_images
from utilities.utils import AverageMeter
import time


def generate_loggers():
    # Image Batches
    loss_batch = AverageMeter()
    loss_original_batch = AverageMeter()
    ssim_batch = AverageMeter()
    ssim_original_batch = AverageMeter()
    psnr_batch = AverageMeter()
    psnr_original_batch = AverageMeter()

    batch_loggers = (loss_batch, loss_original_batch, ssim_batch, ssim_original_batch, psnr_batch, psnr_original_batch)

    # Total Training
    loss_meter_train = AverageMeter()
    loss_original_meter_train = AverageMeter()
    ssim_meter_train = AverageMeter()
    ssim_original_meter_train = AverageMeter()
    psnr_meter_train = AverageMeter()
    psnr_original_meter_train = AverageMeter()

    train_loggers = (loss_meter_train, loss_original_meter_train, ssim_meter_train, ssim_original_meter_train,
                     psnr_meter_train, psnr_original_meter_train)

    # Validation
    loss_meter_val = AverageMeter()
    loss_original_meter_val = AverageMeter()
    ssim_meter_val = AverageMeter()
    ssim_original_meter_val = AverageMeter()
    psnr_meter_val = AverageMeter()
    psnr_original_meter_val = AverageMeter()

    val_loggers = (loss_meter_val, loss_original_meter_val, ssim_meter_val, ssim_original_meter_val, psnr_meter_val,
                   psnr_original_meter_val)

    return batch_loggers, train_loggers, val_loggers


# Loop through the training batches:
def train_loop(epoch, config, batches, model, architecture, device, Loss, MSE, loss_meter, loss_meter_original,
               ssim_meter, ssim_meter_original, psnr_meter, psnr_meter__oiginal, loss_meter_batch,
               loss_meter_original_batch, ssim_meter_batch, ssim_meter_original_batch, psnr_meter_batch,
               psnr_meter__oiginal_batch, backpropagate=False, optimizer=None, mode='Training'):

    index = 0

    for batch in batches:

        t0 = time.time()
        index += 1

        if mode == 'Training':
            loader = load_dataset_images(batch, size_n=config['Training']['Train_N'],
                                         size_m=config['Training']['Train_M'])
        else:
            loader = load_dataset_images(batch, size_crop=256, size_n=config['Training']['Validation_N'],
                                         size_m=config['Training']['Validation_M'])

        # Loop through the batches:
        for j, (inputs, targets) in enumerate(loader):
            # Randomly Flip and rotate the tensors:
            inputs, targets = rand_mod(inputs, targets)

            # Cast to Cuda:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get the Outputs of the network:
            if backpropagate:
                outputs = model(inputs, architecture)
            else:
                with torch.no_grad():
                    outputs = model(inputs, architecture)

            # Calculate the losses:
            loss_train = Loss(outputs, targets)

            with torch.no_grad():
                loss_original = Loss(inputs, targets)

            loss_meter.update(loss_train.item())
            loss_meter_original.update(loss_original.item())

            # Calculate SSIM:
            with torch.no_grad():
                SSIM_train = SSIM(outputs, targets)
                SSIM_original = SSIM(inputs, targets)

            ssim_meter.update(SSIM_train.item())
            ssim_meter_original.update(SSIM_original.item())

            # Calculate PSNR:
            with torch.no_grad():
                MSE_train = MSE(outputs, targets)
                MSE_original = MSE(inputs, targets)

            PSNR_train = PSNR(MSE_train)
            PSNR_original = PSNR(MSE_original)

            psnr_meter.update(PSNR_train.item())
            psnr_meter__oiginal.update(PSNR_original.item())

            if backpropagate:
                # Back-Propagate through the batch:
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            # Update Image Batches logger:
            loss_meter_batch.update(loss_train.item())
            loss_meter_original_batch.update(loss_original.item())
            ssim_meter_batch.update(SSIM_train.item())
            ssim_meter_original_batch.update(SSIM_original.item())
            psnr_meter_batch.update(PSNR_train.item())
            psnr_meter__oiginal_batch.update(PSNR_original.item())

        t1 = time.time()

        if mode == 'Training':
            print("Training Data for Epoch: ", epoch, "Image Batch: ", index)

            Display_Loss = "Loss_DHDN: %.6f" % loss_meter_batch.avg + \
                           "\tLoss_Original: %.6f" % loss_meter_original_batch.avg
            Display_SSIM = "SSIM_DHDN: %.6f" % ssim_meter_batch.avg + \
                           "\tSSIM_Original: %.6f" % ssim_meter_original_batch.avg
            Display_PSNR = "PSNR_DHDN: %.6f" % psnr_meter_batch.avg + \
                           "\tPSNR_Original: %.6f" % psnr_meter__oiginal_batch.avg

            print(Display_Loss)
            print(Display_SSIM)
            print(Display_PSNR)

            display_time(t1 - t0)

        loss_meter_batch.reset()
        loss_meter_original_batch.reset()
        ssim_meter_batch.reset()
        ssim_meter_original_batch.reset()
        psnr_meter_batch.reset()
        psnr_meter__oiginal_batch.reset()
