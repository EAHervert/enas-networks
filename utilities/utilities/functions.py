# Functions
import os
import json
import pandas as pd  # Saving CSV
import torch
import numpy as np
import random
import csv
from utilities.Pytorch_SSIM import ssim
from utilities.utils import AverageMeter


def macro_array(k, kernel_array, down_array, up_array):
    array = []

    i1 = 0
    i2 = 0
    i3 = 0

    for i in range(3 * k):
        if (i + 1) % 3 != 0:
            array.append(kernel_array[i1])
            i1 += 1
        else:
            array.append(down_array[i2])
            i2 += 1

    for i in range(2):
        array.append(kernel_array[i1])
        i1 += 1

    for i in range(3 * k):
        if i % 3 == 0:
            array.append(up_array[i3])
            i3 += 1

        else:
            array.append(kernel_array[i1])
            i1 += 1

    return array


def display_time(time):
    time = abs(time)  # In case of Negative Time
    hours = time // (60 * 60)
    minutes = (time // 60) % 60
    seconds = time % 60

    display = 'Total Time: ' + str(int(hours)) + ' hours, ' + str(int(minutes)) + \
              ' minutes, and ' + str(int(seconds)) + ' seconds.'

    print(display)
    print()

    return None


def rand_mod(tensor1, tensor2):
    # Randomly rotates and flips tensors.

    rand_rot = random.randint(0, 3)
    rand_flip = random.randint(2, 3)

    tensor1 = torch.rot90(tensor1, rand_rot, [2, 3])
    tensor1 = torch.flip(tensor1, [rand_flip])

    tensor2 = torch.rot90(tensor2, rand_rot, [2, 3])
    tensor2 = torch.flip(tensor2, [rand_flip])

    return tensor1, tensor2


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def augmentate(tensor):
    # Tensors are of size BxCxHxW. We want to rotate and flip about the height and width.

    b, _, _, _ = tensor.size()

    t_90 = 0
    t_180 = 0
    t_270 = 0
    t_fh = 0
    t_fv = 0

    for i in range(b):
        # Rotate
        t_90 = torch.rot90(tensor.clone(), 1, [2, 3])
        t_180 = torch.rot90(tensor.clone(), 2, [2, 3])
        t_270 = torch.rot90(tensor.clone(), 3, [2, 3])

        # Flip

        t_fh = torch.flip(tensor.clone(), [2])
        t_fv = torch.flip(tensor.clone(), [3])

    tensor_out = torch.cat((tensor, t_90, t_180, t_270, t_fh, t_fv), dim=0)

    return tensor_out


# SSIM:
# From: https://github.com/Po-Hsun-Su/pytorch-ssim
def SSIM(images_x, images_y):
    # if images_x.dim() == 5:
    #     images_x = image_reshuffle(images_x[:, 0, :, :, :])
    #     images_y = image_reshuffle(images_y[:, 0, :, :, :])

    ssim_ = ssim(images_x, images_y)
    return ssim_


# PSNR:
# The PSNR is given to us by the formula:
# PSNR = 10 * log_10(1 / MSE)
def PSNR(mse):
    psnr = 10 * torch.log10(1 / mse)
    return psnr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def generate_loggers():
    # Image Batches
    loss_batch = AverageMeter()
    loss_original_batch = AverageMeter()
    ssim_batch = AverageMeter()
    ssim_original_batch = AverageMeter()
    psnr_batch = AverageMeter()
    psnr_original_batch = AverageMeter()

    batch_loggers = (loss_batch, loss_original_batch, ssim_batch, ssim_original_batch, psnr_batch, psnr_original_batch)

    # Validation
    loss_meter_val = AverageMeter()
    loss_original_meter_val = AverageMeter()
    ssim_meter_val = AverageMeter()
    ssim_original_meter_val = AverageMeter()
    psnr_meter_val = AverageMeter()
    psnr_original_meter_val = AverageMeter()

    val_loggers = (loss_meter_val, loss_original_meter_val, ssim_meter_val, ssim_original_meter_val, psnr_meter_val,
                   psnr_original_meter_val)

    return batch_loggers, val_loggers
