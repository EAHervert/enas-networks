# Functions
import torch
import numpy as np
import random
import csv
from Pytorch_SSIM import ssim


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


def list_instances(instances_training, instances_validation, partition):
    # Training Instances:
    with open(instances_training, newline='') as f:
        reader = csv.reader(f)
        list_training = list(reader)

    f.close()

    list_training = list_training[1:]

    # Validation Instances:
    with open(instances_validation, newline='') as f:
        reader = csv.reader(f)
        list_validation = list(reader)

    f.close()

    list_validation = list_validation[1:]

    size = len(list_training)

    permutation = np.random.permutation(size)
    training_size = (size // partition) * (partition - 1)

    training_permutation = permutation[0:training_size]
    validation_permutation = permutation[training_size:size]

    training_instances = np.array(list_training)[training_permutation].tolist()
    validation_instances = np.array(list_validation)[validation_permutation].tolist()

    return training_instances, validation_instances


def create_batches(instances, batch_number):
    size = len(instances)
    random.shuffle(instances)

    i = 0

    batch = []
    if batch_number <= size:
        for i in range(size // batch_number):
            begin = i * batch_number
            end = (i + 1) * batch_number

            batch.append(instances[begin:end])

            i += 1

        if end < size:
            batch.append(instances[end:size])
    else:
        batch = [instances]

    return batch


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

    SSIM = ssim(images_x, images_y)
    return SSIM


# PSNR:
# The PSNR is given to us by the formula:
# PSNR = 10 * log_10(1 / MSE)
def PSNR(mse):
    PSNR = 10 * torch.log10(1 / mse)
    return PSNR


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
