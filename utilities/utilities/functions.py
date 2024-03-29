import math
import torch
import numpy as np
import random
from utilities.Pytorch_SSIM import ssim
from utilities.utils import AverageMeter


def val_to_kernel_array(k):
    return k % 2, 1 * (k % 4 > 1), k // 4


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


def generate_cyclegan_loggers():
    # Image Batches
    ssim_meter_batch = AverageMeter()
    ssim_original_meter_batch = AverageMeter()
    psnr_meter_batch = AverageMeter()
    psnr_original_meter_batch = AverageMeter()
    loss_DX = AverageMeter()
    loss_DY = AverageMeter()
    loss_GANG = AverageMeter()
    loss_GANF = AverageMeter()
    loss_Cyc_XYX = AverageMeter()
    loss_Cyc_YXY = AverageMeter()
    loss_IX = AverageMeter()
    loss_IY = AverageMeter()
    loss_Sup_XY = AverageMeter()
    loss_Sup_YX = AverageMeter()

    batch_loggers = (ssim_meter_batch, ssim_original_meter_batch, psnr_meter_batch, psnr_original_meter_batch,
                     loss_DX, loss_DY, loss_GANG, loss_GANF, loss_Cyc_XYX, loss_Cyc_YXY, loss_IX, loss_IY, loss_Sup_XY,
                     loss_Sup_YX)

    # Validation
    ssim_meter_val = AverageMeter()
    ssim_original_meter_val = AverageMeter()
    psnr_meter_val = AverageMeter()
    psnr_original_meter_val = AverageMeter()

    val_loggers = (ssim_meter_val, ssim_original_meter_val, psnr_meter_val, psnr_original_meter_val)

    return batch_loggers, val_loggers


def generate_gan_loggers():
    # Image Batches
    ssim_meter_batch = AverageMeter()
    ssim_original_meter_batch = AverageMeter()
    psnr_meter_batch = AverageMeter()
    psnr_original_meter_batch = AverageMeter()
    loss_D = AverageMeter()
    loss_GANG = AverageMeter()
    loss_IY = AverageMeter()
    loss_Sup_XY = AverageMeter()

    batch_loggers = (ssim_meter_batch, ssim_original_meter_batch, psnr_meter_batch, psnr_original_meter_batch,
                     loss_D, loss_GANG, loss_IY, loss_Sup_XY)

    # Validation
    ssim_meter_val = AverageMeter()
    ssim_original_meter_val = AverageMeter()
    psnr_meter_val = AverageMeter()
    psnr_original_meter_val = AverageMeter()

    val_loggers = (ssim_meter_val, ssim_original_meter_val, psnr_meter_val, psnr_original_meter_val)

    return batch_loggers, val_loggers


def drop_weights(state_dict, p=0.8, device='cpu'):
    state_dict_out = state_dict.copy()

    for key in state_dict.keys():
        tensor = state_dict_out[key]
        if tensor.dtype == torch.float32 and list(tensor.size()) != [1]:
            mask = (torch.randn(tensor.size()) < p) * 1.
            tensor = torch.mul(tensor, mask.to(device))

        state_dict_out[key] = tensor

    return state_dict_out


def gaussian_add_weights(state_dict, k=1, device='cpu'):
    state_dict_out = state_dict.copy()

    for key in state_dict.keys():
        tensor = state_dict_out[key]
        if tensor.dtype == torch.float32 and list(tensor.size()) != [1]:
            std = tensor.std().item()
            noise = torch.clip(k * std * torch.randn(tensor.size()), -k * std, k * std)
            tensor += noise.to(device)

        state_dict_out[key] = tensor

    return state_dict_out


def clip_weights(state_dict, k=1, device='cpu'):
    state_dict_out = state_dict.copy()

    for key in state_dict.keys():
        tensor = state_dict_out[key]
        if tensor.dtype == torch.float32:
            std = tensor.std().item()
            tensor = torch.clamp(tensor, -k * std, k * std)

        state_dict_out[key] = tensor

    return state_dict_out


def get_out(out_tensor):
    out_np = out_tensor.permute(0, 2, 3, 1).cpu().numpy()[:, :, :, :]
    out = np.clip((out_np * 255).round(), 0, 255).astype(np.uint8).tolist()

    return out


def transform_tensor(in_tensor, r=0, s=0):
    out_tensor = in_tensor.clone().detach()
    if s == 1:
        out_tensor = torch.flip(in_tensor, dims=[2, 3])
    if r != 0:
        out_tensor = torch.rot90(out_tensor, k=r, dims=[2, 3])

    return out_tensor


def image_np_to_tensor(np_array, i_range=9, j_range=15, crop_size=256):
    np_array_pt = torch.zeros(i_range, j_range, 3, crop_size, crop_size)
    for i in range(i_range):
        for j in range(j_range):
            sample = torch.tensor(np_array[i * crop_size:(i + 1) * crop_size,
                                  j * crop_size:(j + 1) * crop_size, :] / 255).permute(2, 0, 1)
            np_array_pt[i, j, :, :sample.size()[1], :sample.size()[2]] = sample

    return np_array_pt


def tensor_to_np_image(tensor, i_out=2160, j_out=3840, crop_size=256):
    np_out = np.zeros([i_out, j_out, 3])
    i_range = math.ceil(i_out / crop_size)
    j_range = math.ceil(j_out / crop_size)

    for i in range(i_range):
        for j in range(j_range):
            sample = tensor[i, j, :, :, :].permute(1, 2, 0) * 255
            sample = np.round(sample.numpy())

            if i_range % crop_size:
                if i < 8:
                    np_out[i * crop_size:(i + 1) * crop_size, j * crop_size:(j + 1) * crop_size, :] = sample
                else:
                    final = 2160 - i * crop_size
                    np_out[i * crop_size:, j * crop_size:(j + 1) * crop_size, :] = sample[:final, :, :]
            else:
                np_out[i * crop_size:(i + 1) * crop_size, j * crop_size:(j + 1) * crop_size, :] = sample

    return np_out


def random_architecture_generation(k_value=3, kernel_bool=True, down_bool=True, up_bool=True):
    encoder, bottleneck, decoder = [], [], []
    for _ in range(k_value):
        decoder.append(random.randint(0, 2) if down_bool else 0)
        for _ in range(2):
            encoder.append(random.randint(0, 7) if kernel_bool else 0)
            decoder.append(random.randint(0, 7) if kernel_bool else 0)

        encoder.append(random.randint(0, 2) if up_bool else 0)

    for _ in range(2):
        bottleneck.append(random.randint(0, 7) if kernel_bool else 0)

    return encoder + bottleneck + decoder


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def generate_w_alphas(k_val=3, s_val=1e-3):
    return s_val * torch.randn(2 * (k_val * (6 + 6 + 3)) + 6)


# TODO: Function for learning weights
def w_alphas_to_alphas(w_alphas, k_val=3):
    return 0


def generate_alphas(k_val=3, blocks=3, randomize=False):
    encoder = []
    bottleneck = []
    decoder = []
    block = [0.5, 0.5]
    block_reshape = [0.33, 0.33, 0.33]
    for k in range(k_val):
        encoder_temp, decoder_temp = [], []
        if randomize:
            block_reshape = softmax(np.random.rand(3)).tolist()
        decoder_temp.append(block_reshape)
        for _ in range(2):
            # DRC Blocks
            DRC_temp = [[], [], []]
            for _ in range(blocks):
                for i in range(3):
                    if randomize:
                        block = softmax(np.random.rand(2)).tolist()
                    DRC_temp[i].append(block)

            encoder_temp.append(DRC_temp[0])
            decoder_temp.append(DRC_temp[1])

        if randomize:
            block_reshape = softmax(np.random.rand(3)).tolist()
        encoder_temp.append(block_reshape)

        encoder.append(encoder_temp)
        decoder.append(decoder_temp)

    for _ in range(2):
        bottleneck_temp = []
        for _ in range(blocks):
            if randomize:
                block = softmax(np.random.rand(2)).tolist()
            bottleneck_temp.append(block)
        bottleneck.append(bottleneck_temp)

    return encoder + [bottleneck] + decoder
