import numpy as np
import scipy.io
import os
import torch

from utilities.functions import SSIM, PSNR
from PIL import ImageDraw, ImageFont
import torchvision.transforms as T


class Validation:
    def __init__(self,
                 path,
                 file_noise='ValidationNoisyBlocksSrgb.mat',
                 file_gt='ValidationGtBlocksSrgb.mat'):

        self.np_NOISY = None
        self.np_GT = None
        self.tensor_NOISY = None
        self.tensor_GT = None
        self.path = path
        self.path_noisy = os.path.join(self.path, file_noise)
        self.path_gt = os.path.join(self.path, file_gt)
        self.mat_NOISY = scipy.io.loadmat(self.path_noisy)
        self.mat_GT = scipy.io.loadmat(self.path_gt)

        self.extract_np_arrays()
        self.transform_to_tensor()

        if self.tensor_GT is not None:
            self.size = self.tensor_GT.size()
        else:
            self.size = 0

        self.loss = torch.nn.MSELoss()

        # define a transform to convert a tensor to PIL image
        self.transform = T.ToPILImage()
        self.inverse_transform = T.PILToTensor()

        # Todo: use relative path
        self.font = ImageFont.truetype(path + '/Image_Processor/fonts/arial.ttf', 15)

    def extract_np_arrays(self):
        self.np_NOISY = np.array(self.mat_NOISY['ValidationNoisyBlocksSrgb'])
        self.np_GT = np.array(self.mat_GT['ValidationGtBlocksSrgb'])

    @staticmethod
    def np_to_tensor(array):
        return torch.tensor(array / 255.).permute(0, 1, 4, 2, 3)

    @staticmethod
    def tensor_to_np(tensor):
        return torch.round(tensor * 255).permute(0, 1, 3, 4, 2).numpy()

    def transform_to_tensor(self):
        if self.np_NOISY is not None:
            self.tensor_NOISY = self.np_to_tensor(self.np_NOISY)
            self.tensor_GT = self.np_to_tensor(self.np_GT)

    def evaluate_model(self, model, architecture, samples=3):
        sample = torch.randint(0, self.size[0], (samples,))

        sample_NOISE = self.tensor_NOISY.index_select(0, sample)
        sample_GT = self.tensor_GT.index_select(0, sample)

        # Todo: Transform images to inputs that model will take
        sample_output = model(sample_NOISE, architecture=architecture)

        # Compare GT to model output
        mse_out = self.loss(sample_output, sample_GT)
        mse_original = self.loss(sample_NOISE, sample_GT)
        metrics = {'loss': mse_out,
                   'loss_original': mse_original,
                   'PSNR': PSNR(mse_out),
                   'PSNR_original': PSNR(mse_original),
                   'SSIM': SSIM(sample_output, sample_GT),
                   'SSIM_original': SSIM(sample_NOISE, sample_GT)}
