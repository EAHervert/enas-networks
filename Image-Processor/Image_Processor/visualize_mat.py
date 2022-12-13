import numpy as np
import scipy.io
import os
import torch


class Validation:
    def __init__(self,
                 path='/Users/esauhervert/PycharmProjects/enas-networks/Image-Processor/images/validation/',
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

    def extract_np_arrays(self):
        self.np_NOISY = np.array(self.mat_NOISY['ValidationNoisyBlocksSrgb'] / 255.)
        self.np_GT = np.array(self.mat_GT['ValidationGtBlocksSrgb'] / 255.)

    def transform_to_tensor(self):
        if self.np_NOISY is not None:
            self.tensor_NOISY = torch.tensor(self.np_NOISY).permute(0, 1, 4, 2, 3)
            self.tensor_GT = torch.tensor(self.np_GT).permute(0, 1, 4, 2, 3)
