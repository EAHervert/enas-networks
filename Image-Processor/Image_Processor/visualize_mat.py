import numpy as np
import scipy.io
import os
import torch
import visdom


class Validation:
    def __init__(self,
                 path='/Users/esauhervert/PycharmProjects/enas-networks/Image-Processor/images/validation/',
                 file_noise='ValidationNoisyBlocksSrgb.mat',
                 file_gt='ValidationGtBlocksSrgb.mat',
                 page='Images'):

        self.vis = None
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

        self.page = page

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

    def evaluate_model(self, model, visualize=False, samples=3):
        sample = torch.randint(0, self.size[0], (samples,))
        sample_crop = np.random.randint(0, self.size[1], samples)

        sample_NOISE = self.tensor_NOISY.index_select(0, sample)
        sample_GT = self.tensor_GT.index_select(0, sample)

        sample_output = model(sample_NOISE)

        if visualize:
            self.visdom_client_setup()

            image_list = []
            for j in range(samples):
                for arrays in [sample_NOISE, sample_GT, sample_output]:
                    image_list.append(arrays[j, sample_crop[j], :, :, :].unsqueeze(0))

            image_batch = torch.cat(image_list)
            self.vis.images(image_batch, nrow=samples, padding=1)

    def visdom_client_setup(self):
        self.vis = visdom.Visdom()
        self.vis.env = self.page
