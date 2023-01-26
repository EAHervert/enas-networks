import numpy as np
import scipy.io
import os
import torch
import visdom

from utilities.functions import SSIM, PSNR
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
import torchvision.transforms as T


class Validation:
    def __init__(self,
                 path='/Users/esauhervert/PycharmProjects/enas-networks/Image-Processor/images/validation/',
                 file_noise='ValidationNoisyBlocksSrgb.mat',
                 file_gt='ValidationGtBlocksSrgb.mat',
                 page='Testing'):

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
        self.vis_window = {}

        self.loss = torch.nn.MSELoss()

        # define a transform to convert a tensor to PIL image
        self.transform = T.ToPILImage()
        self.inverse_transform = T.PILToTensor()

        # Todo: use relative path
        self.font = ImageFont.truetype("/Users/esauhervert/PycharmProjects/enas-networks/Image-Processor/Image_Processor/fonts/arial.ttf", 15)

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

    def evaluate_model(self, model, architecture, visualize=False, samples=3, index=0):
        sample = torch.randint(0, self.size[0], (samples,))
        sample_crop = np.random.randint(0, self.size[1], samples)

        sample_NOISE = self.tensor_NOISY.index_select(0, sample)
        sample_GT = self.tensor_GT.index_select(0, sample)

        # Todo: Transform images to inputs that model will take
        sample_output = model(sample_NOISE, architecture=architecture)

        if visualize:
            subpage = self.page + "_image_set_{index}".format(index=str(index))
            self.vis_window[subpage] = None

            image_batch_temp = []
            for j in range(samples):
                image_set = []
                for arrays in [sample_NOISE, sample_GT, sample_output]:
                    image_set.append(arrays[j, sample_crop[j], :, :, :].unsqueeze(0))

                image_batch_temp.append(torch.cat(image_set))

            image_batch = []
            for image_set_ in image_batch_temp:
                N_i = image_set_[0]
                G_i = image_set_[1]
                T_i = image_set_[2]

                # convert the tensor to PIL image using above transform
                image = self.transform(torch.ones_like(N_i))

                I1 = ImageDraw.Draw(image)

                mse_base = self.loss(N_i, G_i)
                mse_targ = self.loss(T_i, G_i)

                psnr_base = PSNR(mse_base).item()
                psnr_targ = PSNR(mse_targ).item()

                ssim_base = SSIM(N_i.unsqueeze(0), G_i.unsqueeze(0)).item()
                ssim_targ = SSIM(T_i.unsqueeze(0), G_i.unsqueeze(0)).item()

                text_noisy = "PSNR (Noisy Image): " + str(round(psnr_base, 4)) + \
                             "\nSSIM (Noisy Image): " + str(round(ssim_base, 4))

                text_output = "PSNR (Output Image): " + str(round(psnr_targ, 4)) + \
                              "\nSSIM (Output Image): " + str(round(ssim_targ, 4))

                text_out = text_noisy + '\n\n' + text_output

                I1.text((15, 15), text_out, font=self.font, fill=(0, 0, 0))
                image_tensor = self.inverse_transform(image) / 255.
                image_batch_ = torch.cat([N_i.unsqueeze(0), G_i.unsqueeze(0), T_i.unsqueeze(0),
                                          image_tensor.unsqueeze(0)])
                image_batch.append(image_batch_)

            image_batch_final = torch.cat(image_batch)

            self.vis_window[subpage] = self.vis.images(image_batch_final, nrow=samples + 1, padding=1,
                                                       win=self.vis_window[subpage],
                                                       opts=dict(title='IMAGES {index}'.format(index=str(index)),
                                                                 xlabel='[Noisy, GT, Out]', ylabel='Image Index'))

    def visdom_client_setup(self):
        self.vis = visdom.Visdom()
        self.vis.env = self.page
