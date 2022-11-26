import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


# Crop Class for local analysis
class Crops:
    def __init__(self,
                 path='/Users/esauhervert/PycharmProjects/enas-networks/Image-Processor/images/SIDD_Medium_Srgb'):
        self.path = path
        self.path_list = [i for i in os.listdir(self.path)]
        self.folder = None
        self.ground_truth_file = None
        self.noisy_file = None
        self.crops = []
        self.dataset = []

    def set_random_images(self):
        # Select a random folder
        folder_length = len(self.path_list)
        val1 = random.randint(0, folder_length - 1)
        val2 = random.randint(0, 1)

        if val2:
            str_val = '010.PNG'
        else:
            str_val = '011.PNG'

        self.folder = self.path_list[val1]
        files = [i for i in os.listdir(self.path + '/' + self.folder)]
        for file in files:
            if str_val in file:
                if 'GT' in file:
                    self.ground_truth_file = file
                else:
                    self.noisy_file = file

    def get_random_crops(self, size=64, sample=5):
        image_noisy = io.imread(self.path + '/' + self.folder + '/' + self.noisy_file)
        image_gt = io.imread(self.path + '/' + self.folder + '/' + self.ground_truth_file)

        x_dim, y_dim, _ = image_noisy.shape

        x = random.sample(range(x_dim - size - 1), sample)
        y = random.sample(range(y_dim - size - 1), sample)

        for x_i in x:
            for y_i in y:
                self.crops.append([image_noisy[x_i: x_i + size, y_i: y_i + size, :],
                                   image_gt[x_i: x_i + size, y_i: y_i + size, :]])

    def set_tensors(self):

        for i, j in self.crops:
            i_ = torch.from_numpy(i).float() / 255
            j_ = torch.from_numpy(j).float() / 255
            self.dataset.append({'Noisy': i_.permute(2, 0, 1),
                                 'GT': j_.permute(2, 0, 1)})

    def plot_crops(self, ):

        noisy = [crop[0] for crop in self.crops]
        gt = [crop[1] for crop in self.crops]

        _, axs = plt.subplots(2, 5, figsize=(10, 25))
        axs = axs.flatten()
        for index, ax in enumerate(axs):

            if index < 5:
                ax.imshow(noisy[index])
                ax.set_title('NOISY ' + str(index))
            else:
                ax.imshow(gt[index - 5])
                ax.set_title('GT ' + str(index - 5))

        plt.show()

    def plot_tensors(self):

        noisy = [crop['Noisy'] for crop in self.dataset]
        gt = [crop['GT'] for crop in self.dataset]

        _, axs = plt.subplots(2, 5, figsize=(10, 25))
        axs = axs.flatten()
        for index, ax in enumerate(axs):

            if index < 5:
                ax.imshow(noisy[index].permute(1, 2, 0))
                ax.set_title('NOISY ' + str(index))
            else:
                ax.imshow(gt[index - 5].permute(1, 2, 0))
                ax.set_title('GT ' + str(index - 5))

        plt.show()


