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


class Crops:
    def __init__(self,
                 path='/Users/esauhervert/PycharmProjects/enas-networks/Image-Processor/images/SIDD_Medium_Srgb'):
        self.path = path
        self.path_list = [i for i in os.listdir(self.path)]
        self.folder = None
        self.ground_truth_file = None
        self.noisy_file = None
        self.crops = []

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
    #TODO: Edit code to show all the crops.

    # def plot_crops(self, ):
    #     fig = plt.figure(figsize=(20, 50))
    #     ax = []
    #     for i in range(5):
    #         ax[i] = [fig.add_subplot(2, 2, i)]
    #         ax[i][0].imshow(self.crops[i][0])
    #         axs[i, 1].imshow(self.crops[i][1])
    #
    #         axs[i, 0].title.set_text('Noisy Crop ' + str(i))
    #         axs[i, 1].title.set_text('Ground Truth ' + str(i))
    #
    #     plt.show()

    # def plot_image(self, option='Noisy'):
    #     if option == 'Noisy':
    #         file = self.noisy_file
    #     elif option == 'GT':
    #         file = self.ground_truth_file
    #
    #     image = io.imread(self.path + '/' + self.folder + '/' + file)
    #
    #     plt.imshow(image)
    #     plt.show()
