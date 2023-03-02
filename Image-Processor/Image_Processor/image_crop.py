import os
import torch
from skimage import io
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Crop Class for local analysis
class Crops:
    def __init__(self, path):
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

    def get_random_crops(self, size=64, sample_x=5, sample_y=5):
        image_noisy = io.imread(self.path + '/' + self.folder + '/' + self.noisy_file)
        image_gt = io.imread(self.path + '/' + self.folder + '/' + self.ground_truth_file)

        x_dim, y_dim, _ = image_noisy.shape

        x = random.sample(range(x_dim - size - 1), sample_x)
        y = random.sample(range(y_dim - size - 1), sample_y)

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
