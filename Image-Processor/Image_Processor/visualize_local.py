import random
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Visualizer:
    def __init__(self,
                 path='/Users/esauhervert/PycharmProjects/enas-networks/Image-Processor/images/SIDD_Medium_Srgb'):
        self.path = path
        self.path_list = []
        self.folder = None
        self.ground_truth_file = None
        self.noisy_file = None

    def set_path_list(self):
        self.path_list = [i for i in os.listdir(self.path)]

    def get_path_list(self):
        return self.path_list

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

    def get_random_images(self):
        return [self.noisy_file, self.ground_truth_file]

    def plot_image(self, option='Noisy'):
        if option == 'Noisy':
            file = self.noisy_file
        elif option == 'GT':
            file = self.ground_truth_file

        image = mpimg.imread(self.path + '/' + self.folder + '/' + file)

        plt.imshow(image)
        plt.show()
