import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io


class SIDD_Medium_Dataset_Images(Dataset):
    def __init__(self, image_instances, size_crop, number_crops, transform=None):
        self.image_instances = image_instances
        self.size_crop = size_crop
        self.number_crop = number_crops
        self.transform = transform
        self.dataset = []

        self.folders = [i for i in self.image_instances]

        for path in self.image_instances:
            self.dataset.extend(self.crops(path, self.size_crop, self.number_crop))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    @staticmethod
    # Method for generating crops for dataloader purposes
    def crops(path, size_crop, number_crop):

        # Load Images as np arrays
        png_path_noisy, png_path_gt = select_paths(path)
        image_noisy = io.imread(png_path_noisy)
        image_gt = io.imread(png_path_gt)

        x_dim, y_dim, _ = image_noisy.shape

        x = random.sample(range(x_dim - size_crop - 1), number_crop)
        y = random.sample(range(y_dim - size_crop - 1), number_crop)

        image_crops = []
        for x_i in x:
            for y_i in y:
                image_crops.append([image_noisy[x_i: x_i + size_crop, y_i: y_i + size_crop, :],
                                    image_gt[x_i: x_i + size_crop, y_i: y_i + size_crop, :]])

        dataset = []
        # Transform to torch tensors with correct entries and shape
        for i, j in image_crops:
            i_ = torch.from_numpy(i).float() / 255
            j_ = torch.from_numpy(j).float() / 255
            dataset.append({'Noisy': i_.permute(2, 0, 1),
                            'GT': j_.permute(2, 0, 1)})


def select_paths(path):
    files = os.listdir(path)

    set_1 = []
    set_2 = []
    for file in files:
        if '_010.' in files:
            set_1.append(file)
        else:
            set_2.append(file)

    randint = random.randint(1)

    if randint:
        for file in set_1:
            if 'NOISY' in file:
                png_path_noisy = file
            else:
                png_path_gt = file
    else:
        for file in set_2:
            if 'NOISY' in file:
                png_path_noisy = file
            else:
                png_path_gt = file

    return png_path_noisy, png_path_gt
