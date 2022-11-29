import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io

# To allow opening the files I need to open.
torch.multiprocessing.set_sharing_strategy('file_system')


class SIDD_Medium_Dataset_Images(Dataset):
    def __init__(self, image_instances, size_crop, number_crops, transform=None):
        self.image_instances = image_instances
        self.size_crop = size_crop
        self.number_crop = number_crops
        self.transform = transform

        self.paths_lists = [{'INPUT': instance['INPUT'],
                             'TARGET': instance['TARGET']} for instance in self.image_instances]

        self.dataset_crops = []
        index = 1
        for paths in self.paths_lists:
            # print('Loading Image Pairs', index, 'out of', len(self.paths_lists))
            self.dataset_crops.extend(self.crops(paths, self.size_crop, self.number_crop))
            index += 1

    def __len__(self):
        return len(self.dataset_crops)

    def __getitem__(self, item):
        return self.dataset_crops[item]['INPUT'], self.dataset_crops[item]['TARGET']

    @staticmethod
    # Method for generating crops for dataloader purposes
    def crops(paths, size_crop, number_crop):

        # Load Images as np arrays
        image_noisy = io.read_image(paths['INPUT'])
        image_gt = io.read_image(paths['TARGET'])

        x_dim, y_dim, z_dim = image_noisy.shape

        y = random.sample(range(y_dim - size_crop - 1), number_crop)
        z = random.sample(range(z_dim - size_crop - 1), number_crop)

        image_crops = []
        for y_i, z_i in zip(y, z):
            image_crops.append([image_noisy[:, y_i: y_i + size_crop, z_i: z_i + size_crop],
                                image_gt[:, y_i: y_i + size_crop, z_i: z_i + size_crop]])

        dataset = []
        # Transform to torch tensors with correct entries and shape
        for i, j in image_crops:
            dataset.append({'INPUT': i.float() / 255,
                            'TARGET': j.float() / 255})

        return dataset


def load_dataset_images(
        instances,
        batch_size=64,
        size_crop=64,
        number_crops=25
):
    dataset = SIDD_Medium_Dataset_Images(instances, size_crop, number_crops)

    # Create Dataloader:
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader
