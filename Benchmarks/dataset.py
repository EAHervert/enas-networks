import random
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class DatasetSIDD(Dataset):
    """SIDD dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_instances = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.csv_instances)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Noisy image
        noisy_name = self.csv_instances.iloc[idx, 0]
        noisy = cv2.imread(noisy_name, cv2.IMREAD_COLOR)  # uint8 image
        noisy = np.array(noisy, dtype=np.single) / 255  # float32

        # GT image
        gt_name = self.csv_instances.iloc[idx, 1]
        gt = cv2.imread(gt_name, cv2.IMREAD_COLOR)  # uint8 image
        gt = np.array(gt, dtype=np.single) / 255  # float32

        # Take image and extract relevant crop
        x_init = self.csv_instances.iloc[idx, 2]
        x_final = self.csv_instances.iloc[idx, 3]
        y_init = self.csv_instances.iloc[idx, 4]
        y_final = self.csv_instances.iloc[idx, 5]

        # Numpy HxWxC -> Torch CxHxW
        noisy_final = torch.tensor(self.crop(noisy, x_init, x_final, y_init, y_final)).permute(2, 0, 1)
        gt_final = torch.tensor(self.crop(gt, x_init, x_final, y_init, y_final)).permute(2, 0, 1)

        # Final sample output
        sample_item = {'NOISY': noisy_final, 'GT': gt_final}

        if self.transform:
            sample_item = self.transform(sample_item)

        return sample_item

    @staticmethod
    def crop(image, x_init, x_final, y_init, y_final):
        return image[x_init:x_final, y_init:y_final, :]


# Randomly flip the image in the H and W channel
class RandomProcessing(object):
    """Randomly flip the image in the H and W channel"""

    def __call__(self, sample):

        noisy = sample['NOISY']
        gt = sample['GT']
        if bool(random.getrandbits(1)):
            noisy = noisy.flip(dims=(1,))
            gt = gt.flip(dims=(1,))
        if bool(random.getrandbits(1)):
            noisy = noisy.flip(dims=(2,))
            gt = gt.flip(dims=(2,))

        return {'NOISY': noisy,
                'GT': gt}
