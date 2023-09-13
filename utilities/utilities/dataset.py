import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class DatasetSIDD(Dataset):
    """SIDD dataset."""

    def __init__(self, csv_file, transform=None, index_set=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_instances = pd.read_csv(csv_file)
        if index_set is not None:
            self.csv_instances = self.csv_instances[self.csv_instances['INDEX'].isin(index_set)]
        self.transform = transform

    def __len__(self):
        return len(self.csv_instances)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Noisy image
        noisy_name = self.csv_instances.iloc[idx, 1]
        noisy = np.load(noisy_name) / 255.  # numpy array

        # GT image
        gt_name = self.csv_instances.iloc[idx, 2]
        gt = np.load(gt_name) / 255.  # numpy array -> float32

        # Numpy HxWxC -> Torch CxHxW
        noisy_final = torch.tensor(noisy).permute(2, 0, 1)
        gt_final = torch.tensor(gt).permute(2, 0, 1)

        # Final sample output
        sample_item = {'NOISY': noisy_final, 'GT': gt_final}

        if self.transform:
            sample_item = self.transform(sample_item)

        return sample_item


class DatasetDAVIS(Dataset):
    """DAVIS dataset."""

    def __init__(self, csv_file, noise_choice='GAUSSIAN_10', transform=None, index_set=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_instances = pd.read_csv(csv_file)
        self.csv_instances = self.csv_instances[['INDEX', 'GT', noise_choice]]
        if index_set is not None:
            self.csv_instances = self.csv_instances[self.csv_instances['INDEX'].isin(index_set)]
        self.transform = transform

    def __len__(self):
        return len(self.csv_instances)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Noisy image
        noisy_name = self.csv_instances.iloc[idx, 1]
        noisy = np.load(noisy_name) / 255.  # numpy array

        # GT image
        gt_name = self.csv_instances.iloc[idx, 2]
        gt = np.load(gt_name) / 255.  # numpy array -> float32

        # Numpy HxWxC -> Torch CxHxW
        noisy_final = torch.tensor(noisy).permute(2, 0, 1)
        gt_final = torch.tensor(gt).permute(2, 0, 1)

        # Final sample output
        sample_item = {'NOISY': noisy_final, 'GT': gt_final}

        if self.transform:
            sample_item = self.transform(sample_item)

        return sample_item


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