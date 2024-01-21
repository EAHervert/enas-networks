import random

from scipy.io import loadmat
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class DatasetSIDD(Dataset):
    """SIDD dataset."""

    def __init__(self, csv_file, transform=None, raw_images=False, device='cpu'):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_instances = pd.read_csv(csv_file)
        self.transform = transform
        self.raw_images = raw_images
        self.device = torch.device(device)

    def __len__(self):
        return len(self.csv_instances)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Noisy image
        noisy_name = self.csv_instances.iloc[idx, 1]
        noisy = np.load(noisy_name)  # numpy array

        # GT image
        gt_name = self.csv_instances.iloc[idx, 2]
        gt = np.load(gt_name)  # numpy array -> float32

        # Numpy HxWxC -> Torch CxHxW
        noisy_final = torch.tensor(noisy, device=self.device).permute(2, 0, 1)
        gt_final = torch.tensor(gt, device=self.device).permute(2, 0, 1)

        # Final sample output
        if self.raw_images:
            sample_item = {'NOISY': noisy_final / 255.,
                           'NOISY_RAW': noisy_final,
                           'GT': gt_final / 255.,
                           'GT_RAW': gt_final}
        else:
            sample_item = {'NOISY': noisy_final / 255.,
                           'GT': gt_final / 255.}

        if self.transform:
            sample_item = self.transform(sample_item)

        return sample_item


class DatasetSIDDMAT(Dataset):
    """SIDD dataset from .mat file."""

    def __init__(self, mat_noisy_file, mat_gt_file=None, transform=None, raw_images=False, device='cpu'):
        """
        Arguments:
            .mat file (.mat): Path to the mat file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Noisy samples
        self.mat_noisy_dict = loadmat(mat_noisy_file)
        self.mat_noisy = self.mat_noisy_dict[next(reversed(self.mat_noisy_dict))]
        self.size_noisy = self.mat_noisy.shape
        self.device = torch.device(device)

        # GT if available
        if mat_gt_file is not None:
            self.gt_avail = True
            self.mat_gt_dict = loadmat(mat_gt_file)
            self.mat_gt = self.mat_gt_dict[next(reversed(self.mat_gt_dict))]
            self.size_gt = self.mat_gt.shape

            assert self.size_noisy == self.size_gt
        else:
            self.gt_avail = False

        self.transform = transform
        self.raw_images = raw_images

    def __len__(self):
        return self.size_noisy[0] * self.size_noisy[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_0 = idx // self.size_noisy[1]
        idx_1 = idx % self.size_noisy[1]

        noisy = self.mat_noisy[idx_0, idx_1]  # Noisy image

        # GT image
        if self.gt_avail:
            gt = self.mat_gt[idx_0, idx_1]
        else:
            gt = np.zeros_like(noisy)

        # Numpy HxWxC -> Torch CxHxW
        noisy_final = torch.tensor(noisy, device=self.device).permute(2, 0, 1)
        gt_final = torch.tensor(gt, device=self.device).permute(2, 0, 1)

        # Final sample output
        if self.raw_images:
            sample_item = {'NOISY': noisy_final / 255.,
                           'NOISY_RAW': noisy_final,
                           'GT': gt_final / 255.,
                           'GT_RAW': gt_final}
        else:
            sample_item = {'NOISY': noisy_final / 255.,
                           'GT': gt_final / 255.}

        if self.transform:
            sample_item = self.transform(sample_item)

        return sample_item


# Todo: Update Davis Dataset
class DatasetDAVIS(Dataset):
    """DAVIS dataset."""

    def __init__(self, csv_file, noise_choice='GAUSSIAN_10', transform=None, index_set=None, raw_images=False):
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
        self.raw_images = raw_images

    def __len__(self):
        return len(self.csv_instances)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Noisy image
        noisy_name = self.csv_instances.iloc[idx, 1]
        noisy = np.load(noisy_name, dtype=np.single)  # numpy array

        # GT image
        gt_name = self.csv_instances.iloc[idx, 2]
        gt = np.load(gt_name, dtype=np.single)  # numpy array -> float32

        # Numpy HxWxC -> Torch CxHxW
        noisy_final = torch.tensor(noisy).permute(2, 0, 1)
        gt_final = torch.tensor(gt).permute(2, 0, 1)

        # Final sample output
        if self.raw_images:
            sample_item = {'NOISY': noisy_final / 255.,
                           'NOISY_RAW': noisy_final,
                           'GT': gt_final / 255.,
                           'GT_RAW': gt_final}
        else:
            sample_item = {'NOISY': noisy_final / 255.,
                           'GT': gt_final / 255.}

        if self.transform:
            sample_item = self.transform(sample_item)

        return sample_item


# Randomly flip the image in the H and W channel
class RandomProcessing(object):
    """Randomly flip the image in the H and W channel"""
    def __init__(self, cutout_images=False):
        self.cutout_images = cutout_images

    def __call__(self, sample):
        r_val = np.random.choice([0, 1, 2, 3])  # Rotates image 90 degrees r_val times
        s_val = np.random.choice([0, 1])  # Flip image book

        noisy = sample['NOISY']
        gt = sample['GT']

        if s_val:
            noisy = torch.flip(noisy, dims=[1, 2])
            gt = torch.flip(gt, dims=[1, 2])

        noisy = torch.rot90(noisy, k=r_val, dims=[1, 2])
        gt = torch.rot90(gt, k=r_val, dims=[1, 2])

        if self.cutout_images:
            noisy, gt = self.cutout(noisy, gt, 8)

        return {'NOISY': noisy,
                'GT': gt}

    @staticmethod
    def cutout(image1, image2, size):
        """Cutout a size x size block to the original image"""
        out1 = image1.copy()
        out2 = image2.copy()
        h, w, c = image1.shape
        h_i = random.randrange(h - size)
        w_i = random.randrange(w - size)
        out1[h_i:h_i + size, w_i:w_i + size] = np.zeros((size, size, c))
        out2[h_i:h_i + size, w_i:w_i + size] = np.zeros((size, size, c))

        return out1, out2
