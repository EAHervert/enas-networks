import unittest
import os
import json

import torch
from DNAS_DHDN.TRAINING_FUNCTIONS import train_loop
from DNAS_DHDN.DIFFERENTIABLE_DHDN import DifferentiableDHDN


DRC_BLOCK_1 = [[0.50, 0.50], [0.40, 0.60], [0.70, 0.30]]
DRC_BLOCK_2 = [[0.45, 0.55], [0.70, 0.30], [0.55, 0.45]]
UP_BLOCK_1 = [0.30, 0.30, 0.40]
DOWN_BLOCK_1 = [0.40, 0.40, 0.20]

ALPHAS_1 = [[DRC_BLOCK_1, DRC_BLOCK_2, DOWN_BLOCK_1],
            [DRC_BLOCK_1, DRC_BLOCK_2],
            [UP_BLOCK_1, DRC_BLOCK_1, DRC_BLOCK_2]]

TRAINING_CSV = 'sidd_np_instances_064_0016'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
K_VALUE = 1


class TestTrainLoop(unittest.TestCase):

    def test_train_loop(self):
        config_path = 'data/config.json'
        test_config = json.load(open(config_path))

        Differential_Network = DifferentiableDHDN(k_value=K_VALUE)

        # Noise Dataset
        path_training = 'data/' + TRAINING_CSV
        path_validation_noisy = test_config['Locations']['Validation_Noisy']
        path_validation_gt = test_config['Locations']['Validation_GT']

        dataset_training = dataset.DatasetNoise(csv_file=path_training,
                                             transform=dataset.RandomProcessing(),
                                             device=DEVICE)

        dataloader_training = DataLoader(dataset=dataset_training,
                                         batch_size=config['Training']['Train_Batch_Size'],
                                         shuffle=True)

        out = train_loop(epoch=0,
                         alphas=ALPHAS_1,
                         shared=Differential_Network,
                         config=test_config,
                         dataloader_sidd_training=dataloader_training,
                         device=DEVICE)

        print(out)