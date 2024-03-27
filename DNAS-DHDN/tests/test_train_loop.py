import unittest
from utilities import dataset
import json
import torch
from torch.utils.data import DataLoader

from DNAS_DHDN.TRAINING_FUNCTIONS import train_loop
from DNAS_DHDN.DIFFERENTIABLE_DHDN import DifferentiableDHDN

DRC_BLOCK_1 = [[0.50, 0.50], [0.40, 0.60], [0.70, 0.30]]
DRC_BLOCK_2 = [[0.45, 0.55], [0.70, 0.30], [0.55, 0.45]]
UP_BLOCK_1 = [0.30, 0.30, 0.40]
DOWN_BLOCK_1 = [0.40, 0.40, 0.20]

ALPHAS_1 = [[DRC_BLOCK_1, DRC_BLOCK_2, DOWN_BLOCK_1],
            [DRC_BLOCK_1, DRC_BLOCK_2],
            [UP_BLOCK_1, DRC_BLOCK_1, DRC_BLOCK_2]]

TRAINING_CSV = 'sidd_np_instances_064_0001.csv'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
K_VALUE = 1
Learning_Rate = 0.001
WEIGHT_DECAY = 0.00


class TestTrainLoop(unittest.TestCase):

    def test_train_loop(self):
        config_path = 'data/config.json'
        with open(config_path, 'r') as f:
            test_config = json.load(f)

        Differential_Network = DifferentiableDHDN(k_value=K_VALUE)
        Differential_Network.to(device=DEVICE)
        optimizer = torch.optim.Adam(Differential_Network.parameters(), lr=Learning_Rate, weight_decay=WEIGHT_DECAY)

        # Noise Dataset
        path_training = 'data/' + TRAINING_CSV

        dataset_training = dataset.DatasetNoise(csv_file=path_training,
                                                transform=dataset.RandomProcessing(),
                                                device=DEVICE)

        dataloader_training = DataLoader(dataset=dataset_training,
                                         batch_size=test_config['Training']['Train_Batch_Size'],
                                         shuffle=True)

        _ = train_loop(epoch=0,
                       alphas=ALPHAS_1,
                       shared=Differential_Network,
                       shared_optimizer=optimizer,
                       config=test_config,
                       dataloader_sidd_training=dataloader_training,
                       device=DEVICE,
                       verbose=False)

        x = torch.randn(1, 3, 64, 64, requires_grad=True, device=DEVICE)
        y = Differential_Network(x, alphas=ALPHAS_1)

        assert list(x.shape) == list(y.shape)


if __name__ == '__main__':
    unittest.main()
