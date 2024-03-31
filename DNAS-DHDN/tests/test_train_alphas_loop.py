import unittest
from utilities import dataset
import json
import torch
from torch.utils.data import DataLoader

from DNAS_DHDN.TRAINING_FUNCTIONS import train_alphas_loop
from DNAS_DHDN.DIFFERENTIABLE_DHDN import DifferentiableDHDN
from utilities.functions import generate_w_alphas

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

        # Noise Dataset
        path_training = '/data/' + TRAINING_CSV
        path_validation_noisy = test_config['Locations']['SIDD']['Validation_Noisy']
        path_validation_gt = test_config['Locations']['SIDD']['Validation_GT']

        SIDD_validation = dataset.DatasetMAT(mat_noisy_file=path_validation_noisy,
                                             mat_gt_file=path_validation_gt,
                                             device=DEVICE)
        SIDD_training = dataset.DatasetNoise(csv_file=path_training,
                                             transform=dataset.RandomProcessing(),
                                             device=DEVICE)

        dataloader_sidd_training = DataLoader(dataset=SIDD_training,
                                              batch_size=test_config['Training']['Train_Batch_Size'],
                                              shuffle=True)
        dataloader_sidd_validation = DataLoader(dataset=SIDD_validation,
                                                batch_size=test_config['Training']['Validation_Batch_Size'],
                                                shuffle=False)

        weights = generate_w_alphas(k_val=K_VALUE)
        _ = train_alphas_loop(epoch=0,
                              weights=weights,
                              shared=Differential_Network,
                              epsilon=0.01,
                              lr_w_alpha=0.001,
                              eta=0,
                              config=test_config,
                              dataloader_sidd_training=dataloader_sidd_training,
                              dataloader_sidd_validation=dataloader_sidd_validation,
                              device=DEVICE,
                              verbose=True)

        x = torch.ones(1, 3, 64, 64, device=DEVICE)
        y = Differential_Network(x, weights=weights)

        assert list(x.shape) == list(y.shape)


if __name__ == '__main__':
    unittest.main()
