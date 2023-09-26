import os
from utilities.dataset import DatasetSIDD, DatasetSIDDMAT, RandomProcessing
from torch.utils.data import DataLoader


file_064 = os.getcwd() + '/instances/sidd_np_instances_064.csv'
file_val_noisy = os.getcwd() + '/data/ValidationNoisyBlocksSrgb.mat'
file_val_gt = os.getcwd() + '/data/ValidationGtBlocksSrgb.mat'

SIDD_training = DatasetSIDD(csv_file=file_064, transform=RandomProcessing())
SIDD_validation = DatasetSIDDMAT(mat_noisy_file=file_val_noisy, mat_gt_file=file_val_gt, transform=RandomProcessing())

print('Printing samples from the dataset:')
print('\tTraining:', SIDD_training[0]['NOISY'].shape, SIDD_training[0]['GT'].shape, SIDD_training[0]['NOISY'].min(),
      SIDD_training[0]['NOISY'].max())
print('\tValidation:', SIDD_validation[-1]['NOISY'].shape, SIDD_validation[-1]['GT'].shape,
      SIDD_validation[0]['NOISY'].min(), SIDD_validation[0]['NOISY'].max())
print('\n')


print('Testing DataLoader')
dataloader_training = DataLoader(dataset=SIDD_training, batch_size=32, shuffle=True)
dataloader_validation = DataLoader(dataset=SIDD_validation, batch_size=16, shuffle=True)

for i_batch, sample_batch in enumerate(dataloader_training):
    x = sample_batch['NOISY']
    y = sample_batch['GT']

    print('Training Batch', i_batch, ': ', x.size(), y.size(), x.min(), x.max())

for i_batch, sample_batch in enumerate(dataloader_validation):
    x = sample_batch['NOISY']
    y = sample_batch['GT']

    print('Validation Batch', i_batch, ': ', x.size(), y.size(), x.min(), x.max())
