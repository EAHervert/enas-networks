import os
import dataset
import dhdn
import time
from torch.utils.data import DataLoader

path = os.getcwd() + '/sidd_instances_064.csv'

SIDD = dataset.DatasetSIDD(csv_file=path, transform=dataset.RandomProcessing())
dataloader_sidd = DataLoader(dataset=SIDD, batch_size=16, shuffle=True, num_workers=0)
net = dhdn.Net()

for i_batch, sample_batch in enumerate(dataloader_sidd):
    time0 = time.time()
    print(i_batch, sample_batch['NOISY'].size(), sample_batch['GT'].size())
    print(net(sample_batch['NOISY']).size())
    print(time.time() - time0)
