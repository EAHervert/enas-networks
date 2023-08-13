import os
import dataset
import dhdn
import time
import torch
from torch.utils.data import DataLoader

path = os.getcwd() + '/instances/sidd_instances_064.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIDD = dataset.DatasetSIDD(csv_file=path, transform=dataset.RandomProcessing())
if torch.cuda.is_available():
    dataloader_sidd = DataLoader(dataset=SIDD, batch_size=128, shuffle=True, num_workers=16)
else:
    dataloader_sidd = DataLoader(dataset=SIDD, batch_size=16, shuffle=True, num_workers=0)

net = dhdn.Net()
net.to(device)

for i_batch, sample_batch in enumerate(dataloader_sidd):
    time0 = time.time()
    print(i_batch, sample_batch['NOISY'].size(), sample_batch['GT'].size())
    x = sample_batch['NOISY'].to(device)
    print(net(x).size())
    print(time.time() - time0)
