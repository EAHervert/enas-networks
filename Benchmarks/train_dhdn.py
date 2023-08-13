import os
import dataset
import dhdn
import torch
from torch.utils.data import DataLoader

# Hyperparameters
EPOCHS = 10

path = os.getcwd() + '/instances/sidd_instances_064.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIDD = dataset.DatasetSIDD(csv_file=path, transform=dataset.RandomProcessing())
if torch.cuda.is_available():
    dataloader_sidd = DataLoader(dataset=SIDD, batch_size=128, shuffle=True, num_workers=32)
else:
    dataloader_sidd = DataLoader(dataset=SIDD, batch_size=16, shuffle=True, num_workers=0)

net = dhdn.Net()
net.to(device)

optim = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
loss = torch.nn.MSELoss()

for _ in range(EPOCHS):
    for i_batch, sample_batch in enumerate(dataloader_sidd):
        print(i_batch, sample_batch['NOISY'].size(), sample_batch['GT'].size())
        x = sample_batch['NOISY'].to(device)
        y = net(x)
        mse = loss(y, sample_batch['GT'].to(device))
        print(mse.item())

        mse.backward()
        optim.step()
