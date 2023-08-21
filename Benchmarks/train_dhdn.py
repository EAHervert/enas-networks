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
    dataloader_sidd = DataLoader(dataset=SIDD, batch_size=64, shuffle=True, num_workers=32)
else:
    dataloader_sidd = DataLoader(dataset=SIDD, batch_size=16, shuffle=True, num_workers=0)

net = dhdn.Net()
net.to(device)

optim = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
loss = torch.nn.L1Loss()

for epoch in range(EPOCHS):
    loss_val_epoch = 0
    for i_batch, sample_batch in enumerate(dataloader_sidd):
        x = sample_batch['NOISY'].to(device)
        y = net(x)
        loss_val = loss(y, sample_batch['GT'].to(device))
        print('EPOCH:', epoch, 'ITERATION', i_batch, 'Loss', loss_val.item())

        index = i_batch + 1
        loss_val_epoch = ((index - 1) * loss_val_epoch + loss_val.item()) / index
        optim.zero_grad()
        loss_val.backward()
        optim.step()

    print('EPOCH:', epoch, 'MSE', loss_val_epoch)
