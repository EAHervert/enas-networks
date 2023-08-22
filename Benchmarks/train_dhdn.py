import os
import dataset
import dhdn
import torch
from torch.utils.data import DataLoader

# Hyperparameters
EPOCHS = 10

path_training = os.getcwd() + '/instances/instances_064.csv'
path_validation = os.getcwd() + '/instances/instances_256.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIDD_training = dataset.DatasetSIDD(csv_file=path_training, transform=dataset.RandomProcessing())
SIDD_validation = dataset.DatasetSIDD(csv_file=path_validation, transform=dataset.RandomProcessing())

if torch.cuda.is_available():
    dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=64, shuffle=True, num_workers=16)
    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=32, shuffle=True, num_workers=8)
else:
    dataloader_sidd_training = DataLoader(dataset=SIDD_training, batch_size=16, shuffle=True, num_workers=0)
    dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=4, shuffle=True, num_workers=0)

net = dhdn.Net()
net.to(device)

optim = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
loss = torch.nn.L1Loss()
MSE = torch.nn.MSELoss()
for epoch in range(EPOCHS):
    loss_val_epoch = 0
    mse_val_validation = 0
    for i_batch, sample_batch in enumerate(dataloader_sidd_training):
        x = sample_batch['NOISY'].to(device)
        y = net(x)
        loss_val = loss(y, sample_batch['GT'].to(device))
        print('EPOCH:', epoch, 'ITERATION:', i_batch, 'Loss', loss_val.item())

        index = i_batch + 1
        loss_val_epoch = ((index - 1) * loss_val_epoch + loss_val.item()) / index
        optim.zero_grad()
        loss_val.backward()
        optim.step()

    print('\nEPOCH:', epoch, 'Loss:', loss_val_epoch)

    for i_validation, validation_batch in enumerate(dataloader_sidd_validation):
        x = validation_batch['NOISY'].to(device)
        with torch.no_grad():
            y = net(x)
            MSE_val = MSE(y, validation_batch['GT'].to(device))

        index = i_validation + 1
        mse_val_validation = ((index - 1) * mse_val_validation + MSE_val.item()) / index
        break  # Only do one pass for Validation

    print('\nEPOCH:', epoch, 'Validation MSE:', mse_val_validation)

model_path = os.getcwd() + '/models/sidd_dhdn.pth'
torch.save(net.state_dict(), model_path)
