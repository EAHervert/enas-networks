from ENAS_DHDN import CONTROLLER
from ENAS_DHDN import SHARED_DHDN
import torch
import torch.nn as nn

device_0 = torch.device('cuda:0')
print(device_0)
controller = CONTROLLER.Controller(k_value=3,
                                   kernel_bool=True,
                                   down_bool=True,
                                   up_bool=True,
                                   LSTM_size=32,
                                   LSTM_num_layers=1)
controller = controller.to(device_0)

print('Controller Weights')
for param in controller.parameters():
    print(param.dtype)

shared = SHARED_DHDN.SharedDHDN()
print('Shared Weights')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    shared = nn.DataParallel(shared)
    shared = shared.to(device_0)

for param in shared.parameters():
    print(param.dtype)

controller()
sample_arc = controller.sample_arc
print(sample_arc)

x = torch.rand([16, 3, 64, 64]).cuda()
with torch.no_grad():
    y = shared(x, sample_arc)

print(y.dtype)
