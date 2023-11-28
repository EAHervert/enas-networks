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
controller()
sample_arc = controller.sample_arc
print(sample_arc)

shared = SHARED_DHDN.SharedDHDN()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    shared = nn.DataParallel(shared)
    shared = shared.to(device_0)

x = torch.rand([16, 3, 64, 64])
with torch.no_grad():
    y = shared(x.to(device_0), sample_arc)

print(y.dtype)
