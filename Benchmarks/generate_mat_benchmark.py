import os
from scipy.io import loadmat, savemat
import numpy as np
import torch
from ENAS_DHDN import SHARED_DHDN as DHDN


# Load benchmark data for processing
mat_file = os.getcwd() + '/data/BenchmarkNoisyBlocksSrgb.mat'
mat = loadmat(mat_file)

# Get the model paths
model_dhdn = 'models/2023_09_11_dhdn.pth'
model_edhdn = 'models/2023_09_11_edhdn.pth'

# Cast to relevant device
if torch.cuda.is_available():
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
else:
    device0 = torch.device('cpu')
    device1 = device0

# Model architectures and parameters
encoder0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
encoder1 = [0, 0, 2, 0, 0, 2, 0, 0, 2]
bottleneck = [0, 0]
decoder = [0, 0, 0, 0, 0, 0, 0, 0, 0]

architecture_vanilla = encoder0 + bottleneck + decoder
dhdn = DHDN.SharedDHDN(architecture=architecture_vanilla)
dhdn.load_state_dict(torch.load(model_dhdn, map_location=device0))

architecture_edhdn = encoder1 + bottleneck + decoder
edhdn = DHDN.SharedDHDN(architecture=architecture_edhdn)
edhdn.load_state_dict(torch.load(model_edhdn, map_location=device1))

# Get the outputs
x_samples = np.array(mat['BenchmarkNoisyBlocksSrgb'])
size = x_samples.shape
y_dhdn_final = []
y_edhdn_final = []
for i in range(size[0]):
    x_sample = x_samples[i, :, :, :, :]
    x_sample_np = (x_sample.astype(dtype=float) / 255)[:, :, :, ::-1]
    x_sample_pt = torch.tensor(x_sample_np.copy(), dtype=torch.float32).permute(0, 3, 1, 2)

    with torch.no_grad():
        y_dhdn = dhdn(x_sample_pt.to(device0))
        y_edhdn = dhdn(x_sample_pt.to(device1))

    y_dhdn_out_np = y_dhdn.permute(0, 2, 3, 1).numpy()[:, :, :, ::-1]
    y_dhdn_out = np.clip((y_dhdn_out_np * 255).round(), 0, 255).astype(np.uint8).tolist()

    y_edhdn_out_np = y_edhdn.permute(0, 2, 3, 1).numpy()[:, :, :, ::-1]
    y_edhdn_out = np.clip((y_edhdn_out_np * 255).round(), 0, 255).astype(np.uint8)

    y_dhdn_final.append(y_dhdn_out)
    y_edhdn_final.append(y_dhdn_out)

y_dhdn_final = np.array(y_dhdn_final, dtype=np.uint8)
y_edhdn_final = np.array(y_edhdn_final, dtype=np.uint8)

file_dhdn = '/results/dhdn/SubmitSrgb.mat'
file_edhdn = '/results/edhdn/SubmitSrgb.mat'

mat_dhdn = mat
mat_edhdn = mat

mat_dhdn['BenchmarkNoisyBlocksSrgb'] = y_dhdn_final
mat_edhdn['BenchmarkNoisyBlocksSrgb'] = y_edhdn_final

savemat(file_dhdn, mat_dhdn)
savemat(file_edhdn, mat_edhdn)


