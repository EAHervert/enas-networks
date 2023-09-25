import os
import copy
from scipy.io import loadmat, savemat
import numpy as np
import torch
from ENAS_DHDN import SHARED_DHDN as DHDN
from utilities.functions import transform_tensor, get_out, SSIM, PSNR

# Load benchmark data for processing
mat_file = os.getcwd() + '/data/ValidationNoisyBlocksSrgb.mat'
mat_file_gt = os.getcwd() + '/data/ValidationGTBlocksSrgb.mat'
mat = loadmat(mat_file)
mat_gt = loadmat(mat_file_gt)

# Get the model paths
model_edhdn = 'models/2023_09_21__15_51_56_edhdn_SIDD.pth'

# Cast to relevant device
if torch.cuda.is_available():
    device0 = torch.device('cuda:0')
else:
    device0 = torch.device('cpu')

# Model architectures and parameters
encoder1, bottleneck, decoder = [0, 0, 2, 0, 0, 2, 0, 0, 2], [0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]
architecture_edhdn = encoder1 + bottleneck + decoder

edhdn = DHDN.SharedDHDN(architecture=architecture_edhdn)
edhdn.to(device0)
edhdn.load_state_dict(torch.load(model_edhdn, map_location=device0))

# Get the outputs
x_samples = np.array(mat['ValidationNoisyBlocksSrgb'])
t_samples = np.array(mat_gt['ValidationGTBlocksSrgb'])
size = x_samples.shape
y_edhdn_final = []
y_edhdn_final_plus = []

for i in range(size[0]):
    x_sample = x_samples[i, :, :, :, :]
    x_sample_np = x_sample.astype(dtype=float) / 255

    t_sample = t_samples[i, :, :, :, :]
    t_sample_np = t_sample.astype(dtype=float) / 255

    x_sample_pt = torch.tensor(x_sample_np, dtype=torch.float).permute(0, 3, 1, 2)
    r_x_sample_pt = transform_tensor(x_sample_pt, r=1, s=0)
    rr_x_sample_pt = transform_tensor(x_sample_pt, r=2, s=0)
    rrr_x_sample_pt = transform_tensor(x_sample_pt, r=3, s=0)
    s_x_sample_pt = transform_tensor(x_sample_pt, r=0, s=1)
    rs_x_sample_pt = transform_tensor(x_sample_pt, r=1, s=1)
    rrs_x_sample_pt = transform_tensor(x_sample_pt, r=2, s=1)
    rrrs_x_sample_pt = transform_tensor(x_sample_pt, r=3, s=1)

    t_sample_pt = torch.tensor(t_sample_np, dtype=torch.float).permute(0, 3, 1, 2)

    # Cast to Cuda
    x_sample_pt = x_sample_pt.to(device0)
    t_sample_pt = t_sample_pt.to(device0)

    with torch.no_grad():
        y_edhdn = edhdn(x_sample_pt)
        r_y_edhdn = edhdn(r_x_sample_pt.to(device0))
        rr_y_edhdn = edhdn(rr_x_sample_pt.to(device0))
        rrr_y_edhdn = edhdn(rrr_x_sample_pt.to(device0))
        s_y_edhdn = edhdn(s_x_sample_pt.to(device0))
        sr_y_edhdn = edhdn(rs_x_sample_pt.to(device0))
        srr_y_edhdn = edhdn(rrs_x_sample_pt.to(device0))
        srrr_y_edhdn = edhdn(rrrs_x_sample_pt.to(device0))

        y_edhdn_plus = y_edhdn + transform_tensor(r_y_edhdn, r=3, s=0) + transform_tensor(rr_y_edhdn, r=2, s=0) + \
                       transform_tensor(rrr_y_edhdn, r=1, s=0) + transform_tensor(s_y_edhdn, r=0, s=1) + \
                       transform_tensor(sr_y_edhdn, r=3, s=1) + transform_tensor(srr_y_edhdn, r=2, s=1) + \
                       transform_tensor(srrr_y_edhdn, r=1, s=1)
        y_edhdn_plus /= 8

        ssim = [SSIM(x_sample_pt, t_sample_pt),
                SSIM(x_sample_pt, y_edhdn),
                SSIM(x_sample_pt, y_edhdn_plus)]
        ssim_display = "SSIM_Noisy: %.6f" % ssim[0] + "\tSSIM_EDHDN: %.6f" % ssim[1] + \
                       "\tSSIM_EDHDN_Plus: %.6f" % ssim[2]
        mse = [torch.square(x_sample_pt - t_sample_pt).mean(),
               torch.square(x_sample_pt - y_edhdn).mean(),
               torch.square(x_sample_pt - y_edhdn_plus).mean()]
        psnr = [PSNR(mse_i) for mse_i in mse]
        psnr_display = "PSNR_Noisy: %.6f" % psnr[0] + "\tPSNR_EDHDN: %.6f" % psnr[1] + \
                       "\tPSNR_EDHDN_Plus: %.6f" % psnr[2]

        print(ssim_display)
        print(psnr_display)
        print()

    y_edhdn_out = get_out(y_edhdn)
    y_edhdn_out_plus = get_out(y_edhdn_plus)

    y_edhdn_final.append(y_edhdn_out)
    y_edhdn_final_plus.append(y_edhdn_out_plus)

    del x_sample_pt
    del y_edhdn, r_y_edhdn, rr_y_edhdn, rrr_y_edhdn, s_y_edhdn, sr_y_edhdn, srr_y_edhdn, srrr_y_edhdn

# y_edhdn_final = np.array(y_edhdn_final, dtype=np.uint8)
# file_edhdn = 'results/single/edhdn/SubmitSrgb.mat'
# mat_edhdn = copy.deepcopy(mat)
# mat_edhdn['BenchmarkNoisyBlocksSrgb'] = y_edhdn_final
# savemat(file_edhdn, mat_edhdn)
#
# y_edhdn_final_plus = np.array(y_edhdn_final_plus, dtype=np.uint8)
# file_edhdn_plus = 'results/ensemble/edhdn/SubmitSrgb.mat'
# mat_edhdn_plus = copy.deepcopy(mat)
# mat_edhdn_plus['BenchmarkNoisyBlocksSrgb'] = y_edhdn_final_plus
# savemat(file_edhdn_plus, mat_edhdn_plus)
