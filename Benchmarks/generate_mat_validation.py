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
model_dhdn = 'models/2023_09_21__15_51_56_dhdn_SIDD.pth'
model_edhdn = 'models/2023_09_21__15_51_56_edhdn_SIDD.pth'

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
bottleneck, decoder = [0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]

architecture_vanilla = encoder0 + bottleneck + decoder
architecture_edhdn = encoder1 + bottleneck + decoder

dhdn = DHDN.SharedDHDN(architecture=architecture_vanilla)
edhdn = DHDN.SharedDHDN(architecture=architecture_edhdn)

dhdn.to(device0)
edhdn.to(device1)

dhdn.load_state_dict(torch.load(model_dhdn, map_location=device0))
edhdn.load_state_dict(torch.load(model_edhdn, map_location=device1))

# Get the outputs
x_samples = np.array(mat['ValidationNoisyBlocksSrgb'])
t_samples = np.array(mat_gt['ValidationGTBlocksSrgb'])
size = x_samples.shape
y_dhdn_final = []
y_edhdn_final = []
y_dhdn_final_plus = []
y_edhdn_final_plus = []

for i in range(size[0]):
    x_sample = x_samples[i, :, :, :, :]
    x_sample_np = (x_sample.astype(dtype=float) / 255)

    t_sample = t_samples[i, :, :, :, :]
    t_sample_np = (t_sample.astype(dtype=float) / 255)

    x_sample_pt = torch.tensor(x_sample_np, dtype=torch.float).permute(0, 3, 1, 2)
    r_x_sample_pt = transform_tensor(x_sample_pt, r=1, s=0)
    rr_x_sample_pt = transform_tensor(x_sample_pt, r=2, s=0)
    rrr_x_sample_pt = transform_tensor(x_sample_pt, r=3, s=0)
    s_x_sample_pt = transform_tensor(x_sample_pt, r=0, s=1)
    rs_x_sample_pt = transform_tensor(x_sample_pt, r=1, s=1)
    rrs_x_sample_pt = transform_tensor(x_sample_pt, r=2, s=1)
    rrrs_x_sample_pt = transform_tensor(x_sample_pt, r=3, s=1)

    t_sample_pt = torch.tensor(t_sample_np, dtype=torch.float).permute(0, 3, 1, 2)

    with torch.no_grad():
        y_dhdn = dhdn(x_sample_pt.to(device0))
        r_y_dhdn = dhdn(r_x_sample_pt.to(device0))
        rr_y_dhdn = dhdn(rr_x_sample_pt.to(device0))
        rrr_y_dhdn = dhdn(rrr_x_sample_pt.to(device0))
        s_y_dhdn = dhdn(s_x_sample_pt.to(device0))
        sr_y_dhdn = dhdn(rs_x_sample_pt.to(device0))
        srr_y_dhdn = dhdn(rrs_x_sample_pt.to(device0))
        srrr_y_dhdn = dhdn(rrrs_x_sample_pt.to(device0))

        y_dhdn_plus = y_dhdn + transform_tensor(r_y_dhdn, r=3, s=0) + transform_tensor(rr_y_dhdn, r=2, s=0) + \
                      transform_tensor(rrr_y_dhdn, r=1, s=0) + transform_tensor(s_y_dhdn, r=0, s=1) + \
                      transform_tensor(sr_y_dhdn, r=3, s=1) + transform_tensor(srr_y_dhdn, r=2, s=1) + \
                      transform_tensor(srrr_y_dhdn, r=1, s=1)
        y_dhdn_plus /= 8

        y_edhdn = edhdn(x_sample_pt.to(device1))
        r_y_edhdn = edhdn(r_x_sample_pt.to(device1))
        rr_y_edhdn = edhdn(rr_x_sample_pt.to(device1))
        rrr_y_edhdn = edhdn(rrr_x_sample_pt.to(device1))
        s_y_edhdn = edhdn(s_x_sample_pt.to(device1))
        sr_y_edhdn = edhdn(rs_x_sample_pt.to(device1))
        srr_y_edhdn = edhdn(rrs_x_sample_pt.to(device1))
        srrr_y_edhdn = edhdn(rrrs_x_sample_pt.to(device1))

        y_edhdn_plus = y_edhdn + transform_tensor(r_y_edhdn, r=3, s=0) + transform_tensor(rr_y_edhdn, r=2, s=0) + \
                       transform_tensor(rrr_y_edhdn, r=1, s=0) + transform_tensor(s_y_edhdn, r=0, s=1) + \
                       transform_tensor(sr_y_edhdn, r=3, s=1) + transform_tensor(srr_y_edhdn, r=2, s=1) + \
                       transform_tensor(srrr_y_edhdn, r=1, s=1)
        y_edhdn_plus /= 8

        ssim = [SSIM(x_sample_pt.to(device0), t_sample_pt.to(device0)),
                SSIM(x_sample_pt.to(device0), y_dhdn),
                SSIM(x_sample_pt.to(device1), y_edhdn),
                SSIM(x_sample_pt.to(device0), y_dhdn_plus),
                SSIM(x_sample_pt.to(device1), y_edhdn_plus)]
        ssim_display = "SSIM_Noisy: %.6f" % ssim[0] + "\tSSIM_DHDN: %.6f" % ssim[1] + "\tSSIM_EDHDN: %.6f" % ssim[2] + \
                       "\tSSIM_DHDN_Plus: %.6f" % ssim[3] + "\tSSIM_EDHDN_Plus: %.6f" % ssim[4]
        mse = [torch.square(x_sample_pt.to(device1) - t_sample_pt.to(device1)).mean(),
               torch.square(x_sample_pt.to(device0) - y_dhdn).mean(),
               torch.square(x_sample_pt.to(device1) - y_edhdn).mean(),
               torch.square(x_sample_pt.to(device0) - y_dhdn_plus).mean(),
               torch.square(x_sample_pt.to(device1) - y_edhdn_plus).mean()]
        psnr = [PSNR(mse_i) for mse_i in mse]
        psnr_display = "PSNR_Noisy: %.6f" % psnr[0] + "\tPSNR_DHDN: %.6f" % psnr[1] + "\tPSNR_EDHDN: %.6f" % psnr[2] + \
                       "\tPSNR_DHDN_Plus: %.6f" % psnr[3] + "\tPSNR_EDHDN_Plus: %.6f" % psnr[4]

        print(ssim_display)
        print(psnr_display)
        print()

    y_dhdn_out = get_out(y_dhdn)
    y_edhdn_out = get_out(y_edhdn)
    y_dhdn_out_plus = get_out(y_dhdn_plus)
    y_edhdn_out_plus = get_out(y_edhdn_plus)

    y_dhdn_final.append(y_dhdn_out)
    y_dhdn_final_plus.append(y_dhdn_out_plus)
    y_edhdn_final.append(y_edhdn_out)
    y_edhdn_final_plus.append(y_edhdn_out_plus)

    del x_sample_pt
    del y_dhdn, r_y_dhdn, rr_y_dhdn, rrr_y_dhdn, s_y_dhdn, sr_y_dhdn, srr_y_dhdn, srrr_y_dhdn
    del y_edhdn, r_y_edhdn, rr_y_edhdn, rrr_y_edhdn, s_y_edhdn, sr_y_edhdn, srr_y_edhdn, srrr_y_edhdn

y_dhdn_final = np.array(y_dhdn_final, dtype=np.uint8)
file_dhdn = 'results/single/dhdn/SubmitSrgb.mat'
mat_dhdn = copy.deepcopy(mat)
mat_dhdn['BenchmarkNoisyBlocksSrgb'] = y_dhdn_final
savemat(file_dhdn, mat_dhdn)

y_dhdn_final_plus = np.array(y_dhdn_final_plus, dtype=np.uint8)
file_dhdn_plus = 'results/ensemble/dhdn/SubmitSrgb.mat'
mat_dhdn_plus = copy.deepcopy(mat)
mat_dhdn_plus['BenchmarkNoisyBlocksSrgb'] = y_dhdn_final_plus
savemat(file_dhdn_plus, mat_dhdn_plus)

y_edhdn_final = np.array(y_edhdn_final, dtype=np.uint8)
file_edhdn = 'results/single/edhdn/SubmitSrgb.mat'
mat_edhdn = copy.deepcopy(mat)
mat_edhdn['BenchmarkNoisyBlocksSrgb'] = y_edhdn_final
savemat(file_edhdn, mat_edhdn)

y_edhdn_final_plus = np.array(y_edhdn_final_plus, dtype=np.uint8)
file_edhdn_plus = 'results/ensemble/edhdn/SubmitSrgb.mat'
mat_edhdn_plus = copy.deepcopy(mat)
mat_edhdn_plus['BenchmarkNoisyBlocksSrgb'] = y_edhdn_final_plus
savemat(file_edhdn_plus, mat_edhdn_plus)
