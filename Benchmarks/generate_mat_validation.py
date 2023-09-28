import os
import datetime
import json
from scipy.io import loadmat
import torch
from ENAS_DHDN import SHARED_DHDN as DHDN
from utilities.functions import transform_tensor, get_out, SSIM, PSNR
import utilities.dataset as dataset
from torch.utils.data import DataLoader

current_time = datetime.datetime.now()
d1 = current_time.strftime('%Y_%m_%d__%H_%M_%S')
dir_current = os.getcwd()
config_path = dir_current + '/configs/config_dhdn.json'
config = json.load(open(config_path))

# Load benchmark data for processing
mat_file = os.getcwd() + config['Locations']['Validation_Noisy']
mat_file_gt = os.getcwd() + config['Locations']['Validation_GT']
mat = loadmat(mat_file)
mat_gt = loadmat(mat_file_gt)

# Get the model paths
model_dhdn = os.getcwd() + config['Training']['Model_Path_DHDN']
model_edhdn = os.getcwd() + config['Training']['Model_Path_EDHDN']

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

SIDD_validation = dataset.DatasetSIDDMAT(mat_noisy_file=mat_file, mat_gt_file=mat_file_gt)
dataloader_sidd_validation = DataLoader(dataset=SIDD_validation, batch_size=config['Training']['Validation_Batch_Size'],
                                        shuffle=False, num_workers=8)

y_dhdn_final = []
y_edhdn_final = []
y_dhdn_final_plus = []
y_edhdn_final_plus = []

transforms = [[1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]]
for i_batch, sample_batch in enumerate(dataloader_sidd_validation):
    x_sample_pt = sample_batch['NOISY']
    t_sample_pt = sample_batch['GT']
    with torch.no_grad():
        y_dhdn = dhdn(x_sample_pt.to(device0))
        y_edhdn = edhdn(x_sample_pt.to(device1))

        y_dhdn_plus = y_dhdn.detach().clone()
        y_edhdn_plus = y_edhdn.detach().clone()
        for transform in transforms:
            y_dhdn_plus += transform_tensor(dhdn(transform_tensor(x_sample_pt,
                                                                  r=transform[0], s=transform[1]).to(device0)),
                                            r=4 - transform[0], s=transform[1])
            y_edhdn_plus += transform_tensor(edhdn(transform_tensor(x_sample_pt,
                                                                    r=transform[0], s=transform[1]).to(device1)),
                                             r=4 - transform[0], s=transform[1])
        y_dhdn_plus /= 8
        y_edhdn_plus /= 8

        ssim = [SSIM(x_sample_pt.to(device0), t_sample_pt.to(device0)),
                SSIM(x_sample_pt.to(device0), y_dhdn),
                SSIM(x_sample_pt.to(device1), y_edhdn),
                SSIM(x_sample_pt.to(device0), y_dhdn_plus),
                SSIM(x_sample_pt.to(device1), y_edhdn_plus)
                ]
        ssim_display = "SSIM_Noisy: %.6f" % ssim[0] + "\tSSIM_DHDN: %.6f" % ssim[1] + "\tSSIM_EDHDN: %.6f" % ssim[2] + \
                       "\tSSIM_DHDN_Plus: %.6f" % ssim[3] + "\tSSIM_EDHDN_Plus: %.6f" % ssim[4]
        mse = [torch.square(x_sample_pt.to(device1) - t_sample_pt.to(device1)).mean(),
               torch.square(x_sample_pt.to(device0) - y_dhdn).mean(),
               torch.square(x_sample_pt.to(device1) - y_edhdn).mean(),
               torch.square(x_sample_pt.to(device0) - y_dhdn_plus).mean(),
               torch.square(x_sample_pt.to(device1) - y_edhdn_plus).mean()
               ]
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
    del y_dhdn
    del y_edhdn

# y_dhdn_final = np.array(y_dhdn_final, dtype=np.uint8)
# file_dhdn = 'results/single/dhdn/SubmitSrgb.mat'
# mat_dhdn = copy.deepcopy(mat)
# mat_dhdn['BenchmarkNoisyBlocksSrgb'] = y_dhdn_final
# savemat(file_dhdn, mat_dhdn)
#
# y_dhdn_final_plus = np.array(y_dhdn_final_plus, dtype=np.uint8)
# file_dhdn_plus = 'results/ensemble/dhdn/SubmitSrgb.mat'
# mat_dhdn_plus = copy.deepcopy(mat)
# mat_dhdn_plus['BenchmarkNoisyBlocksSrgb'] = y_dhdn_final_plus
# savemat(file_dhdn_plus, mat_dhdn_plus)
#
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
