import torch
import torchvision.io as io
import plotly.express as px

path = '/Users/esauhervert/PycharmProjects/enas-networks/Image-Processor/images/SIDD_Medium_Srgb/0001_001_S6_00100_00060_3200_L'
image_noisy = '0001_NOISY_SRGB_010.PNG'
image_gt = '0001_GT_SRGB_010.PNG'

original = io.read_image(path + '/' + image_noisy)

image_noisy_data = io.read_image(path + '/' + image_noisy).float() / 255
image_gt_data = io.read_image(path + '/' + image_gt).float() / 255
final = (image_noisy_data.permute((1, 2, 0)) * 255).type(torch.int8)
fig = px.imshow(final)
fig.show()

print(torch.mean((original.permute((1, 2, 0)) - final).float()))
