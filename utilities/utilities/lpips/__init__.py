from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utilities.lpips import *
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision.models import alexnet as tv_alexnent
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim
from skimage import color
import cv2
import rawpy
import numpy as np

import warnings

warnings.filterwarnings("ignore")


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False,
                 pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for tests mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS, self).__init__()
        if verbose:
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]' %
                  ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type == 'alex':
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if pretrained:
                if model_path is None:
                    import inspect
                    import os
                    model_path = os.path.abspath(
                        os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)))
                    print(model_path)

                if verbose:
                    print('Loading model from: %s' % model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        if eval_mode:
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (
            in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), \
                                     normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if self.spatial:
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]

        # Todo: Make this value s.t. it can be backpropagated on
        if retPerLayer:
            return val, res
        else:
            return val


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv_alexnent(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True), ]
        if use_sigmoid:
            layers += [nn.Sigmoid(), ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.) / 2.
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):
    def forward(self, in0, in1):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if self.colorspace == 'RGB':
            (N, C, X, Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y),
                               dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = l2(tensor2np(tensor2tensorlab(in0.data, to_norm=False)),
                       tensor2np(tensor2tensorlab(in1.data, to_norm=False)),
                       range=100.).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            if self.use_gpu:
                ret_var = ret_var.cuda()
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if self.colorspace == 'RGB':
            value = dssim(1. * tensor2im(in0.data), 1. * tensor2im(in1.data), range=255.).astype('float')
        elif self.colorspace == 'Lab':
            value = dssim(tensor2np(tensor2tensorlab(in0.data, to_norm=False)),
                          tensor2np(tensor2tensorlab(in1.data, to_norm=False)),
                          range=100.).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        if self.use_gpu:
            ret_var = ret_var.cuda()
        return ret_var


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network', net)
    print('Total number of parameters: %d' % num_params)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def l2(p0, p1, range=255.):
    return .5 * np.mean((p0 / range - p1 / range) ** 2)


def psnr(p0, p1, peak=255.):
    return 10 * np.log10(peak ** 2 / np.mean((1. * p0 - 1. * p1) ** 2))


def dssim(p0, p1, range=255.):
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.


def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1, 2, 0))


def np2tensor(np_obj):
    # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    # image tensor to lab tensor
    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if to_norm and not mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab = img_lab / 100.

    return np2tensor(img_lab)


def tensorlab2tensor(lab_tensor, return_inbnd=False):
    lab = tensor2np(lab_tensor) * 100.
    lab[:, :, 0] = lab[:, :, 0] + 50

    rgb_back = 255. * np.clip(color.lab2rgb(lab.astype('float')), 0, 1)
    if return_inbnd:
        # convert back to lab, see if we match
        lab_back = color.rgb2lab(rgb_back.astype('uint8'))
        mask = 1. * np.isclose(lab_back, lab, atol=2.)
        mask = np2tensor(np.prod(mask, axis=2)[:, :, np.newaxis])
        return (im2tensor(rgb_back), mask)
    else:
        return im2tensor(rgb_back)


def load_image(path):
    if path[-3:] == 'dng':
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    elif path[-3:] == 'bmp' or path[-3:] == 'jpg' or path[-3:] == 'png' or path[-4:] == 'jpeg':
        return cv2.imread(path)[:, :, ::-1]
    else:
        import matplotlib.pyplot as plt
        img = (255 * plt.imread(path)[:, :, :3]).astype('uint8')

    return img


def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255. / 2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)


def im2tensor(image, cent=1., factor=255. / 2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def tensor2vec(vector_tensor):
    return vector_tensor.data.cpu().numpy()[:, :, 0, 0]


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If the value use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
