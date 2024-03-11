import unittest
import torch

from DNAS_DHDN import RESAMPLING

CHANNELS = 16
ALPHAS = [0.1, 0.7, 0.2]


def test_downsampling_dnas(size, alphas):
    x = torch.randn(1, CHANNELS, size[0], size[1])
    target = [1, CHANNELS * 2, size[0] // 2, size[1] // 2]
    dnas_down = RESAMPLING._down_DNAS(channel_in=CHANNELS)

    return list(dnas_down(x, alphas).shape) == target


def test_downsampling_fixed(size):
    x = torch.randn(1, CHANNELS, size[0], size[1])
    target = [1, CHANNELS * 2, size[0] // 2, size[1] // 2]

    bool_vals = []
    for i in range(3):
        fixed_down = RESAMPLING._down_Fixed(channel_in=CHANNELS, architecture_k=i)
        bool_vals.append(list(fixed_down(x).shape) == target)

    return bool_vals[0] and bool_vals[1] and bool_vals[2]


def test_downsampling(size):
    x = torch.randn(1, CHANNELS, size[0], size[1])
    target = [1, CHANNELS * 2, size[0] // 2, size[1] // 2]
    max_pool = RESAMPLING._down_Max(channel_in=CHANNELS)
    avg_pool = RESAMPLING._down_Avg(channel_in=CHANNELS)
    cnv_proc = RESAMPLING._down_Conv(channel_in=CHANNELS)

    bool_max = list(max_pool(x).shape) == target
    bool_avg = list(avg_pool(x).shape) == target
    bool_cnv = list(cnv_proc(x).shape) == target

    return bool_max and bool_avg and bool_cnv


def test_upsampling_dnas(size, alphas):
    x = torch.randn(1, CHANNELS, size[0], size[1])
    target = [1, CHANNELS // 4, size[0] * 2, size[1] * 2]
    dnas_up = RESAMPLING._up_DNAS(channel_in=CHANNELS)

    return list(dnas_up(x, alphas).shape) == target


def test_upsampling_fixed(size):
    x = torch.randn(1, CHANNELS, size[0], size[1])
    target = [1, CHANNELS // 4, size[0] * 2, size[1] * 2]

    bool_vals = []
    for i in range(3):
        fixed_up = RESAMPLING._up_Fixed(channel_in=CHANNELS, architecture_k=i)
        bool_vals.append(list(fixed_up(x).shape) == target)

    return bool_vals[0] and bool_vals[1] and bool_vals[2]


def test_upsampling(size):
    x = torch.randn(1, CHANNELS, size[0], size[1])
    target = [1, CHANNELS // 4, size[0] * 2, size[1] * 2]
    ps_up = RESAMPLING._up_PS(channel_in=CHANNELS)
    convT_pool = RESAMPLING._up_ConvT(channel_in=CHANNELS)
    BL_proc = RESAMPLING._up_BL(channel_in=CHANNELS)

    bool_ps = list(ps_up(x).shape) == target
    bool_convT = list(convT_pool(x).shape) == target
    bool_bl = list(BL_proc(x).shape) == target

    return bool_ps and bool_convT and bool_bl


class test_RESAMPLING(unittest.TestCase):
    def test_sampling(self):
        assert test_downsampling_dnas(size=[64, 64], alphas=ALPHAS)
        assert test_downsampling_fixed(size=[64, 64])
        assert test_downsampling(size=[64, 64])
        assert test_upsampling_dnas(size=[64, 64], alphas=ALPHAS)
        assert test_upsampling_fixed(size=[64, 64])
        assert test_upsampling(size=[64, 64])


if __name__ == '__main__':
    unittest.main()
