import unittest
import torch

from ENAS_DHDN import SHARED_DHDN

ENCODER_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
ENCODER_2 = [0, 0, 2, 0, 0, 2, 0, 0, 2]

BOTTLENECK = [0, 0]

DECODER_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
DECODER_2 = [1, 2, 3, 2, 3, 4, 0, 5, 6]


def test_shared_dhdn(architecture, initialize=False):
    if initialize:
        model = SHARED_DHDN.SharedDHDN(architecture=architecture)
    else:
        model = SHARED_DHDN.SharedDHDN()
    x = torch.randn(1, 3, 64, 64)
    y = model(x, architecture=architecture)

    return list(x.shape) == list(y.shape)


class test_SHARED_DHDN(unittest.TestCase):
    assert test_shared_dhdn(architecture=ENCODER_1 + BOTTLENECK + ENCODER_1, initialize=True)
    assert test_shared_dhdn(architecture=ENCODER_2 + BOTTLENECK + ENCODER_2)
