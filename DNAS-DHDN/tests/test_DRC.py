import unittest
import torch
import numpy as np

from DNAS_DHDN import DRC


def test_drc_out(size, weights):
    array = [np.argmax(w) for w in weights]

    DRC_DNAS = DRC._DRC_block_DNAS(channel_in=16, size=size)
    DRC_Fixed = DRC._DCR_block_Fixed(channel_in=16, array=array)

    in_tensor = torch.rand(1, 16, 64, 64)
    out_DNAS = DRC_DNAS(in_tensor, alphas=weights)
    out_Fixed = DRC_Fixed(in_tensor)

    return in_tensor.size() == out_DNAS.size() and in_tensor.size() == out_Fixed.size()


class test_DRC(unittest.TestCase):
    def test_out(self):
        # Test 1:
        size_1 = 3
        weights_1 = [[0.4, 0.6], [0.55, 0.45], [0.5, 0.5]]
        self.assertTrue(test_drc_out(size=size_1, weights=weights_1))

        # Test 2:
        size_2 = 4
        weights_2 = [[0.1, 0.9], [0.4, 0.6], [0.55, 0.45], [0.5, 0.5]]
        self.assertTrue(test_drc_out(size=size_2, weights=weights_2))


if __name__ == '__main__':
    unittest.main()
