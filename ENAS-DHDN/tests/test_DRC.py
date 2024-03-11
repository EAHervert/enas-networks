import unittest
import torch

from ENAS_DHDN import DRC


def test_drc_out(size, array):
    DRC_ENAS = DRC._DRC_block_ENAS(channel_in=16, size=size)
    DRC_Fixed = DRC._DCR_block_Fixed(channel_in=16, array=array)

    in_tensor = torch.rand(1, 16, 64, 64)
    out_ENAS = DRC_ENAS(in_tensor, array=array)
    out_Fixed = DRC_Fixed(in_tensor, array=array)

    return in_tensor.size() == out_ENAS.size() and in_tensor.size() == out_Fixed.size()


class test_DRC(unittest.TestCase):
    def test_out(self):
        # Test 1:
        size = 3
        array = [0, 1, 0]
        self.assertTrue(test_drc_out(size=size, array=array))


if __name__ == '__main__':
    unittest.main()
