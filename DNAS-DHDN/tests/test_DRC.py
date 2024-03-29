import unittest
import torch

from DNAS_DHDN import DRC


def test_drc_out(size, alphas):
    DRC_DNAS = DRC._DRC_block_DNAS(channel_in=16, size=size)
    DRC_Fixed = DRC._DCR_block_Fixed(alphas_block=alphas, channel_in=16)

    in_tensor = torch.rand(1, 16, 64, 64)
    out_DNAS = DRC_DNAS(in_tensor, alphas_block=alphas)
    out_Fixed = DRC_Fixed(in_tensor)

    return in_tensor.size() == out_DNAS.size() and in_tensor.size() == out_Fixed.size()


class test_DRC(unittest.TestCase):
    def test_out(self):
        # Test 1:
        size_1 = 3
        alphas_1 = [torch.tensor([0.40, 0.60]),
                    torch.tensor([0.55, 0.45]),
                    torch.tensor([0.50, 0.50])]
        self.assertTrue(test_drc_out(size=size_1, alphas=alphas_1))

        # Test 2:
        size_2 = 4
        alphas_2 = [torch.tensor([0.10, 0.90]),
                    torch.tensor([0.40, 0.60]),
                    torch.tensor([0.55, 0.45]),
                    torch.tensor([0.50, 0.50])]
        self.assertTrue(test_drc_out(size=size_2, alphas=alphas_2))


if __name__ == '__main__':
    unittest.main()
