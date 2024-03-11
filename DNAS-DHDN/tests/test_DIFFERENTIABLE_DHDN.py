import unittest
import torch

from DNAS_DHDN import DIFFERENTIABLE_DHDN

DRC_BLOCK_1 = [[0.50, 0.50], [0.40, 0.60], [0.70, 0.30]]
DRC_BLOCK_2 = [[0.45, 0.55], [0.70, 0.30], [0.55, 0.45]]
DRC_BLOCK_3 = [[1.00, 0.00], [0.00, 0.10], [0.50, 0.50]]

UP_BLOCK_1 = [0.30, 0.30, 0.40]
UP_BLOCK_2 = [0.00, 1.00, 0.00]

DOWN_BLOCK_1 = [0.40, 0.40, 0.20]
DOWN_BLOCK_2 = [0.50, 0.00, 0.50]

ALPHAS_1 = [[DRC_BLOCK_1, DRC_BLOCK_2, DOWN_BLOCK_1],
            [DRC_BLOCK_1, DRC_BLOCK_2],
            [UP_BLOCK_1, DRC_BLOCK_1, DRC_BLOCK_2]]
ALPHAS_2 = [[DRC_BLOCK_1, DRC_BLOCK_2, DOWN_BLOCK_1],
            [DRC_BLOCK_2, DRC_BLOCK_3, DOWN_BLOCK_2],
            [DRC_BLOCK_1, DRC_BLOCK_2],
            [UP_BLOCK_1, DRC_BLOCK_1, DRC_BLOCK_2],
            [UP_BLOCK_2, DRC_BLOCK_2, DRC_BLOCK_3]]
ALPHAS_3 = [[DRC_BLOCK_1, DRC_BLOCK_2, DOWN_BLOCK_1],
            [DRC_BLOCK_2, DRC_BLOCK_3, DOWN_BLOCK_2],
            [DRC_BLOCK_1, DRC_BLOCK_3, DOWN_BLOCK_1],
            [DRC_BLOCK_1, DRC_BLOCK_2],
            [UP_BLOCK_1, DRC_BLOCK_1, DRC_BLOCK_2],
            [UP_BLOCK_2, DRC_BLOCK_2, DRC_BLOCK_3],
            [UP_BLOCK_1, DRC_BLOCK_1, DRC_BLOCK_3]]


def test_alphas(alphas, k_value):
    DDHDN = DIFFERENTIABLE_DHDN.DifferentiableDHDN(k_value=k_value, channels=16)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)

    y = DDHDN(x, alphas=alphas)

    return list(x.shape) == list(y.shape)


class test_DIFFERENTIABLE_DHDN(unittest.TestCase):
    def test_architecture(self):
        assert test_alphas(ALPHAS_1, k_value=1)
        assert test_alphas(ALPHAS_2, k_value=2)
        assert test_alphas(ALPHAS_3, k_value=3)


if __name__ == '__main__':
    unittest.main()
