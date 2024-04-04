import unittest

from ENAS_DHDN import CONTROLLER
from ENAS_DHDN import SHARED_DHDN
import torch
import torch.nn as nn


class test_controller(unittest.TestCase):

    def test_controller(self):
        # Cast to GPU if available
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        # Initialize controller
        controller = CONTROLLER.ReducedController(k_value=3,
                                                  encoder=True,
                                                  bottleneck=True,
                                                  decoder=True,
                                                  lstm_size=32,
                                                  lstm_num_layers=1)

        controller = controller.to(device)
        controller()
        sample_arc = controller.sample_arc

        shared = SHARED_DHDN.SharedDHDN()

        # If more than one GPU:
        if torch.cuda.device_count() > 1:
            shared = nn.DataParallel(shared)
            shared = shared.to(device)

        x = torch.rand([16, 3, 64, 64])
        with torch.no_grad():
            y = shared(x.to(device), sample_arc)
        print(sample_arc)
        assert list(x.shape) == list(y.shape)


if __name__ == '__main__':
    unittest.main()
