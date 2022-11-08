# Libraries that will be used.
import torch
import torch.nn as nn
from ENAS_DHDN.RESAMPLING import _down_ENAS, _down_Fixed, _up_ENAS, _up_Fixed
from ENAS_DHDN.DRC import _DRC_block_ENAS, _DCR_block_Fixed


# Now, here comes the full network.
class SharedDHDN(nn.Module):
    def __init__(self,
                 k_value=3,
                 channels=128,
                 architecture=None
                 ):
        super(SharedDHDN, self).__init__()

        self.network_size = 2 * k_value + 1
        self.channels = channels
        self.architecture = architecture

        # Initial and final convolutions
        self.init_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.PReLU()
        )

        # Where all the layers will be
        self.layers = nn.ModuleList([])

        k = 0

        # Let us do the encoder:
        for i in range(self.network_size):

            # First the encoder:
            if i < (self.network_size // 2 + 1):
                if architecture is None:
                    layer1 = _DRC_block_ENAS(self.channels * (2 ** i))
                    layer2 = _DRC_block_ENAS(self.channels * (2 ** i))
                    down_layer = _down_ENAS(self.channels * (2 ** i))

                else:
                    a11 = self.architecture[k] % 2
                    a12 = self.architecture[k + 1] % 2

                    if (self.architecture[k] == 1) or (self.architecture[k] == 5):
                        a21 = 1
                    else:
                        a21 = 0
                    if (self.architecture[k + 1] == 1) or (self.architecture[k + 1] == 5):
                        a22 = 1
                    else:
                        a22 = 0

                    a31 = self.architecture[k] // 4
                    a32 = self.architecture[k + 1] // 4

                    layer1 = _DCR_block_Fixed(self.channels * (2 ** i), [a11, a21, a31])
                    layer2 = _DCR_block_Fixed(self.channels * (2 ** i), [a12, a22, a32])

                    down_layer = _down_Fixed(self.channels * (2 ** i), self.architecture[k + 2])

                    k += 2

                self.layers.append(layer1)
                self.layers.append(layer2)

                if i < (self.network_size // 2):
                    self.layers.append(down_layer)

                    k += 1

            # Now the decoder:
            if i >= (self.network_size // 2 + 1):
                if architecture is None:
                    up_layer = _up_ENAS(self.channels * 2 * (2 ** (self.network_size - i)))
                    layer1 = _DRC_block_ENAS(self.channels * (2 ** (self.network_size - i)))
                    layer2 = _DRC_block_ENAS(self.channels * (2 ** (self.network_size - i)))

                else:
                    up_layer = _up_Fixed(self.channels * 2 * (2 ** (self.network_size - i)), self.architecture[k])
                    a11 = self.architecture[k + 1] % 2
                    a12 = self.architecture[k + 2] % 2

                    if (self.architecture[k + 1] == 1) or (self.architecture[k + 1] == 5):
                        a21 = 1
                    else:
                        a21 = 0
                    if (self.architecture[k + 2] == 1) or (self.architecture[k + 2] == 5):
                        a22 = 1
                    else:
                        a22 = 0

                    a31 = self.architecture[k + 1] // 4
                    a32 = self.architecture[k + 2] // 4

                    layer1 = _DCR_block_Fixed(self.channels * (2 ** (self.network_size - i)), [a11, a21, a31])
                    layer2 = _DCR_block_Fixed(self.channels * (2 ** (self.network_size - i)), [a12, a22, a32])

                    k += 3

                self.layers.append(up_layer)
                self.layers.append(layer1)
                self.layers.append(layer2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * self.channels,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, x, architecture):

        # Initial Convolution:
        out = self.init_conv(x)

        skip = []
        k = 0
        index = -1

        # Loop through the network:
        for i in range(self.network_size):

            # First the encoder:
            if i < (self.network_size // 2 + 1):
                out_0 = out

                # First Block:
                a1 = architecture[k] % 2

                if (architecture[k] == 1) or (architecture[k] == 5):
                    a2 = 1
                else:
                    a2 = 0

                a3 = architecture[k] // 4
                out_1 = self.layers[k](out, [a1, a2, a3])
                out_1 += out_0
                k += 1

                # Second Block:
                a1 = architecture[k] % 2

                if (architecture[k] == 1) or (architecture[k] == 5):
                    a2 = 1
                else:
                    a2 = 0

                a3 = architecture[k] // 4
                out_2 = self.layers[k](out, [a1, a2, a3])
                out = out_1 + out_2
                k += 1
                skip.append(out)

                if i < (self.network_size // 2):
                    # Downsample:
                    out = self.layers[k](out, architecture[k])
                    k += 1

            # Bottleneck Concatenation:
            if i == (self.network_size // 2):
                out = torch.cat((out, skip[index]), dim=1)
                index -= 1

            # Now the decoder:
            if i >= (self.network_size // 2 + 1):

                # Upsample:
                out = self.layers[k](out, architecture[k])
                k += 1

                # Concatenate:
                out = torch.cat((out, skip[index]), dim=1)
                index -= 1

                # First Block:
                out_0 = out

                # First Block:
                a1 = architecture[k] % 2

                if (architecture[k] == 1) or (architecture[k] == 5):
                    a2 = 1
                else:
                    a2 = 0

                a3 = architecture[k] // 4
                out_1 = self.layers[k](out, [a1, a2, a3])
                out_1 += out_0
                k += 1

                # Second Block:
                a1 = architecture[k] % 2

                if (architecture[k] == 1) or (architecture[k] == 5):
                    a2 = 1
                else:
                    a2 = 0

                a3 = architecture[k] // 4
                out_2 = self.layers[k](out, [a1, a2, a3])
                out = out_1 + out_2
                k += 1

        # Final Layer:
        out = self.final_conv(out)
        out += x

        return out
