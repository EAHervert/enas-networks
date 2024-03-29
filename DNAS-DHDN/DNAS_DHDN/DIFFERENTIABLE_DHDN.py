# Libraries that will be used.
import torch
import torch.nn as nn
from DNAS_DHDN.RESAMPLING import _down_DNAS, _up_DNAS
from DNAS_DHDN.DRC import _DRC_block_DNAS

from utilities.functions import w_alphas_to_alphas


# Now, here comes the full network.
class DifferentiableDHDN(nn.Module):
    def __init__(self,
                 k_value=3,
                 channels=128,
                 outer_sum=False,
                 ):
        super(DifferentiableDHDN, self).__init__()

        self.k_value = k_value
        self.network_size = 2 * self.k_value + 1
        self.channels = channels
        self.outer_sum = outer_sum
        self._set_network()

    def _set_network(self):
        # Initial convolution
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

        # Let us do the encoder:
        for i in range(self.network_size):

            # First the encoder:
            if i < (self.network_size // 2):
                block1 = _DRC_block_DNAS(channel_in=self.channels * (2 ** i))
                block2 = _DRC_block_DNAS(channel_in=self.channels * (2 ** i))
                down_layer = _down_DNAS(channel_in=self.channels * (2 ** i))

                self.layers.append(block1)
                self.layers.append(block2)
                self.layers.append(down_layer)

            # Now the Bottleneck:
            elif i == self.network_size // 2:
                block1 = _DRC_block_DNAS(channel_in=self.channels * (2 ** i))
                block2 = _DRC_block_DNAS(channel_in=self.channels * (2 ** i))

                self.layers.append(block1)
                self.layers.append(block2)

            # Now the decoder:
            elif i >= (self.network_size // 2 + 1):
                up_layer = _up_DNAS(channel_in=self.channels * 2 * (2 ** (self.network_size - i)))
                block1 = _DRC_block_DNAS(channel_in=self.channels * (2 ** (self.network_size - i)))
                block2 = _DRC_block_DNAS(channel_in=self.channels * (2 ** (self.network_size - i)))

                self.layers.append(up_layer)
                self.layers.append(block1)
                self.layers.append(block2)

        # Final Convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * self.channels,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.PReLU()
        )

    # The value alphas will be the array of values alphas = [alphas_i]
    # The value alpha_i will be the array of weights at level index i of the network
    def forward(self, x, weights):
        alphas = w_alphas_to_alphas(weights, k_val=self.k_value)

        # Initial Convolution:
        out = self.init_conv(x)

        skip = []
        k = 0
        index = -1

        # Loop through the network:
        for i in range(self.network_size):
            alphas_i = alphas[i]

            # First the encoder:
            if i < (self.network_size // 2 + 1):
                # First Block:
                out_1 = self.layers[k](out, alphas_i[0])
                if self.outer_sum:
                    out_1 += out
                k += 1

                # Second Block:
                out_2 = self.layers[k](out_1, alphas_i[1])
                if self.outer_sum:
                    out_2 += out_1
                skip.append(out_2)
                k += 1

                if i < (self.network_size // 2):
                    # Downsample:
                    out = self.layers[k](out_2, alphas_i[2])
                    k += 1

            # Bottleneck Concatenation:
            if i == (self.network_size // 2):
                out = torch.cat((out, skip[index]), dim=1)
                index -= 1

            # Now the decoder:
            if i >= (self.network_size // 2 + 1):
                # Upsample:
                out_1 = self.layers[k](out, alphas_i[0])
                k += 1

                # Concatenate:
                out_1 = torch.cat((skip[index], out_1), dim=1)
                index -= 1

                # First Block:
                out_2 = self.layers[k](out_1, alphas_i[1])
                if self.outer_sum:
                    out_2 += out_1
                k += 1

                # Second Block:
                out = self.layers[k](out_2, alphas_i[2])
                if self.outer_sum:
                    out += out_2
                k += 1

        # Final Layer:
        out = self.final_conv(out)
        out += x

        return out
