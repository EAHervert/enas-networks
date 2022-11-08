import torch
import torch.nn as nn


# Example of a block of size 3:
# ---> add
# - -> concat

#  -----------------------------------------------
# |                                               v
# I-> conv -> relu -> conv -> relu -> conv ->relu -> O
# |                 ^               ^
#  - - - - - - - - - - - - - - - - >|
# |                                 |
#  - - - - - - - - - - - - - - - - -

# This class will represent the DRC block as a graph where we take the DRC block above and allow for the following:
#   Different number of convolutions (parametarized by "size"
class _DRC_block_ENAS(nn.Module):
    def __init__(self,
                 channel_in,
                 size=3):
        super(_DRC_block_ENAS, self).__init__()

        self.graph = []

        for i in range(size):
            if i != size - 1:
                temp_conv3 = nn.Conv2d(
                    in_channels=int((1 + i * .5) * channel_in),
                    out_channels=int(channel_in / 2.),
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
                temp_conv5 = nn.Conv2d(
                    in_channels=int((1 + i * .5) * channel_in),
                    out_channels=int(channel_in / 2.),
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            else:
                temp_conv3 = nn.Conv2d(
                    in_channels=int((1 + i * .5) * channel_in),
                    out_channels=channel_in,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
                temp_conv5 = nn.Conv2d(
                    in_channels=int((1 + i * .5) * channel_in),
                    out_channels=channel_in,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            node_conv = [temp_conv3, temp_conv5]
            self.graph.append(node_conv)

            node_relu = nn.PReLU()
            self.graph.append(node_relu)

    def forward(self, x, array):
        residual = x

        in_val = x
        index = 0
        for val in array[:-1]:
            out_val = self.graph[index][val](in_val)
            out_val = self.graph[index + 1](out_val)

            in_val = torch.cat([in_val, out_val], dim=1)
            index += 2

        out_val = self.graph[index][array[-1]](in_val)
        out_val = self.graph[index + 1](out_val)

        out_val = torch.add(out_val, residual)

        return out_val

# This will be the branch that is fixed depending on the architecture.
class _DCR_block_Fixed(nn.Module):
    def __init__(self,
                 channel_in,
                 array):
        super(_DCR_block_Fixed, self).__init__()

        self.path = nn.ModuleList([])

        for index, val in enumerate(array):
            if index != len(array) - 1:
                self.path.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=int((1 + index * .5) * channel_in),
                            out_channels=int(channel_in / 2.),
                            kernel_size=2 * val + 3,
                            stride=1,
                            padding=val + 1
                        ),
                        nn.PReLU()
                    )
                )
            else:
                self.path.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=int((1 + index * .5) * channel_in),
                            out_channels=channel_in,
                            kernel_size=2 * val + 3,
                            stride=1,
                            padding=val + 1
                        ),
                        nn.PReLU()
                    )
                )


    def forward(self, x, array):
        residual = x

        out = self.path[0](x)

        cat = torch.cat([x, out], 1)
        out = self.path[1](cat)

        cat = torch.cat([cat, out], 1)
        out = self.path[2](cat)

        out = torch.add(out, residual)

        return out
