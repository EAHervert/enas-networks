import torch
import torch.nn as nn


# Example:
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
#   Different number of convolutions parameterized by "size"
class _DRC_block_DNAS(nn.Module):
    def __init__(self,
                 channel_in,
                 size=3):
        super(_DRC_block_DNAS, self).__init__()

        self.graph = nn.ModuleList([])

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
            node_conv = nn.ModuleList([temp_conv3, temp_conv5])
            self.graph.append(node_conv)

            node_relu = nn.PReLU()
            self.graph.append(node_relu)

    def forward(self, x, alphas):
        residual = x
        in_val = x
        out_val = 0
        for index, vals in enumerate(alphas[:-1]):
            for index_v, val in enumerate(vals):
                out_val += val * self.graph[2 * index][index_v](in_val)
            out_val = self.graph[2 * index + 1](out_val)  # PReLU activation

            in_val = torch.cat([in_val, out_val], dim=1)  # Concatenation (Dense Learning)

        out_val_final = 0
        for vals in alphas[-1:]:
            for index_v, val in enumerate(vals):
                out_val_final += val * self.graph[-2][index_v](in_val)
        out_val = self.graph[- 1](out_val_final)  # PReLU activation

        out_val = torch.add(out_val, residual)  # Residual learning

        return out_val


# This will be the branch that is fixed depending on the architecture.
class _DCR_block_Fixed(nn.Module):
    def __init__(self,
                 channel_in,
                 array):
        super(_DCR_block_Fixed, self).__init__()

        self.array = array
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

    def forward(self, x):
        residual = x
        cat = x
        for op in self.path[:-1]:
            out = op(cat)
            cat = torch.cat([cat, out], 1)

        out = self.path[-1](cat)  # Final operation - No Concatenation
        out = torch.add(out, residual)  # Residual Learning

        return out
