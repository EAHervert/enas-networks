import torch
import torch.nn as nn


# Have a cell that contains the downsampling paths and a forward which can choose accordingly.
# This will be part of the searching for the ideal architecture.

# This will be the full branch.
class _down_DNAS(nn.Module):
    def __init__(self,
                 alphas_down,
                 channel_in):
        super(_down_DNAS, self).__init__()
        self.alphas_down = alphas_down
        self.maxpool = nn.Sequential(nn.MaxPool2d(2),
                                     nn.Conv2d(
                                         in_channels=channel_in,
                                         out_channels=2 * channel_in,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0))
        self.avgpool = nn.Sequential(nn.AvgPool2d(2),
                                     nn.Conv2d(
                                         in_channels=channel_in,
                                         out_channels=2 * channel_in,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0))
        self.downconv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=2 * channel_in,
            kernel_size=2,
            stride=2,
            padding=0
        )

        self.downsamples = nn.ModuleList([self.maxpool, self.avgpool, self.downconv])
        self.relu = nn.PReLU()

    def forward(self, x):
        out = 0
        for index, alpha in enumerate(self.alphas_down):
            out += alpha * self.downsamples[index](x)

        return self.relu(out)


# This will be the branch that is fixed depending on the architecture.
class _down_Fixed(nn.Module):
    def __init__(self, alphas_down, channel_in):
        super(_down_Fixed, self).__init__()

        self.down_op = torch.argmax(alphas_down).item()

        if self.down_op == 2:
            self.down = nn.Conv2d(
                in_channels=channel_in,
                out_channels=2 * channel_in,
                kernel_size=2,
                stride=2,
                padding=0
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=channel_in,
                out_channels=2 * channel_in,
                kernel_size=1,
                stride=1,
                padding=0
            )
            if self.down_op == 0:
                self.down = nn.MaxPool2d(2)
            elif self.down_op == 1:
                self.down = nn.AvgPool2d(2)

        self.relu = nn.PReLU()

    def forward(self, x):
        if self.down_op != 2:
            return self.relu(self.conv(self.down(x)))
        else:
            return self.relu(self.down(x))


# Using Max Pooling to downsample the images.
class _down_Max(nn.Module):
    def __init__(self, channel_in):
        super(_down_Max, self).__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=2 * channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.conv(self.pool(x)))


# Using Average Pooling to downsample the images.
class _down_Avg(nn.Module):
    def __init__(self, channel_in):
        super(_down_Avg, self).__init__()

        self.pool = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=2 * channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.conv(self.pool(x)))


# Using Convolution to downsample the images.
class _down_Conv(nn.Module):
    def __init__(self, channel_in):
        super(_down_Conv, self).__init__()

        self.downconv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=2 * channel_in,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.downconv(x))


# Have a cell that contains the upsampling paths and a forward which can choose accordingly.
# This will be part of the searching for the ideal architecture.

# This will be the full branch.
class _up_DNAS(nn.Module):
    def __init__(self, alphas_up, channel_in):
        super(_up_DNAS, self).__init__()

        self.alphas_up = alphas_up
        self.preprocess = nn.Sequential(nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        ),
            nn.PReLU()
        )
        self.PS = nn.PixelShuffle(2)
        self.convT = nn.ConvTranspose2d(
            in_channels=channel_in,
            out_channels=channel_in // 4,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.BL = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_in,
                out_channels=channel_in // 4,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.upsamples = nn.ModuleList([self.PS, self.convT, self.BL])

    def forward(self, x):
        out = self.preprocess(x)
        out_final = 0
        for index, alpha in enumerate(self.alphas_up):
            out_final += alpha * self.upsamples[index](out)

        return out_final


# This will be the branch that is fixed depending on the architecture.
class _up_Fixed(nn.Module):
    def __init__(self, alphas_up, channel_in):
        super(_up_Fixed, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.PReLU()

        self.up_op = torch.argmax(alphas_up).item()

        if self.up_op == 0:
            self.up = nn.PixelShuffle(2)
        elif self.up_op == 1:
            self.up = nn.ConvTranspose2d(
                in_channels=channel_in,
                out_channels=channel_in // 4,
                kernel_size=2,
                stride=2,
                padding=0
            )
        elif self.up_op == 2:
            self.up = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel_in,
                    out_channels=channel_in // 4,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

    def forward(self, x):
        return self.up(self.relu(self.conv(x)))


# Using Pixel Shuffling to upsample the images.
class _up_PS(nn.Module):
    def __init__(self, channel_in):
        super(_up_PS, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.PReLU()
        self.PS = nn.PixelShuffle(2)

    def forward(self, x):
        return self.PS(self.relu(self.conv(x)))


# Using Transpose Convolution to upsample the images.
class _up_ConvT(nn.Module):
    def __init__(self, channel_in):
        super(_up_ConvT, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.PReLU()
        self.convT = nn.ConvTranspose2d(
            in_channels=channel_in,
            out_channels=channel_in // 4,
            kernel_size=2,
            stride=2,
            padding=0
        )

    def forward(self, x):
        return self.convT(self.relu(self.conv(x)))


# Using Bilinear Interpolation to downsample the images.
class _up_BL(nn.Module):
    def __init__(self, channel_in):
        super(_up_BL, self).__init__()

        nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.PReLU()
        self.BL = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_in,
                out_channels=channel_in // 4,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.BL(self.relu(self.conv(x)))
