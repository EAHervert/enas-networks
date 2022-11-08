import torch.nn as nn


# Have a cell that contains the downsampling paths and a forward which can choose accordingly.
# This will be part of the searching for the ideal architecture.

# This will be the full branch.
class _down_ENAS(nn.Module):
    def __init__(self, channel_in):
        super(_down_ENAS, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.downconv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=2 * channel_in,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=2 * channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.PReLU()

    def forward(self, x, int_):
        if int_ == 0:
            out = self.relu(self.conv(self.maxpool(x)))
        elif int_ == 1:
            out = self.relu(self.conv(self.avgpool(x)))
        elif int_ == 2:
            out = self.relu(self.downconv(x))

        return out


# This will be the branch that is fixed depending on the architecture.
class _down_Fixed(nn.Module):
    def __init__(self, channel_in, architecture_k):
        super(_down_Fixed, self).__init__()

        self.architecture_k = architecture_k

        if self.architecture_k == 0:
            self.down = nn.MaxPool2d(2)
        elif self.architecture_k == 1:
            self.down = nn.AvgPool2d(2)
        elif self.architecture_k == 2:
            self.down = nn.Conv2d(
                in_channels=channel_in,
                out_channels=2 * channel_in,
                kernel_size=2,
                stride=2,
                padding=0
            )

        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=2 * channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.relu = nn.PReLU()

    def forward(self, x, int_):
        out = self.down(x)
        if self.architecture_k != 2:
            out = self.relu(self.conv(out))
        else:
            out = self.relu(out)

        return out


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
        out = self.maxpool(x)
        out = self.relu(self.conv(out))

        return out


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
        out = self.pool(x)
        out = self.relu(self.conv(out))

        return out


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
        out = self.relu(self.downconv(x))

        return out


# Have a cell that contains the upsampling paths and a forward which can choose accordingly.
# This will be part of the searching for the ideal architecture.

# This will be the full branch.
class _up_ENAS(nn.Module):
    def __init__(self, channel_in):
        super(_up_ENAS, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.PReLU()
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
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x, int_):
        out = self.relu(self.conv(x))

        if int_ == 0:
            out = self.PS(out)
        elif int_ == 1:
            out = self.convT(out)
        elif int_ == 2:
            out = self.BL(out)

        return out


# This will be the branch that is fixed depending on the architecture.
class _up_Fixed(nn.Module):
    def __init__(self, channel_in, architecture_k):
        super(_up_Fixed, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=channel_in,
            out_channels=channel_in,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.PReLU()

        self.architecture_k = architecture_k

        if self.architecture_k == 0:
            self.up = nn.PixelShuffle(2)
        elif self.architecture_k == 1:
            self.up = nn.ConvTranspose2d(
                in_channels=channel_in,
                out_channels=channel_in // 4,
                kernel_size=2,
                stride=2,
                padding=0
            )
        elif self.architecture_k == 2:
            self.up = nn.Sequential(
                nn.Conv2d(
                    in_channels=channel_in,
                    out_channels=channel_in // 4,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )

    def forward(self, x, int_):
        out = self.relu(self.conv(x))
        out = self.up(out)

        return out


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
        out = self.relu(self.conv(x))
        out = self.PS(out)

        return out


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
        out = self.relu(self.conv(x))
        out = self.convT(out)

        return out


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
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.BL(out)

        return out
