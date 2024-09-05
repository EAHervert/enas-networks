import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self,
                 channels=3,
                 kernel_size=3,
                 padding=1,
                 features=64,
                 num_of_layers=17):
        super(DnCNN, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.features = features
        layers = []

        # Pre-Processing: Input Image -> H_1
        layers.append(nn.Conv2d(in_channels=self.channels,
                                out_channels=self.features,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Hidden Layers: H_1 -> ... -> H_n
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=self.features,
                          out_channels=self.features,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(self.features))
            layers.append(nn.ReLU(inplace=True))

        # Post-Processing: H_n -> Output Image
        layers.append(nn.Conv2d(in_channels=self.features,
                                out_channels=self.channels,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)
