import torch
from torch import nn


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=bias, stride=stride),
            nn.BatchNorm2d(num_features=n_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DnCNN(nn.Module):
    def __init__(self, depth=10, n_channels=64, image_channels=1, kernel_size=3, stride=1,
                 training=True):
        super().__init__()
        layers = []

        padding = kernel_size // 2
        self.training = training

        layers.append(nn.Conv2d(
            in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
            bias=False, stride=stride))
        layers.append(nn.ReLU())

        for _ in range(depth - 1):
            layers.append(ResBlock(n_channels, kernel_size, stride, padding))

        layers.append(nn.Conv2d(
            in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
            bias=False, stride=stride))

        self.learned_resi = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.learned_resi(x)
