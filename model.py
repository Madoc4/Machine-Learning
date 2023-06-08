import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, channels_img, channels):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input Shape: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, channels, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self.block(channels, channels*2, 4, 2, 1),      # 16 x 16
            self.block(channels*2, channels*4, 4, 2, 1),    # 8 x 8
            self.block(channels*4, channels*8, 4, 2, 1),    # 4 x 4
            nn.Conv2d(channels*8, 1, kernel_size=4, stride=2, padding=0), # 1 x 1
            nn.Sigmoid()
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, noise, channels_img, channels):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input Shape: N x noise x 1 x 1
            self._block(noise, channels*16, 4, 1, 0),    # 4 x 4
            self._block(channels*16, channels*8, 4, 2, 1),    # 8 x 8
            self._block(channels*8, channels*4, 4, 2, 1),     # 16 x 16
            self._block(channels*4, channels*2, 4, 2, 1),     # 32 x 32
            nn.ConvTranspose2d(
                channels*2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # 64 x 64
            nn.Tanh()   # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)


# Initialize Weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# Test Discriminator and Generator
def test():
    N, in_channels, height, weight = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, height, weight))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator Test Failed"
    gen = Generator(noise_dim, in_channels, 8)
    initialize_weights(gen)
    noise = torch.randn((N, noise_dim, 1, 1))
    assert gen(noise).shape == (N, in_channels, height, weight), "Generator Test Failed"
    print("All Tests Passed")


test()
