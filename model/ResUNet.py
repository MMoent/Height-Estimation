import math
import torch
import torch.nn as nn
from .Components import *


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, use_skip=True)
        self.decoder = Decoder()
        self.head = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        feats = [None,]
        feats.extend(self.encoder(x))
        feats = feats[::-1]
        y = self.decoder(feats)
        return y


if __name__ == "__main__":
    x = torch.randn(4, 3, 256, 256)
    resunet = ResUNet(3, 64)
    y = resunet(x)
    print(y.shape)
    print(count_model_param(resunet))