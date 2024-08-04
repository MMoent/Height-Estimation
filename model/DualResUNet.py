import math
import torch
import torch.nn as nn
from .Components import *
    

class DualResUNet(nn.Module):
    def __init__(self, in_channels_aerial, in_channels_pano, out_channels):
        super().__init__()
        self.aerial_encoder = Encoder(in_channels_aerial, use_skip=True)
        self.pano_encoder = Encoder(in_channels_pano)
        self.fusion = MultiheadAttention(1024, 1024, 64)
        self.decoder = Decoder()
        self.head = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, pano):
        feats = [None,]
        feats.extend(self.aerial_encoder(x))
        feats = feats[::-1]
        pano_feat = self.pano_encoder(pano)
        feats[0] = self.fusion(feats[0], pano_feat).permute(0, 2, 1).view(-1, 1024, 8, 8)
        y = self.decoder(feats)
        return y
    
    
if __name__ == "__main__":
    x = torch.randn(4, 3, 256, 256)
    pano = torch.randn(4, 3, 256, 512)
    model = DualResUNet(3, 3, 64)
    y = model(x, pano)
    print(y.shape)
    print(count_model_param(model))
