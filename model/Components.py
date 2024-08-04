import math
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=(1,3,1), stride=(1,1,1), padding=(0,1,0), down_kernel_size=1, down_stride=1, use_res=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size[2], stride=stride[2], padding=padding[0])
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_res = use_res
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=down_kernel_size, stride=down_stride),
            nn.BatchNorm2d(out_channels)
        ) if self.use_res else None

    def forward(self, x):
        origin = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x + self.downsample(origin) if self.use_res else x


class Encoder(nn.Module):
    def __init__(self, in_channels, encoder_channels=(64,128,256,512,1024), use_skip=False) -> None:
        super().__init__()
        self.use_skip = use_skip
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)

        self.layer1 = nn.Sequential(
            Bottleneck(encoder_channels[0], encoder_channels[0], encoder_channels[1], stride=(1,1,1), use_res=True, down_kernel_size=1, down_stride=1),
            Bottleneck(encoder_channels[1], encoder_channels[0], encoder_channels[1]),
            Bottleneck(encoder_channels[1], encoder_channels[0], encoder_channels[1]),
        )
        self.layer2 = nn.Sequential(
            Bottleneck(encoder_channels[1], encoder_channels[0], encoder_channels[2], stride=(1,2,1), use_res=True, down_kernel_size=1, down_stride=2),
            Bottleneck(encoder_channels[2], encoder_channels[0], encoder_channels[2]),
            Bottleneck(encoder_channels[2], encoder_channels[0], encoder_channels[2]),
            Bottleneck(encoder_channels[2], encoder_channels[0], encoder_channels[2])
        )
        self.layer3 = nn.Sequential(
            Bottleneck(encoder_channels[2], encoder_channels[1], encoder_channels[3], stride=(1,2,1), use_res=True, down_kernel_size=1, down_stride=2),
            Bottleneck(encoder_channels[3], encoder_channels[1], encoder_channels[3]),
            Bottleneck(encoder_channels[3], encoder_channels[1], encoder_channels[3]),
            Bottleneck(encoder_channels[3], encoder_channels[1], encoder_channels[3]),
            Bottleneck(encoder_channels[3], encoder_channels[1], encoder_channels[3]),
            Bottleneck(encoder_channels[3], encoder_channels[1], encoder_channels[3]),
        )
        self.layer4 = nn.Sequential(
            Bottleneck(encoder_channels[3], encoder_channels[2], encoder_channels[4], stride=(1,2,1), use_res=True, down_kernel_size=1, down_stride=2),
            Bottleneck(encoder_channels[4], encoder_channels[2], encoder_channels[4]),
            Bottleneck(encoder_channels[4], encoder_channels[2], encoder_channels[4]),
        )

    def forward(self, x):
        x = self.layer0(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x, x1, x2, x3, x4] if self.use_skip else x4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels+skip_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)
            

class Decoder(nn.Module):
    def __init__(self, decoder_channels=(1024, 512, 256, 128, 64)) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(decoder_channels[0], 512, 512),
            DecoderBlock(decoder_channels[1], 256, 256),
            DecoderBlock(decoder_channels[2], 128, 128),
            DecoderBlock(decoder_channels[3], 64, 64),
            DecoderBlock(decoder_channels[4], 0, 64)
        ])
    
    def forward(self, feats):
        x = feats[0]
        for i, block in enumerate(self.blocks):
            # print(x.shape, feats[i+1].shape, end=' -> ')
            x = block(x, feats[i+1])
            # print(x.shape)
        return x
    

class CrossAttentionHead(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim, p=0.1) -> None:
        super().__init__()
        self.in_dim1, self.in_dim2, self.out_dim = in_dim1, in_dim2, out_dim
        self.w_q = nn.Linear(in_dim1, out_dim)
        self.w_k = nn.Linear(in_dim2, out_dim)
        self.w_v = nn.Linear(in_dim2, out_dim)
        self.dropout = nn.Dropout(p)
    def forward(self, x, xt):
        Q = self.w_q(x.flatten(2).transpose(1,2))
        K = self.w_k(xt.flatten(2).transpose(1,2))
        V = self.w_v(xt.flatten(2).transpose(1,2))
        attention = nn.functional.softmax(Q @ K.transpose(1,2) / math.sqrt(self.out_dim), dim=-1)
        attention = self.dropout(attention)
        output = attention @ V
        return output


class MultiheadAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim_head, p=0.1) -> None:
        super().__init__()
        self.in_dim1, self.in_dim2, self.out_dim_head = in_dim1, in_dim2, out_dim_head
        self.head_num = in_dim1 // out_dim_head
        self.total_out_dim = self.head_num * self.out_dim_head
        self.heads = nn.ModuleList()
        [self.heads.append(CrossAttentionHead(self.in_dim1, self.in_dim2, self.out_dim_head)) for _ in range(self.head_num)]
        self.proj = nn.Linear(self.total_out_dim, self.total_out_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, x, xt):
        x = torch.cat([head(x, xt) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


def count_model_param(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
