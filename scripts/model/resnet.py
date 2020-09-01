# -*- coding: utf-8 -*-
import torch.nn as nn


class ResNetConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding,
                 norm=nn.BatchNorm2d, no_relu=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              **kwargs)
        self.norm = norm(out_ch)
        if not no_relu:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.norm(self.conv(x))
        if hasattr(self, 'relu'):
            x = self.relu(x)
        return x


class BottleneckConv(nn.Module):
    def __init__(self, in_ch, norm, padding_mode, bias, dropout):
        super().__init__()

        self.conv1 = ResNetConv(in_ch, in_ch, 3, 1, 1, norm=norm,
                                padding_mode=padding_mode, bias=bias)
        if dropout:
            self.dropout = nn.Dropout(0.5)
        self.conv2 = ResNetConv(in_ch, in_ch, 3, 1, 1, norm=norm, no_relu=True,
                                padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        h = self.conv1(x)
        if hasattr(self, 'dropout'):
            h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm, bias):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, 1, bias=bias)
        self.norm = norm(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        return self.norm(self.relu(x))


class ResNet(nn.Module):
    def __init__(self, in_ch, out_ch,
                 feature=64, depth=2, n_blocks=9,
                 padding_mode='reflect', norm_mode='batch',
                 bias=False, dropout=False):
        super().__init__()

        if norm_mode == 'batch':
            norm = lambda x: nn.BatchNorm2d(x, affine=True,
                                            track_running_stats=True)
        elif norm_mode == 'instance':
            norm = lambda x: nn.InstanceNorm2d(x, affine=False,
                                               track_running_stats=False)
        else:
            raise KeyError(
                'Normalization type must be one of ["batch", "instance"]')

        cin, cout = in_ch, feature
        down_layers = [ResNetConv(cin, cout, 7, 1, 3, norm=norm,
                                  bias=bias, padding_mode='reflect')]
        for i in range(depth):
            cin, cout = cout, cout * 2
            conv = ResNetConv(cin, cout, 3, 2, 1, norm=norm, bias=bias)
            down_layers.append(conv)
        self.downsample = nn.Sequential(*down_layers)

        bottleneck = list()
        for _ in range(n_blocks):
            conv = BottleneckConv(cout, norm, padding_mode, bias, dropout)
            bottleneck.append(conv)
        self.bottleneck = nn.Sequential(*bottleneck)

        up_layers = list()
        for i in range(depth):
            cin, cout = cout, cout // 2
            up_layers.append(UpsampleConv(cin, cout, norm, bias))
        self.upsample = nn.Sequential(*up_layers)

        self.pred = nn.Sequential(
            nn.Conv2d(cout, out_ch, 7, 1, 3, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.bottleneck(x)
        x = self.upsample(x)
        return self.pred(x)
