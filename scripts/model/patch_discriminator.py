# -*- coding: utf-8 -*-
import torch.nn as nn


class PatchDiscriminatorConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride, norm=None):
        super().__init__()
        use_bias = norm is None or norm == nn.InstanceNorm2d
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=use_bias)
        if norm is not None:
            self.norm = norm(out_ch)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        return self.relu(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch, feature=64, depth=3, norm=nn.InstanceNorm2d):
        super().__init__()

        out_ch = feature
        layer = [PatchDiscriminatorConv(in_ch, out_ch, 2)]
        for _ in range(1, depth):
            in_ch, out_ch = out_ch, min(out_ch * 2, feature * 8)
            layer.append(PatchDiscriminatorConv(in_ch, out_ch, 2, norm))

        in_ch, out_ch = out_ch, min(out_ch * 2, feature * 8)
        layer.append(PatchDiscriminatorConv(in_ch, out_ch, 1, norm))
        self.layers = nn.Sequential(*layer)
        self.pred = nn.Sequential(
            nn.Conv2d(out_ch, 1, 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        return self.pred(x)
