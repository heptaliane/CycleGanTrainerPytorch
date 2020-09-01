# -*- coding: utf-8 -*-
import torch.nn as nn


class UNetConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(self.bn(x))


class UNetLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = UNetConv(in_ch, out_ch)
        self.conv2 = UNetConv(out_ch, out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetEncodeLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = UNetLayer(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)


class UNetDecodeLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = UNetLayer(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, depth=4, feature=32):
        super().__init__()

        encoders = [UNetEncodeLayer(in_ch, feature)]
        for _ in range(1, depth):
            encoders.append(UNetEncodeLayer(feature, feature * 2))
            feature *= 2
        self.encoder = nn.Sequential(*encoders)

        self.bottleneck = UNetLayer(feature, feature * 2)

        decoders = list()
        for _ in range(depth):
            decoders.append(UNetDecodeLayer(feature * 2, feature))
            feature = feature // 2
        self.decoder = nn.Sequential(*decoders)

        self.pred = nn.Sequential(
            nn.Conv2d(feature * 2, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ys = list()
        for encoder in self.encoder:
            x = encoder(x)
            ys.append(x)

        x = self.bottleneck(x)

        for decoder, y in zip(self.decoder, ys[::-1]):
            x = decoder(x, y)

        return self.pred(x)
