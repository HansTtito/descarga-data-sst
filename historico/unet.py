from __future__ import annotations

import torch
from torch import nn


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class C17UNet3D(nn.Module):
    def __init__(self, context: int, horizon: int, base_channels: int = 16):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.encoder1 = DoubleConv3d(1, c1)
        self.encoder2 = DoubleConv3d(c1, c2)
        self.bottleneck = DoubleConv3d(c2, c3)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder2 = DoubleConv3d(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder1 = DoubleConv3d(c1 + c1, c1)
        self.output = nn.Conv3d(c1, 1, kernel_size=1)
        self.temporal_projection = nn.Linear(int(context), int(horizon))

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.decoder2(torch.cat((self.up2(b), e2), dim=1))
        d1 = self.decoder1(torch.cat((self.up1(d2), e1), dim=1))
        out = self.output(d1)
        projected = self.temporal_projection(out.permute(0, 1, 3, 4, 2))
        return projected.permute(0, 1, 4, 2, 3) + x[:, :1, -1:, :, :]


def masked_mse(prediction, target, mask):
    weights = mask.to(dtype=prediction.dtype)
    denominator = weights.sum()
    if denominator.item() == 0:
        raise ValueError("batch sin objetivos validos")
    return (((prediction - target) ** 2) * weights).sum() / denominator
