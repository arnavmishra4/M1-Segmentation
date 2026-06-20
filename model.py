"""
model.py
--------
3D Residual U-Net with Attention Gates.

Mirrors the architecture used during training exactly so that checkpoints
load without key mismatches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants (must match training) ──────────────────────────────────────────
IN_CHANNELS  = 4
NUM_CLASSES  = 4
BASE_FILTERS = 24


# ── Helper ────────────────────────────────────────────────────────────────────
def group_norm(num_channels: int) -> nn.GroupNorm:
    """Returns the largest valid GroupNorm for the given channel count."""
    for g in [32, 16, 8, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


# ── Building blocks ───────────────────────────────────────────────────────────
class ConvBnRelu(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__(
            nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            group_norm(out_c),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBnRelu(in_c, out_c, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            group_norm(out_c),
        )
        self.relu = nn.ReLU(inplace=True)
        self.skip = (
            nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False),
                group_norm(out_c),
            )
            if (in_c != out_c or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.skip(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block1 = ResBlock(in_c, out_c)
        self.block2 = ResBlock(out_c, out_c)
        self.pool   = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor):
        x    = self.block1(x)
        skip = self.block2(x)
        return self.pool(skip), skip


class AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_x: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, 1, bias=False),
            group_norm(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(f_x, f_int, 1, bias=False),
            group_norm(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, 1, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(
            self.W_g(g), size=x.shape[2:], mode="trilinear", align_corners=False
        )
        att = self.relu(g_up + self.W_x(x))
        att = self.psi(att)
        return x * att


class DecoderBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up     = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2)
        self.att    = AttentionGate(f_g=out_c, f_x=skip_c, f_int=max(skip_c // 2, 8))
        self.block1 = ResBlock(out_c + skip_c, out_c)
        self.block2 = ResBlock(out_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        x    = F.interpolate(
            x, size=skip.shape[2:], mode="trilinear", align_corners=False
        )
        skip = self.att(g=x, x=skip)
        x    = self.block1(torch.cat([x, skip], dim=1))
        return self.block2(x)


# ── Full model ────────────────────────────────────────────────────────────────
class ResUNet3D(nn.Module):
    """
    3-D Residual U-Net with Attention Gates.

    At inference time (model.eval()) the forward pass returns only the
    full-resolution logit map (B, C, D, H, W).

    At training time it additionally returns three deep-supervision heads
    (aux4, aux3, aux2) upsampled to the same spatial size as the main output.
    """

    def __init__(
        self,
        in_channels:  int = IN_CHANNELS,
        num_classes:  int = NUM_CLASSES,
        base_filters: int = BASE_FILTERS,
    ):
        super().__init__()
        f = base_filters
        self.enc1   = EncoderBlock(in_channels, f)
        self.enc2   = EncoderBlock(f,    f * 2)
        self.enc3   = EncoderBlock(f * 2, f * 4)
        self.enc4   = EncoderBlock(f * 4, f * 8)
        self.bridge = nn.Sequential(
            ResBlock(f * 8,  f * 16),
            ResBlock(f * 16, f * 16),
            ResBlock(f * 16, f * 16),
        )
        self.dec4 = DecoderBlock(f * 16, f * 8,  f * 8)
        self.dec3 = DecoderBlock(f * 8,  f * 4,  f * 4)
        self.dec2 = DecoderBlock(f * 4,  f * 2,  f * 2)
        self.dec1 = DecoderBlock(f * 2,  f,       f)
        self.head = nn.Conv3d(f, num_classes, kernel_size=1)

        # Deep-supervision heads (used only during training)
        self.ds4 = nn.Conv3d(f * 8,  num_classes, kernel_size=1)
        self.ds3 = nn.Conv3d(f * 4,  num_classes, kernel_size=1)
        self.ds2 = nn.Conv3d(f * 2,  num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x  = self.bridge(x)
        d4 = self.dec4(x,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        out = self.head(d1)

        if self.training:
            size = out.shape[2:]
            aux4 = F.interpolate(self.ds4(d4), size=size,
                                 mode="trilinear", align_corners=False)
            aux3 = F.interpolate(self.ds3(d3), size=size,
                                 mode="trilinear", align_corners=False)
            aux2 = F.interpolate(self.ds2(d2), size=size,
                                 mode="trilinear", align_corners=False)
            return out, aux4, aux3, aux2

        return out