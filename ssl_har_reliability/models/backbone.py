"""
Shared backbone architectures for all SSL methods.

ResNet1D: residual encoder that outputs one feature vector
CNN1D: plain convolutional encoder that keeps the time dimension

Both accept input shape (B, C, T) = (B, 6, 128).
"""

import torch
import torch.nn as nn


# ─── ResNet1D ─────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                               padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.ReLU(inplace=True)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class ResNet1D(nn.Module):
    """
    Small 1D ResNet backbone.
    Input: (B, 6, 128)
    Output: (B, out_dim)
    """
    def __init__(self, in_channels: int = 6, base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, f, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock1D(f,     f,     stride=1)
        self.layer2 = ResBlock1D(f,     f * 2, stride=2)
        self.layer3 = ResBlock1D(f * 2, f * 4, stride=2)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.out_dim = f * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.gap(x).squeeze(-1)   # (B, out_dim)


# ─── CNN1D (sequence-preserving, for TS-TCC) ──────────────────────────────────

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 8):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """
    Simple stacked convolutional encoder that keeps the time dimension.
    Input: (B, 6, 128)
    Output: (B, out_channels, T)
    """
    def __init__(self, in_channels: int = 6, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock1D(in_channels,   32,  kernel=8),
            ConvBlock1D(32,            64,  kernel=5),
            ConvBlock1D(64,  out_channels,  kernel=3),
        )
        self.out_channels = out_channels
        self.out_dim      = out_channels   # for compatibility with ResNet1D API

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, out_channels, T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.forward(x)   # (B, out_channels, T)
        return seq.mean(dim=-1)  # (B, out_channels)


# ─── Linear classifier head ───────────────────────────────────────────────────

class LinearHead(nn.Module):
    """Linear head for probing."""
    def __init__(self, in_dim: int, n_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPHead(nn.Module):
    """Two-layer MLP head for fine-tuning."""
    def __init__(self, in_dim: int, hidden_dim: int = 256, n_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
