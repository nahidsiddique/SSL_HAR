"""
SimCLR for time-series IMU data.
Chen et al., "A Simple Framework for Contrastive Learning", ICML 2020.

Uses NT-Xent loss.  Projection head discarded after pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NTXentLoss(nn.Module):
    """
    NT-Xent loss (Normalized Temperature-Scaled Cross-Entropy).
    Expects z1, z2 of shape (B, D) — already L2-normalised or not.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.T = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B   = z1.shape[0]
        z   = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2B, D)
        sim = torch.mm(z, z.T) / self.T                         # (2B, 2B)

        # mask diagonal (self-similarity)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # positive for z_i is z_{i+B} and vice-versa
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device),
        ])
        return F.cross_entropy(sim, labels)


class SimCLR(nn.Module):
    """
    SimCLR wrapper around any ResNet1D backbone.

    During pretraining:  forward(x1, x2) → loss
    During eval:         encode(x)        → feature vector
    """
    def __init__(
        self,
        backbone:   nn.Module,
        proj_dim:   int   = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.backbone  = backbone
        self.projector = ProjectionHead(
            in_dim     = backbone.out_dim,
            hidden_dim = backbone.out_dim,
            out_dim    = proj_dim,
        )
        self.criterion = NTXentLoss(temperature)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return self.criterion(z1, z2)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
