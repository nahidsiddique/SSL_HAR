"""SoftCLT model used in this project."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_ntxent_instance(
    z1: torch.Tensor,
    z2: torch.Tensor,
    x_raw: torch.Tensor,
    rho: float = 1.0,
    temperature: float = 0.1,
) -> torch.Tensor:
    B = z1.shape[0]
    if B < 2:
        return torch.zeros((), device=z1.device, dtype=z1.dtype)

    rho = max(float(rho), 1e-4)
    temperature = max(float(temperature), 1e-4)

    x_flat = x_raw.reshape(B, -1).float()
    x_flat = F.normalize(x_flat, dim=1, eps=1e-6)
    dist2 = torch.cdist(x_flat, x_flat, p=2).pow(2).clamp_(min=0.0, max=1e4)
    weights = torch.exp(torch.clamp(-dist2 / (2.0 * rho * rho), min=-50.0, max=0.0))

    z1 = F.normalize(z1.float(), dim=1, eps=1e-6)
    z2 = F.normalize(z2.float(), dim=1, eps=1e-6)
    z = torch.cat([z1, z2], dim=0)

    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)

    soft_targets = torch.zeros(2 * B, 2 * B, device=z.device, dtype=sim.dtype)
    soft_targets[:B, B:] = weights
    soft_targets[B:, :B] = weights
    soft_targets = soft_targets.masked_fill(mask, 0.0)
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)

    log_probs = F.log_softmax(sim, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss.mean()


def soft_ntxent_temporal(
    z_seq: torch.Tensor,
    rho: float = 10.0,
    temperature: float = 0.1,
) -> torch.Tensor:
    B, _, T = z_seq.shape
    if T < 2:
        return torch.zeros((), device=z_seq.device, dtype=z_seq.dtype)

    rho = max(float(rho), 1e-4)
    temperature = max(float(temperature), 1e-4)

    z_seq = F.normalize(z_seq.float(), dim=1, eps=1e-6)
    z_mean = z_seq.mean(dim=0).T

    sim = torch.mm(z_mean, z_mean.T) / temperature
    mask_diag = torch.eye(T, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask_diag, -1e9)

    t_idx = torch.arange(T, device=sim.device).float()
    dist_mat = (t_idx[:, None] - t_idx[None, :]).abs()
    weights = torch.exp(torch.clamp(-dist_mat / rho, min=-50.0, max=0.0))
    weights = weights.masked_fill(mask_diag, 0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    log_probs = F.log_softmax(sim, dim=1)
    loss = -(weights * log_probs).sum(dim=1)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss.mean()


class SoftCLT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 128,
        rho_instance: float = 1.0,
        rho_temporal: float = 10.0,
        lambda_temporal: float = 0.5,
        temp_instance: float = 0.10,
        temp_temporal: float = 0.10,
    ):
        super().__init__()
        self.backbone = backbone
        self.lambda_temporal = lambda_temporal
        self.rho_instance = rho_instance
        self.rho_temporal = rho_temporal
        self.temp_instance = temp_instance
        self.temp_temporal = temp_temporal

        in_dim = getattr(backbone, "out_channels", backbone.out_dim)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )
        self.out_dim = getattr(backbone, "out_dim", in_dim)

    def forward(self, x: torch.Tensor, x_aug: torch.Tensor) -> torch.Tensor:
        seq = self.backbone(x)
        seq_aug = self.backbone(x_aug)

        h = seq.mean(dim=-1)
        h_aug = seq_aug.mean(dim=-1)

        z = self.projector(h)
        z_aug = self.projector(h_aug)

        l_instance = soft_ntxent_instance(
            z,
            z_aug,
            x,
            self.rho_instance,
            self.temp_instance,
        )
        l_temporal = soft_ntxent_temporal(
            seq,
            self.rho_temporal,
            self.temp_temporal,
        )
        loss = l_instance + self.lambda_temporal * l_temporal
        return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)
