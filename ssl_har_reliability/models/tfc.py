"""
TF-C: Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency
Zhang et al., NeurIPS 2022.

Three losses:
  L_time : NT-Xent within time-domain branch
  L_freq : NT-Xent within frequency-domain branch
  L_cross: cross-space consistency (time ↔ freq embeddings of same sample)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Transformer encoder block ───────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Lightweight Transformer encoder for 1D sequences.
    Input  : (B, C, T) — channels-first
    Output : (B, d_model) — global average pooled
    """
    def __init__(
        self,
        in_channels: int,
        d_model:     int = 64,
        n_heads:     int = 4,
        n_layers:    int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C) → project → transformer → pool
        x = x.permute(0, 2, 1)              # (B, T, C)
        x = self.input_proj(x)              # (B, T, d_model)
        x = self.transformer(x)             # (B, T, d_model)
        return x.mean(dim=1)                # (B, d_model)


# ─── NT-Xent loss ────────────────────────────────────────────────────────────

def ntxent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    B   = z1.shape[0]
    z   = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ─── Cross-space consistency loss ────────────────────────────────────────────

def cross_space_loss(
    z_time: torch.Tensor,
    z_freq: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Pull time-domain and freq-domain embeddings of same sample together.
    Positive pair: (z_time_i, z_freq_i).  All cross-sample pairs = negatives.
    """
    B = z_time.shape[0]
    zt = F.normalize(z_time, dim=1)   # (B, D)
    zf = F.normalize(z_freq, dim=1)   # (B, D)

    # similarity between all time-freq pairs
    sim    = torch.mm(zt, zf.T) / temperature   # (B, B)
    labels = torch.arange(B, device=sim.device)

    # symmetric: time→freq and freq→time
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


# ─── TF-C model ──────────────────────────────────────────────────────────────

class TFC(nn.Module):
    """
    Time-Frequency Consistency pretraining model.

    encode(x) → (B, 2*d_model)  — concat of time + freq embeddings.
    forward(x, x_aug) → scalar loss.
    """
    def __init__(
        self,
        in_channels: int   = 6,
        seq_len:     int   = 128,
        d_model:     int   = 64,
        n_heads:     int   = 4,
        n_layers:    int   = 2,
        proj_dim:    int   = 64,
        alpha:       float = 1.0,   # weight for L_time
        beta:        float = 1.0,   # weight for L_freq
        gamma:       float = 1.0,   # weight for L_cross
        temperature: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.temp  = temperature

        freq_len = seq_len // 2 + 1   # rfft output length

        self.time_encoder = TransformerEncoder(in_channels, d_model, n_heads, n_layers)
        self.freq_encoder = TransformerEncoder(in_channels, d_model, n_heads, n_layers)

        # projectors mapping into shared joint latent space
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(d_model, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        self.out_dim = d_model * 2   # concat time + freq for downstream

    @staticmethod
    def _to_freq(x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, C, T//2+1) magnitude spectrum."""
        return torch.abs(torch.fft.rfft(x, dim=-1))

    def _encode_both(self, x: torch.Tensor):
        x_freq = self._to_freq(x)
        h_t    = self.time_encoder(x)        # (B, d_model)
        h_f    = self.freq_encoder(x_freq)   # (B, d_model)
        return h_t, h_f

    def forward(self, x: torch.Tensor, x_aug: torch.Tensor) -> torch.Tensor:
        """
        x     : original view    (B, 6, 128)
        x_aug : augmented view   (B, 6, 128)
        """
        h_t,     h_f     = self._encode_both(x)
        h_t_aug, h_f_aug = self._encode_both(x_aug)

        # within-domain projections
        z_t     = self.time_proj(h_t)
        z_t_aug = self.time_proj(h_t_aug)
        z_f     = self.freq_proj(h_f)
        z_f_aug = self.freq_proj(h_f_aug)

        l_time  = ntxent(z_t, z_t_aug, self.temp)
        l_freq  = ntxent(z_f, z_f_aug, self.temp)

        # cross-space: time ↔ freq of same sample (both views)
        l_cross = (cross_space_loss(z_t, z_f, self.temp) +
                   cross_space_loss(z_t_aug, z_f_aug, self.temp)) / 2

        return self.alpha * l_time + self.beta * l_freq + self.gamma * l_cross

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h_t, h_f = self._encode_both(x)
        return torch.cat([h_t, h_f], dim=1)   # (B, 2*d_model)
