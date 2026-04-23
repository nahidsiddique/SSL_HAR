"""
TS-TCC: Time-Series Representation Learning via Temporal and Contextual Contrasting
Eldele et al., IJCAI 2021.

Two modules:
  1. Temporal Contrasting  — cross-view future prediction (autoregressive GRU)
  2. Contextual Contrasting — NT-Xent on aggregated context vectors

Input augmentations: WeakStrongTransform (asymmetric)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContrastingModule(nn.Module):
    """
    Autoregressive GRU that:
      - processes a latent sequence from view A → context vectors c_t
      - predicts future latent steps of view B  (cross-view InfoNCE)
    """
    def __init__(
        self,
        input_dim:    int,   # CNN encoder output channels
        context_dim:  int = 256,
        n_steps:      int = 5,      # future steps to predict
        temperature:  float = 0.07,
    ):
        super().__init__()
        self.n_steps     = n_steps
        self.temperature = temperature
        self.context_dim = context_dim

        self.ar  = nn.GRU(input_size=input_dim, hidden_size=context_dim,
                           num_layers=1, batch_first=True)

        # one linear predictor per future step
        self.W_k = nn.ModuleList([
            nn.Linear(context_dim, input_dim) for _ in range(n_steps)
        ])

    def forward(
        self,
        z_weak:   torch.Tensor,   # (B, D, T')  from CNN encoder
        z_strong: torch.Tensor,   # (B, D, T')  from CNN encoder
    ) -> tuple:
        """
        Returns (loss, context_weak, context_strong)
        context_*: (B, context_dim)  — last GRU hidden state
        """
        B, D, T = z_weak.shape

        # (B, T, D) for GRU
        seq_w = z_weak.permute(0, 2, 1)
        seq_s = z_strong.permute(0, 2, 1)

        c_w, _ = self.ar(seq_w)   # (B, T, context_dim)
        c_s, _ = self.ar(seq_s)

        loss = self._cross_view_infoNCE(c_w, seq_s) + \
               self._cross_view_infoNCE(c_s, seq_w)

        # last context vector = global representation
        ctx_w = c_w[:, -1, :]    # (B, context_dim)
        ctx_s = c_s[:, -1, :]

        return loss, ctx_w, ctx_s

    def _cross_view_infoNCE(
        self,
        context: torch.Tensor,   # (B, T, context_dim)
        targets: torch.Tensor,   # (B, T, D)  future latents from other view
    ) -> torch.Tensor:
        """
        For each time t, predict targets at t+1 ... t+K using W_k(c_t).
        Loss = InfoNCE with all other samples in batch as negatives.
        """
        B, T, D_ctx = context.shape
        loss = torch.tensor(0.0, device=context.device)
        count = 0

        for k, W in enumerate(self.W_k):
            t_ctx = T - self.n_steps + k - 1   # context time step
            t_tgt = t_ctx + k + 1               # target time step
            if t_ctx < 0 or t_tgt >= T:
                continue

            pred = W(context[:, t_ctx, :])           # (B, D)
            tgt  = targets[:, t_tgt, :]               # (B, D)

            pred = F.normalize(pred, dim=1)
            tgt  = F.normalize(tgt,  dim=1)

            # similarity matrix (B, B)
            sim    = torch.mm(pred, tgt.T) / self.temperature
            labels = torch.arange(B, device=sim.device)
            loss  += F.cross_entropy(sim, labels)
            count += 1

        return loss / max(count, 1)


class ContextualContrastingModule(nn.Module):
    """NT-Xent on aggregated context vectors."""
    def __init__(self, context_dim: int, proj_dim: int = 128, temperature: float = 0.2):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, proj_dim),
        )
        self.T = temperature

    def forward(self, c_w: torch.Tensor, c_s: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(self.projector(c_w), dim=1)
        z2 = F.normalize(self.projector(c_s), dim=1)

        B   = z1.shape[0]
        z   = torch.cat([z1, z2], dim=0)          # (2B, D)
        sim = torch.mm(z, z.T) / self.T

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device),
        ])
        return F.cross_entropy(sim, labels)


class TSTCC(nn.Module):
    """
    Full TS-TCC model.

    encode(x) → (B, context_dim)  for downstream tasks.
    forward(x_weak, x_strong) → scalar loss.
    """
    def __init__(
        self,
        encoder:        nn.Module,   # CNN1D, output shape (B, out_channels, T)
        context_dim:    int   = 256,
        proj_dim:       int   = 128,
        n_steps:        int   = 5,
        lambda_tc:      float = 1.0,
        lambda_cc:      float = 0.7,
        temp_tc:        float = 0.07,
        temp_cc:        float = 0.20,
    ):
        super().__init__()
        self.encoder     = encoder
        self.lambda_tc   = lambda_tc
        self.lambda_cc   = lambda_cc

        self.tc = TemporalContrastingModule(
            input_dim   = encoder.out_channels,
            context_dim = context_dim,
            n_steps     = n_steps,
            temperature = temp_tc,
        )
        self.cc = ContextualContrastingModule(
            context_dim = context_dim,
            proj_dim    = proj_dim,
            temperature = temp_cc,
        )

        self.out_dim = context_dim   # feature dim exposed to classifier

    def forward(
        self,
        x_weak:   torch.Tensor,   # (B, 6, 128)
        x_strong: torch.Tensor,
    ) -> torch.Tensor:
        z_w = self.encoder(x_weak)    # (B, D, T)
        z_s = self.encoder(x_strong)

        tc_loss, ctx_w, ctx_s = self.tc(z_w, z_s)
        cc_loss               = self.cc(ctx_w, ctx_s)

        return self.lambda_tc * tc_loss + self.lambda_cc * cc_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encoder(x)               # (B, D, T)
        seq_t = seq.permute(0, 2, 1)        # (B, T, D)
        ctx, _ = self.tc.ar(seq_t)
        return ctx[:, -1, :]                # (B, context_dim)
