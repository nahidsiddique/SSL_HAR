"""Pretraining helpers for the SSL models."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def _apply_batch_augment(aug_fn: Callable, batch: torch.Tensor):
    view1, view2 = [], []
    for sample in batch:
        aug_out = aug_fn(sample)
        if isinstance(aug_out, tuple):
            sample_v1, sample_v2 = aug_out
        else:
            sample_v1, sample_v2 = sample, aug_out
        view1.append(sample_v1)
        view2.append(sample_v2)
    return torch.stack(view1, dim=0), torch.stack(view2, dim=0)


def pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    n_epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    aug_fn: Optional[Callable] = None,
    method: str = "simclr",
    scheduler: str = "cosine",
    use_amp: bool = True,
    verbose: bool = True,
) -> list:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            eta_min=lr * 0.01,
        )
    else:
        sched = None

    amp_enabled = bool(use_amp and device != "cpu" and method != "softclt")
    scaler = GradScaler(enabled=amp_enabled)
    loss_history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        skipped = 0

        for x, _ in train_loader:
            x = x.to(device)
            if aug_fn is not None:
                with torch.no_grad():
                    v1, v2 = _apply_batch_augment(aug_fn, x)
                    v1 = v1.to(device)
                    v2 = v2.to(device)
            else:
                v1 = v2 = x

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                if method not in {"simclr", "tstcc", "tfc", "softclt"}:
                    raise ValueError(f"Unknown method: {method}")
                loss = model(v1, v2)

            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()

            if method == "softclt":
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        epoch_loss = total_loss / max(n_batches, 1)
        loss_history.append(epoch_loss if n_batches > 0 else float("nan"))

        if sched is not None:
            sched.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"[{method.upper()}] Epoch {epoch:3d}/{n_epochs} | "
                f"Loss: {loss_history[-1]:.4f} | "
                f"Skipped: {skipped:3d} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    return loss_history


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: str = "cuda") -> tuple:
    model.eval()
    model.to(device)
    feats_list, labels_list = [], []

    for x, y in loader:
        x = x.to(device)
        h = model.encode(x)
        feats_list.append(h.cpu().numpy())
        labels_list.append(y.numpy())

    return (
        np.concatenate(feats_list, axis=0),
        np.concatenate(labels_list, axis=0),
    )
