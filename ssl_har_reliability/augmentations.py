"""
Training-time signal transforms for self-supervised IMU HAR.

These transforms are used during SSL pretraining. All functions work on
tensors of shape ``(C, T)`` in channels-first format.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ─── Core signal transforms ─────────────────────────────────────────────────

def jitter(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    factor = 1.0 + torch.randn(x.shape[0], 1, device=x.device) * sigma
    return x * factor


def negation(x: torch.Tensor) -> torch.Tensor:
    return -x


def time_flip(x: torch.Tensor) -> torch.Tensor:
    return x.flip(-1)


def permutation(x: torch.Tensor, n_segs: int = 4) -> torch.Tensor:
    C, T   = x.shape
    seg    = T // n_segs
    segs   = [x[:, i * seg:(i + 1) * seg] for i in range(n_segs)]
    perm   = torch.randperm(n_segs)
    return torch.cat([segs[i] for i in perm], dim=1)


def time_mask(x: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    C, T      = x.shape
    mask_len  = max(1, int(T * ratio))
    start     = np.random.randint(0, T - mask_len + 1)
    x         = x.clone()
    x[:, start:start + mask_len] = 0.0
    return x


def channel_dropout(x: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    mask = (torch.rand(x.shape[0], 1, device=x.device) > p).float()
    return x * mask


def window_crop(x: torch.Tensor, crop_ratio: float = 0.9) -> torch.Tensor:
    """Random temporal crop, resized back to original length."""
    C, T      = x.shape
    crop_len  = max(2, int(T * crop_ratio))
    start     = np.random.randint(0, T - crop_len + 1)
    cropped   = x[:, start:start + crop_len].unsqueeze(0)  # (1, C, crop_len)
    return F.interpolate(cropped, size=T, mode='linear', align_corners=False).squeeze(0)


def magnitude_warp(x: torch.Tensor, sigma: float = 0.1, n_knots: int = 4) -> torch.Tensor:
    """Smooth random amplitude distortion via spline-like warp."""
    C, T     = x.shape
    knots    = 1.0 + torch.randn(C, n_knots, device=x.device) * sigma
    warp     = F.interpolate(knots.unsqueeze(0), size=T,
                             mode='linear', align_corners=False).squeeze(0)  # (C, T)
    return x * warp


# ─── Frequency-domain transforms used by TF-C ───────────────────────────────

def freq_mask(x_freq: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    """Zero out random frequency bands. x_freq: (C, F) magnitude spectrum."""
    C, F     = x_freq.shape
    n_mask   = max(1, int(F * ratio))
    start    = np.random.randint(0, F - n_mask + 1)
    x_freq   = x_freq.clone()
    x_freq[:, start:start + n_mask] = 0.0
    return x_freq


def freq_inject(x_freq: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Add small random noise to frequency magnitudes."""
    return x_freq + torch.randn_like(x_freq).abs() * sigma * x_freq.mean()


# ─── Named transform registry ────────────────────────────────────────────────

AUG_FNS = {
    'jitter':          jitter,
    'scaling':         scaling,
    'negation':        negation,
    'time_flip':       time_flip,
    'permutation':     permutation,
    'time_mask':       time_mask,
    'channel_dropout': channel_dropout,
    'window_crop':     window_crop,
    'magnitude_warp':  magnitude_warp,
    'identity':        lambda x: x,
}


def gen_aug(x: torch.Tensor, aug_type: str) -> torch.Tensor:
    if aug_type not in AUG_FNS:
        raise ValueError(f'Unknown augmentation: {aug_type}. Choose from {list(AUG_FNS)}')
    return AUG_FNS[aug_type](x)


# ─── composite transforms ────────────────────────────────────────────────────

class TwoViewTransform:
    """
    Build the paired views required by SimCLR and TF-C.

    Returns two independently transformed views with shape ``(C, T)``.
    """
    def __init__(self, aug1: str = 'jitter', aug2: str = 'scaling'):
        self.aug1 = aug1
        self.aug2 = aug2

    def __call__(self, x: torch.Tensor):
        return gen_aug(x, self.aug1), gen_aug(x, self.aug2)


class WeakStrongTransform:
    """
    Build the asymmetric weak/strong pair used by TS-TCC.

    Weak view:
    small jitter + small scaling

    Strong view:
    permutation + stronger jitter
    """
    def __call__(self, x: torch.Tensor):
        weak   = scaling(jitter(x, sigma=0.01), sigma=0.1)
        strong = jitter(permutation(x, n_segs=4), sigma=0.08)
        return weak, strong


class SoftCLTTransform:
    """
    Build the single transformed view used by SoftCLT.

    Temporal weighting is handled inside the SoftCLT loss, not here.
    """
    def __init__(self, aug: str = 'jitter'):
        self.aug = aug

    def __call__(self, x: torch.Tensor):
        return gen_aug(x, self.aug)
