"""
Downstream evaluation: linear probing and fine-tuning.
Produces logits + features used by metrics.py for ECE / conformal.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler as SKScaler

from ssl_har_reliability.models.backbone import MLPHead
from ssl_har_reliability.training.pretrain import extract_features


# ─── linear probe (sklearn) ──────────────────────────────────────────────────

def linear_probe_sklearn(
    model:        nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    cal_loader:   DataLoader,
    device:       str = 'cuda',
    max_iter:     int = 1000,
) -> dict:
    """
    Fit a LogisticRegression on frozen features.
    Returns dict with logits_test, logits_cal, labels_test, labels_cal.
    """
    X_tr, y_tr = extract_features(model, train_loader, device)
    X_te, y_te = extract_features(model, test_loader,  device)
    X_cal, y_cal = extract_features(model, cal_loader, device)

    scaler = SKScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_te  = scaler.transform(X_te)
    X_cal = scaler.transform(X_cal)

    clf = LogisticRegression(max_iter=max_iter, C=1.0, solver='lbfgs',
                              multi_class='multinomial')
    clf.fit(X_tr, y_tr)

    logits_te  = clf.predict_log_proba(X_te).astype(np.float32)
    logits_cal = clf.predict_log_proba(X_cal).astype(np.float32)

    return {
        'logits_test':  logits_te,
        'logits_cal':   logits_cal,
        'labels_test':  y_te,
        'labels_cal':   y_cal,
        'features_test': X_te,
    }


# ─── fine-tuning (end-to-end) ─────────────────────────────────────────────────

def finetune(
    model:        nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    cal_loader:   DataLoader,
    n_classes:    int   = 5,
    n_epochs:     int   = 50,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    freeze_encoder: bool = False,
    device:       str   = 'cuda',
    verbose:      bool  = True,
) -> dict:
    """
    Fine-tune encoder + MLP head end-to-end (or with frozen encoder).
    Returns logits for test and cal splits.
    """
    model = model.to(device)
    head  = MLPHead(model.out_dim, hidden_dim=256, n_classes=n_classes).to(device)

    if freeze_encoder:
        for p in model.parameters():
            p.requires_grad = False
        params = head.parameters()
    else:
        params = list(model.parameters()) + list(head.parameters())

    optimiser = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=lr * 0.01
    )

    for epoch in range(1, n_epochs + 1):
        model.train()
        head.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            h    = model.encode(x) if hasattr(model, 'encode') else model(x)
            logits = head(h)
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        scheduler.step()

        if verbose and epoch % 10 == 0:
            print(f'  Fine-tune epoch {epoch:3d}/{n_epochs} | '
                  f'Loss: {total_loss / len(train_loader):.4f}')

    return _collect_logits(model, head, test_loader, cal_loader, device)


def _collect_logits(
    model:       nn.Module,
    head:        nn.Module,
    test_loader:  DataLoader,
    cal_loader:  DataLoader,
    device:      str,
) -> dict:
    model.eval()
    head.eval()

    def _run(loader):
        logits_all, labels_all = [], []
        with torch.no_grad():
            for x, y in loader:
                x     = x.to(device)
                h     = model.encode(x) if hasattr(model, 'encode') else model(x)
                logits = head(h)
                logits_all.append(logits.cpu().numpy())
                labels_all.append(y.numpy())
        return (np.concatenate(logits_all,  axis=0),
                np.concatenate(labels_all, axis=0))

    logits_te, y_te   = _run(test_loader)
    logits_cal, y_cal = _run(cal_loader)

    return {
        'logits_test':  logits_te.astype(np.float32),
        'logits_cal':   logits_cal.astype(np.float32),
        'labels_test':  y_te,
        'labels_cal':   y_cal,
    }
