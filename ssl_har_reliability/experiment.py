"""Helper functions for running the main experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ssl_har_reliability.augmentations import SoftCLTTransform, TwoViewTransform, WeakStrongTransform
from ssl_har_reliability.data import (
    build_transfer_splits,
    load_hhar,
    load_motionsense,
    load_pamap2,
    load_ucihar,
)
from ssl_har_reliability.metrics import full_evaluation
from ssl_har_reliability.models.backbone import MLPHead, CNN1D, ResNet1D
from ssl_har_reliability.models.simclr import SimCLR
from ssl_har_reliability.models.softclt import SoftCLT
from ssl_har_reliability.models.tfc import TFC
from ssl_har_reliability.models.tstcc import TSTCC
from ssl_har_reliability.training.evaluate import finetune, linear_probe_sklearn
from ssl_har_reliability.training.pretrain import pretrain


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_all_datasets(dataset_roots: dict) -> dict:
    return {
        "ucihar": load_ucihar(dataset_roots["ucihar"]),
        "hhar": load_hhar(dataset_roots["hhar"]),
        "pamap2": load_pamap2(dataset_roots["pamap2"]),
        "motionsense": load_motionsense(dataset_roots["motionsense"]),
    }


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_ssl_model(method: str):
    method = method.lower()
    if method == "simclr":
        return SimCLR(ResNet1D())
    if method == "tstcc":
        return TSTCC(CNN1D())
    if method == "tfc":
        return TFC()
    if method == "softclt":
        return SoftCLT(CNN1D())
    raise ValueError(f"Unknown method: {method}")


def get_ssl_augmenter(method: str):
    method = method.lower()
    if method in {"simclr", "tfc"}:
        return TwoViewTransform("jitter", "scaling")
    if method == "tstcc":
        return WeakStrongTransform()
    if method == "softclt":
        return SoftCLTTransform("jitter")
    raise ValueError(f"Unknown method: {method}")


def _summarize_eval_results(method: str, setting: str, metrics: dict) -> dict:
    conformal = metrics["conformal"]
    return {
        "method": method,
        "setting": setting,
        "acc": metrics["raw_acc"],
        "macro_f1": metrics["raw_f1"],
        "ece_raw": metrics["raw_ece"],
        "ece_ts": metrics["ts_ece"],
        "temperature": metrics["temperature"],
        "nll": metrics["raw_nll"],
        "brier": metrics["raw_brier"],
        "coverage": conformal["empirical_coverage"],
        "avg_set_size": conformal["avg_set_size"],
    }


def run_ssl_method(
    method: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    pretrain_epochs: int = 100,
    finetune_epochs: int = 50,
    pretrain_lr: float = 3e-4,
    finetune_lr: float = 1e-3,
) -> dict:
    model = build_ssl_model(method)
    augmenter = get_ssl_augmenter(method)
    loss_history = pretrain(
        model,
        train_loader,
        n_epochs=pretrain_epochs,
        lr=pretrain_lr,
        device=device,
        aug_fn=augmenter,
        method=method.lower(),
    )

    linear_output = linear_probe_sklearn(
        model,
        train_loader,
        test_loader,
        val_loader,
        device=device,
    )
    linear_metrics = full_evaluation(
        linear_output["logits_test"],
        linear_output["labels_test"],
        linear_output["logits_cal"],
        linear_output["labels_cal"],
    )

    finetune_output = finetune(
        model,
        train_loader,
        test_loader,
        val_loader,
        n_epochs=finetune_epochs,
        lr=finetune_lr,
        device=device,
    )
    finetune_metrics = full_evaluation(
        finetune_output["logits_test"],
        finetune_output["labels_test"],
        finetune_output["logits_cal"],
        finetune_output["labels_cal"],
    )

    return {
        "model": model,
        "pretrain_loss_history": loss_history,
        "linear": linear_output,
        "linear_metrics": linear_metrics,
        "finetune": finetune_output,
        "finetune_metrics": finetune_metrics,
        "summary_rows": [
            _summarize_eval_results(method, "linear", linear_metrics),
            _summarize_eval_results(method, "finetune", finetune_metrics),
        ],
    }


def run_transfer_eval_for_method(
    method: str,
    datasets: dict,
    protocol: str,
    batch_size: int,
    device: str,
    pretrain_epochs: int = 100,
    seed: int = 42,
) -> list[dict]:
    rows = []
    for target_name, target_dataset in datasets.items():
        source_datasets = {name: value for name, value in datasets.items() if name != target_name}
        transfer_splits = build_transfer_splits(
            source_datasets,
            target_dataset,
            target_name,
            protocol=protocol,
            seed=seed,
        )

        source_loader = make_loader(transfer_splits["source"], batch_size, shuffle=True)
        cal_loader = make_loader(transfer_splits["cal"], batch_size, shuffle=False)
        test_loader = make_loader(transfer_splits["test"], batch_size, shuffle=False)
        model = build_ssl_model(method)
        augmenter = get_ssl_augmenter(method)
        pretrain(
            model,
            source_loader,
            n_epochs=pretrain_epochs,
            device=device,
            aug_fn=augmenter,
            method=method.lower(),
        )
        linear_output = linear_probe_sklearn(
            model,
            source_loader,
            test_loader,
            cal_loader,
            device=device,
        )
        linear_metrics = full_evaluation(
            linear_output["logits_test"],
            linear_output["labels_test"],
            linear_output["logits_cal"],
            linear_output["labels_cal"],
        )
        rows.append(
            {
                "method": method,
                "target": target_name,
                "acc": linear_metrics["raw_acc"],
                "macro_f1": linear_metrics["raw_f1"],
                "ece_raw": linear_metrics["raw_ece"],
                "ece_ts": linear_metrics["ts_ece"],
                "coverage": linear_metrics["conformal"]["empirical_coverage"],
                "avg_set_size": linear_metrics["conformal"]["avg_set_size"],
            }
        )
    return rows


def _collect_logits(encoder, head, loader: DataLoader, device: str):
    logits_all, labels_all = [], []
    encoder.eval()
    head.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            features = encoder(x)
            logits = head(features)
            logits_all.append(logits.cpu().numpy())
            labels_all.append(y.numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)


def run_supervised_baseline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    encoder = ResNet1D().to(device)
    head = MLPHead(encoder.out_dim, hidden_dim=256, n_classes=5).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=lr * 0.01,
    )

    loss_history = []
    for _ in range(n_epochs):
        encoder.train()
        head.train()
        running_loss = 0.0
        batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(encoder(x))
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        loss_history.append(running_loss / max(batches, 1))

    logits_test, labels_test = _collect_logits(encoder, head, test_loader, device)
    logits_cal, labels_cal = _collect_logits(encoder, head, val_loader, device)
    metrics = full_evaluation(logits_test, labels_test, logits_cal, labels_cal)

    return {
        "encoder": encoder,
        "head": head,
        "loss_history": loss_history,
        "metrics": metrics,
    }


def write_json(path: Path, payload: dict) -> None:
    def _convert(value):
        if isinstance(value, dict):
            return {key: _convert(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, tuple):
            return [_convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_convert(payload), indent=2))
