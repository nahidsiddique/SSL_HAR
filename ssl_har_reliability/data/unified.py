"""
Utilities for building the unified HAR datasets and splits.

This file handles:
- balanced subject-wise splits
- exact subject-wise splits
- leave-one-dataset-out transfer splits
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

CLASS_NAMES = ["walking", "upstairs", "downstairs", "sitting", "standing"]
N_CLASSES = 5

DATASET_IDS = {"ucihar": 0, "hhar": 1, "pamap2": 2, "motionsense": 3}
DATASET_NAMES = {value: key for key, value in DATASET_IDS.items()}


class HARDataset(Dataset):
    """Simple PyTorch dataset wrapper for the unified HAR windows."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        sources: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self.X = torch.from_numpy(X.transpose(0, 2, 1)).float()
        self.y = torch.from_numpy(y).long()
        self.subjects = subjects
        self.sources = sources
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]


def _prepare_unified_arrays(datasets):
    all_X, all_y, all_sub, all_src = [], [], [], []
    dataset_ranges = {}
    start = 0
    subj_offset = 0

    for name, (X, y, subjects) in datasets.items():
        src_id = DATASET_IDS[name]
        subjects = subjects.astype(np.int64) + subj_offset
        all_X.append(X)
        all_y.append(y)
        all_sub.append(subjects)
        all_src.append(np.full(len(y), src_id, dtype=np.int64))
        dataset_ranges[name] = np.arange(start, start + len(y))
        start += len(y)
        subj_offset = int(subjects.max()) + 1

    return (
        np.concatenate(all_X, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_sub, axis=0),
        np.concatenate(all_src, axis=0),
        dataset_ranges,
    )


def concatenate_unified_arrays(datasets: dict) -> dict:
    X, y, subjects, sources, _ = _prepare_unified_arrays(datasets)
    return {"X": X, "y": y, "subjects": subjects, "sources": sources}


def _subject_histograms(labels, subjects):
    unique_subjects = np.unique(subjects)
    hist = {}
    for sid in unique_subjects:
        hist[sid] = np.bincount(labels[subjects == sid], minlength=N_CLASSES).astype(np.float64)
    return unique_subjects, hist


def _balanced_subject_split(labels, subjects, val_fraction=0.10, test_fraction=0.20, seed=42):
    unique_subjects, hist = _subject_histograms(labels, subjects)
    n_subjects = len(unique_subjects)
    if n_subjects < 3:
        raise ValueError(f"Need at least 3 subjects for train/val/test split, got {n_subjects}.")

    rng = np.random.default_rng(seed)
    total_hist = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    target_fracs = {"train": 1.0 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    target_subjects = {
        "test": max(1, int(round(n_subjects * test_fraction))),
        "val": max(1, int(round(n_subjects * val_fraction))),
    }
    if target_subjects["test"] + target_subjects["val"] >= n_subjects:
        target_subjects["test"] = min(target_subjects["test"], n_subjects - 2)
        target_subjects["val"] = min(target_subjects["val"], n_subjects - target_subjects["test"] - 1)
    target_subjects["train"] = n_subjects - target_subjects["val"] - target_subjects["test"]

    target_hist = {name: total_hist * frac for name, frac in target_fracs.items()}
    assigned = {"train": [], "val": [], "test": []}
    current_hist = {name: np.zeros(N_CLASSES, dtype=np.float64) for name in assigned}
    current_subjects = {name: 0 for name in assigned}
    current_windows = {name: 0.0 for name in assigned}

    order = list(unique_subjects)
    rng.shuffle(order)
    order.sort(key=lambda sid: hist[sid].sum(), reverse=True)

    for sid in order:
        sid_hist = hist[sid]
        remaining = len(order) - sum(current_subjects.values())

        candidates = []
        for split in ("train", "val", "test"):
            if current_subjects[split] >= target_subjects[split]:
                continue
            needed_elsewhere = sum(
                max(0, target_subjects[name] - current_subjects[name])
                for name in ("train", "val", "test")
                if name != split
            )
            if remaining - 1 < needed_elsewhere:
                continue
            candidates.append(split)

        if not candidates:
            candidates = [
                split for split in ("train", "val", "test")
                if current_subjects[split] < target_subjects[split]
            ]

        best_split = None
        best_score = None
        for split in candidates:
            new_hist = current_hist[split] + sid_hist
            class_score = np.abs(target_hist[split] - new_hist).sum()
            size_score = abs(target_hist[split].sum() - (current_windows[split] + sid_hist.sum()))
            subject_score = abs(target_subjects[split] - (current_subjects[split] + 1))
            score = class_score + 0.25 * size_score + 5.0 * subject_score
            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        assigned[best_split].append(sid)
        current_hist[best_split] += sid_hist
        current_subjects[best_split] += 1
        current_windows[best_split] += sid_hist.sum()

    if not assigned["train"] or not assigned["val"] or not assigned["test"]:
        raise ValueError("Balanced subject split produced an empty split.")

    return (
        np.array(assigned["train"], dtype=subjects.dtype),
        np.array(assigned["val"], dtype=subjects.dtype),
        np.array(assigned["test"], dtype=subjects.dtype),
    )


def _exact_subject_split(subjects, val_fraction=0.10, test_fraction=0.20, seed=42):
    rng = np.random.default_rng(seed)
    unique_subjects = rng.permutation(np.unique(subjects))
    n_subjects = len(unique_subjects)
    if n_subjects < 3:
        raise ValueError(f"Need at least 3 subjects for train/val/test split, got {n_subjects}.")

    n_test = max(1, int(round(n_subjects * test_fraction)))
    n_val = max(1, int(round(n_subjects * val_fraction)))

    if n_test + n_val >= n_subjects:
        n_test = min(n_test, n_subjects - 2)
        n_val = min(n_val, n_subjects - n_test - 1)

    test_subjects = unique_subjects[:n_test]
    val_subjects = unique_subjects[n_test:n_test + n_val]
    train_subjects = unique_subjects[n_test + n_val:]

    if len(train_subjects) == 0 or len(val_subjects) == 0 or len(test_subjects) == 0:
        raise ValueError("Exact subject split produced an empty split.")

    return train_subjects, val_subjects, test_subjects


def _indices_from_subjects(subjects, subject_ids):
    return np.where(np.isin(subjects, subject_ids))[0]


def _subjectwise_split_indices(y, sub, idxs, protocol, val_fraction, test_fraction, seed):
    local_y = y[idxs]
    local_sub = sub[idxs]
    if protocol == "balanced":
        tr_subj, va_subj, te_subj = _balanced_subject_split(
            local_y,
            local_sub,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    elif protocol == "exact":
        tr_subj, va_subj, te_subj = _exact_subject_split(
            local_sub,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return (
        idxs[np.isin(local_sub, tr_subj)],
        idxs[np.isin(local_sub, va_subj)],
        idxs[np.isin(local_sub, te_subj)],
    )


def build_subjectwise_dataset(
    datasets: dict,
    protocol: str = "balanced",
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
    seed: int = 42,
    transform_train: Optional[Callable] = None,
    transform_eval: Optional[Callable] = None,
) -> dict:
    """Build a balanced or exact subject-wise split."""

    X, y, sub, src, dataset_ranges = _prepare_unified_arrays(datasets)
    train_parts, val_parts, test_parts = [], [], []

    for ds_i, (_, idxs) in enumerate(dataset_ranges.items()):
        tr_idx, va_idx, te_idx = _subjectwise_split_indices(
            y,
            sub,
            idxs,
            protocol=protocol,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed + ds_i,
        )
        train_parts.append(tr_idx)
        val_parts.append(va_idx)
        test_parts.append(te_idx)

    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    test_idx = np.sort(np.concatenate(test_parts))

    if (
        set(sub[train_idx]) & set(sub[val_idx])
        or set(sub[train_idx]) & set(sub[test_idx])
        or set(sub[val_idx]) & set(sub[test_idx])
    ):
        raise RuntimeError("Subject overlap detected between splits.")

    _, T, C = X.shape
    scaler = StandardScaler()
    scaler.fit(X[train_idx].reshape(-1, C))

    def _normalise(idxs):
        Xn = X[idxs].reshape(-1, C)
        Xn = scaler.transform(Xn).reshape(-1, T, C)
        return Xn.astype(np.float32)

    return {
        "train": HARDataset(_normalise(train_idx), y[train_idx], sub[train_idx], src[train_idx], transform_train),
        "val": HARDataset(_normalise(val_idx), y[val_idx], sub[val_idx], src[val_idx], transform_eval),
        "test": HARDataset(_normalise(test_idx), y[test_idx], sub[test_idx], src[test_idx], transform_eval),
        "scaler": scaler,
        "meta": {
            "protocol": protocol,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "train_subjects": sub[train_idx],
            "val_subjects": sub[val_idx],
            "test_subjects": sub[test_idx],
            "test_sources": src[test_idx],
        },
    }


def _pick_calibration_subjects_balanced(labels, subjects, min_fraction=0.10, min_windows=200, seed=42):
    unique_subjects, hist = _subject_histograms(labels, subjects)
    rng = np.random.default_rng(seed)
    order = list(unique_subjects)
    rng.shuffle(order)
    order.sort(key=lambda sid: hist[sid].sum(), reverse=True)

    total_hist = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    target_windows = max(int(round(len(labels) * min_fraction)), min_windows)
    target_hist = total_hist * min_fraction

    cal_subjects = []
    current_hist = np.zeros(N_CLASSES, dtype=np.float64)
    current_windows = 0.0

    while order and current_windows < target_windows:
        best_sid = None
        best_score = None
        for sid in order:
            new_hist = current_hist + hist[sid]
            score = np.abs(target_hist - new_hist).sum()
            score += 0.1 * abs(target_windows - (current_windows + hist[sid].sum()))
            if best_score is None or score < best_score:
                best_score = score
                best_sid = sid
        cal_subjects.append(best_sid)
        current_hist += hist[best_sid]
        current_windows += hist[best_sid].sum()
        order.remove(best_sid)
        if len(order) == 0:
            break

    if len(order) == 0:
        raise ValueError("Balanced calibration split consumed all target subjects.")

    return np.array(cal_subjects, dtype=subjects.dtype), np.array(order, dtype=subjects.dtype)


def _pick_calibration_subjects_exact(subjects, min_fraction=0.10, min_windows=200, seed=42):
    rng = np.random.default_rng(seed)
    unique_subjects = rng.permutation(np.unique(subjects))
    target_windows = max(int(round(len(subjects) * min_fraction)), min_windows)

    cal_subjects = []
    current = 0
    for sid in unique_subjects:
        count = int(np.sum(subjects == sid))
        if len(cal_subjects) < len(unique_subjects) - 1:
            cal_subjects.append(sid)
            current += count
        if current >= target_windows and len(cal_subjects) < len(unique_subjects):
            break

    cal_subjects = np.array(cal_subjects, dtype=subjects.dtype)
    test_subjects = np.array(
        [sid for sid in unique_subjects if sid not in set(cal_subjects)],
        dtype=subjects.dtype,
    )
    if len(test_subjects) == 0:
        raise ValueError("Exact calibration split consumed all target subjects.")
    return cal_subjects, test_subjects


def build_transfer_splits(
    source_datasets: dict,
    target_dataset: tuple,
    target_name: str,
    protocol: str = "balanced",
    val_fraction: float = 0.10,
    seed: int = 42,
) -> dict:
    """Build leave-one-dataset-out transfer splits for the selected protocol."""

    src_X_list = [value[0] for value in source_datasets.values()]
    src_X_all = np.concatenate(src_X_list, axis=0)
    _, T, C = src_X_all.shape
    scaler = StandardScaler().fit(src_X_all.reshape(-1, C))

    def _norm(X):
        return scaler.transform(X.reshape(-1, C)).reshape(-1, T, C).astype(np.float32)

    all_X, all_y, all_sub, all_src = [], [], [], []
    subj_offset = 0
    for name, (X, y, subjects) in source_datasets.items():
        subjects = subjects.astype(np.int64) + subj_offset
        all_X.append(_norm(X))
        all_y.append(y)
        all_sub.append(subjects)
        all_src.append(np.full(len(y), DATASET_IDS[name], dtype=np.int64))
        subj_offset = int(subjects.max()) + 1

    source_ds = HARDataset(
        np.concatenate(all_X, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_sub, axis=0),
        np.concatenate(all_src, axis=0),
    )

    tgt_X, tgt_y, tgt_sub = target_dataset
    tgt_sub = tgt_sub.astype(np.int64)
    if protocol == "balanced":
        cal_subjects, test_subjects = _pick_calibration_subjects_balanced(
            tgt_y,
            tgt_sub,
            min_fraction=val_fraction,
            min_windows=200,
            seed=seed,
        )
    elif protocol == "exact":
        cal_subjects, test_subjects = _pick_calibration_subjects_exact(
            tgt_sub,
            min_fraction=val_fraction,
            min_windows=200,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    cal_idx = _indices_from_subjects(tgt_sub, cal_subjects)
    test_idx = _indices_from_subjects(tgt_sub, test_subjects)
    tgt_X_norm = _norm(tgt_X)
    tgt_src = np.full(len(tgt_y), DATASET_IDS.get(target_name, -1), dtype=np.int64)

    return {
        "source": HARDataset(
            np.concatenate(all_X, axis=0),
            np.concatenate(all_y, axis=0),
            np.concatenate(all_sub, axis=0),
            np.concatenate(all_src, axis=0),
        ),
        "cal": HARDataset(tgt_X_norm[cal_idx], tgt_y[cal_idx], tgt_sub[cal_idx], tgt_src[cal_idx]),
        "test": HARDataset(tgt_X_norm[test_idx], tgt_y[test_idx], tgt_sub[test_idx], tgt_src[test_idx]),
        "scaler": scaler,
        "meta": {
            "protocol": protocol,
            "cal_subjects": tgt_sub[cal_idx],
            "test_subjects": tgt_sub[test_idx],
        },
    }
