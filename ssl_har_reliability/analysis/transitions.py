"""Helpers for the transition-window analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from ssl_har_reliability.data import DATASET_IDS, DATASET_NAMES
from ssl_har_reliability.metrics import compute_ece


def compute_transition_mask(
    labels: np.ndarray,
    subjects: np.ndarray,
    sources: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=bool)
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and subjects[end] == subjects[start] and sources[end] == sources[start]:
            end += 1

        segment_labels = labels[start:end]
        change_points = np.where(segment_labels[1:] != segment_labels[:-1])[0]
        for boundary in change_points:
            lo = max(0, boundary - radius + 1)
            hi = min(len(segment_labels), boundary + radius + 1)
            mask[start + lo:start + hi] = True
        start = end
    return mask


def _subset_metrics(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray, method: str, split_name: str, dataset_name: str) -> dict:
    logits_sub = logits[mask]
    labels_sub = labels[mask]
    probs = torch.softmax(torch.as_tensor(logits_sub), dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    acc = accuracy_score(labels_sub, preds)
    macro_f1 = f1_score(labels_sub, preds, average="macro", zero_division=0)
    ece = compute_ece(probs, labels_sub)
    return {
        "Method": method,
        "Split": split_name,
        "Dataset": dataset_name,
        "N": int(mask.sum()),
        "Acc": float(acc),
        "Macro-F1": float(macro_f1),
        "ECE": float(ece),
        "Avg Confidence": float(conf.mean()),
        "Overconfidence Gap": float(conf.mean() - acc),
    }


def build_transition_reliability_table(
    outputs_by_method: dict,
    full_arrays: dict,
    test_idx: np.ndarray,
    radius: int = 2,
) -> pd.DataFrame:
    full_transition_mask = compute_transition_mask(
        full_arrays["y"],
        full_arrays["subjects"],
        full_arrays["sources"],
        radius=radius,
    )

    test_transition_mask = full_transition_mask[test_idx]
    test_source_ids = full_arrays["sources"][test_idx]
    id_to_name = {value: key for key, value in DATASET_IDS.items()}

    rows = []
    for method_name, output in outputs_by_method.items():
        logits = output["logits_test"]
        labels = output["labels_test"]

        all_mask = np.ones(len(labels), dtype=bool)
        stable_mask = ~test_transition_mask

        rows.append(_subset_metrics(logits, labels, all_mask, method_name, "all", "overall"))
        if test_transition_mask.any():
            rows.append(_subset_metrics(logits, labels, test_transition_mask, method_name, "transition", "overall"))
        if stable_mask.any():
            rows.append(_subset_metrics(logits, labels, stable_mask, method_name, "stable", "overall"))

        for src_id, dataset_name in sorted(id_to_name.items()):
            ds_mask = test_source_ids == src_id
            if not ds_mask.any():
                continue
            ds_transition = ds_mask & test_transition_mask
            ds_stable = ds_mask & stable_mask
            if ds_transition.any():
                rows.append(_subset_metrics(logits, labels, ds_transition, method_name, "transition", dataset_name))
            if ds_stable.any():
                rows.append(_subset_metrics(logits, labels, ds_stable, method_name, "stable", dataset_name))

    return pd.DataFrame(rows).sort_values(["Dataset", "Method", "Split"]).reset_index(drop=True)
