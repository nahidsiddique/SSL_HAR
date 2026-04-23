"""
Evaluation metrics:
  - Accuracy, Macro-F1
  - ECE, MCE, NLL, Brier score
  - Reliability diagram data
  - Temperature scaling (post-hoc calibration)
  - Conformal prediction (split conformal, APS)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


# ─── classification ──────────────────────────────────────────────────────────

def compute_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    return float(accuracy_score(labels, preds))


def compute_macro_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    return float(f1_score(labels, preds, average='macro', zero_division=0))


# ─── calibration ─────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (equal-width bins on max confidence)."""
    conf   = probs.max(axis=1)
    preds  = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_b  = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece   += mask.mean() * abs(acc_b - conf_b)
    return float(ece)


def compute_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error."""
    conf    = probs.max(axis=1)
    preds   = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce   = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(correct[mask].mean() - conf[mask].mean()))
    return float(mce)


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-10
    return float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    K      = probs.shape[1]
    onehot = np.eye(K)[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def reliability_diagram_data(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict:
    """
    Returns bin statistics for plotting reliability diagrams.
    """
    conf    = probs.max(axis=1)
    preds   = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)

    edges      = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs  = []
    bin_accs   = []
    bin_counts = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            bin_confs.append(float((lo + hi) / 2))
            bin_accs.append(0.0)
        else:
            bin_confs.append(float(conf[mask].mean()))
            bin_accs.append(float(correct[mask].mean()))

    return {
        'bin_confidences': np.array(bin_confs),
        'bin_accuracies':  np.array(bin_accs),
        'bin_counts':      np.array(bin_counts),
        'ece':             compute_ece(probs, labels, n_bins),
        'mce':             compute_mce(probs, labels, n_bins),
    }


# ─── temperature scaling ──────────────────────────────────────────────────────

def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 1000,
) -> float:
    """
    Fit a single temperature T on validation logits by minimising NLL.
    Returns optimal T > 0.  (T > 1 → softer / less overconfident)
    """
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()

    T         = torch.nn.Parameter(torch.ones(1))
    optimiser = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(logits_t / T.clamp(min=1e-4), labels_t)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(T.item())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Scale logits by T and convert to probabilities."""
    from scipy.special import softmax
    return softmax(logits / T, axis=1).astype(np.float32)


# ─── conformal prediction ─────────────────────────────────────────────────────

def conformal_calibrate(
    probs_cal: np.ndarray,
    labels_cal: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Split conformal calibration using LAC nonconformity score.
    s_i = 1 - p_hat[y_i]  (probability assigned to true class)
    Returns threshold q such that coverage >= 1 - alpha.
    """
    n      = len(labels_cal)
    scores = 1.0 - probs_cal[np.arange(n), labels_cal]
    level  = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method='higher'))


def conformal_predict(probs_test: np.ndarray, q: float) -> np.ndarray:
    """Return boolean prediction sets: (N, K). Class k in set iff 1-p_k <= q."""
    return (probs_test >= (1.0 - q))


def aps_calibrate(
    probs_cal: np.ndarray,
    labels_cal: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Adaptive Prediction Sets (APS) calibration.
    Score = cumulative sum of sorted probabilities up to and including true class.
    Produces variable-size sets: larger in uncertain regions.
    """
    n      = len(labels_cal)
    scores = []
    for i in range(n):
        order    = np.argsort(probs_cal[i])[::-1]   # descending by prob
        cum      = np.cumsum(probs_cal[i][order])
        rank     = int(np.where(order == labels_cal[i])[0][0])
        scores.append(float(cum[rank]))
    scores = np.array(scores)
    level  = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method='higher'))


def aps_predict(probs_test: np.ndarray, q: float) -> np.ndarray:
    """APS prediction sets given threshold q."""
    N, K   = probs_test.shape
    sets   = np.zeros((N, K), dtype=bool)
    for i in range(N):
        order   = np.argsort(probs_test[i])[::-1]
        cum     = np.cumsum(probs_test[i][order])
        n_incl  = int(np.searchsorted(cum, q)) + 1
        sets[i, order[:n_incl]] = True
    return sets


def conformal_evaluate(
    probs_cal:  np.ndarray,
    labels_cal: np.ndarray,
    probs_test: np.ndarray,
    labels_test: np.ndarray,
    alpha: float = 0.10,
    method: str = 'lac',   # 'lac' or 'aps'
) -> dict:
    """
    Full conformal evaluation: calibrate on cal, evaluate on test.

    Returns dict with coverage, average set size, per-class coverage.
    """
    if method == 'aps':
        q    = aps_calibrate(probs_cal, labels_cal, alpha)
        sets = aps_predict(probs_test, q)
    else:   # lac
        q    = conformal_calibrate(probs_cal, labels_cal, alpha)
        sets = conformal_predict(probs_test, q)

    N = len(labels_test)
    K = probs_test.shape[1]

    coverage        = float(sets[np.arange(N), labels_test].mean())
    avg_size        = float(sets.sum(axis=1).mean())
    singleton_rate  = float((sets.sum(axis=1) == 1).mean())
    empty_rate      = float((sets.sum(axis=1) == 0).mean())

    per_class = {}
    for c in range(K):
        mask = labels_test == c
        if mask.sum() > 0:
            per_class[c] = float(sets[np.where(mask)[0], c].mean())

    return {
        'method':             method,
        'alpha':              alpha,
        'q_threshold':        q,
        'empirical_coverage': coverage,
        'nominal_coverage':   1 - alpha,
        'avg_set_size':       avg_size,
        'singleton_rate':     singleton_rate,
        'empty_rate':         empty_rate,
        'per_class_coverage': per_class,
    }


# ─── full eval summary ────────────────────────────────────────────────────────

def full_evaluation(
    logits:      np.ndarray,      # (N, K) raw logits
    labels:      np.ndarray,      # (N,)
    logits_cal:  np.ndarray,      # (N_cal, K) for temp scaling + conformal
    labels_cal:  np.ndarray,
    alpha:       float = 0.10,
    n_bins:      int   = 15,
) -> dict:
    """
    Compute all metrics before and after temperature scaling.
    """
    from scipy.special import softmax

    probs_raw = softmax(logits, axis=1).astype(np.float32)

    # temperature scaling on calibration set
    T            = fit_temperature(logits_cal, labels_cal)
    probs_scaled = apply_temperature(logits, T)
    probs_cal_ts = apply_temperature(logits_cal, T)

    def _metrics(probs, tag):
        return {
            f'{tag}_acc':     compute_accuracy(probs, labels),
            f'{tag}_f1':      compute_macro_f1(probs, labels),
            f'{tag}_ece':     compute_ece(probs, labels, n_bins),
            f'{tag}_mce':     compute_mce(probs, labels, n_bins),
            f'{tag}_nll':     compute_nll(probs, labels),
            f'{tag}_brier':   compute_brier(probs, labels),
        }

    results = {}
    results.update(_metrics(probs_raw,    'raw'))
    results.update(_metrics(probs_scaled, 'ts'))
    results['temperature'] = T

    # conformal on scaled probs
    cal_probs = apply_temperature(logits_cal, T)
    conformal = conformal_evaluate(cal_probs, labels_cal,
                                   probs_scaled, labels, alpha=alpha)
    results['conformal'] = conformal

    results['reliability_raw'] = reliability_diagram_data(probs_raw, labels, n_bins)
    results['reliability_ts']  = reliability_diagram_data(probs_scaled, labels, n_bins)

    return results
