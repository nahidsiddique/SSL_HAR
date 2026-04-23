"""
HHAR Dataset Loader
Files: Phones_accelerometer.csv, Phones_gyroscope.csv (or Watch_*)
Columns: Index, Arrival_Time, Creation_Time, x, y, z, User, Model, Device, gt
Activities kept: walk, stairsup, stairsdown, sit, stand  (bike/null dropped)
Output: (N, 128, 6)  at 50 Hz
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sp_signal

ACTIVITY_MAP = {
    'walk':       0,
    'stairsup':   1,
    'stairsdown': 2,
    'sit':        3,
    'stand':      4,
}

TARGET_FS  = 50
WINDOW_LEN = 128   # 2.56 s at 50 Hz
STRIDE     = 64    # 50 % overlap


def _resample(data: np.ndarray, src_fs: float, tgt_fs: float = TARGET_FS) -> np.ndarray:
    if abs(src_fs - tgt_fs) < 2.0:
        return data
    n_out = max(1, int(round(len(data) * tgt_fs / src_fs)))
    return sp_signal.resample(data, n_out, axis=0)


def _slide(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    starts = range(0, n - WINDOW_LEN + 1, STRIDE)
    if not starts:
        return np.empty((0, WINDOW_LEN, data.shape[1]), dtype=np.float32)
    return np.stack([data[s:s + WINDOW_LEN] for s in starts])


def _estimate_fs(timestamps_ns: np.ndarray) -> float:
    diffs = np.diff(timestamps_ns)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return TARGET_FS
    median_dt_s = np.median(diffs) * 1e-9
    return 1.0 / median_dt_s if median_dt_s > 0 else TARGET_FS


def load_hhar(root: str, device_type: str = 'phone') -> tuple:
    """
    Args:
        root        : path to 'Activity recognition exp/' directory
        device_type : 'phone' or 'watch'

    Returns:
        X        : (N, 128, 6)  float32
        y        : (N,)         int64   [0..4]
        subjects : (N,)         int64
    """
    root   = Path(root)
    prefix = 'Phones' if device_type == 'phone' else 'Watch'

    acc_df = pd.read_csv(root / f'{prefix}_accelerometer.csv')
    gyr_df = pd.read_csv(root / f'{prefix}_gyroscope.csv')

    acc_df = acc_df[acc_df['gt'].isin(ACTIVITY_MAP)].copy()
    gyr_df = gyr_df[gyr_df['gt'].isin(ACTIVITY_MAP)].copy()

    acc_df.sort_values(['User', 'Device', 'gt', 'Creation_Time'], inplace=True)
    gyr_df.sort_values(['User', 'Device', 'gt', 'Creation_Time'], inplace=True)

    user_map = {u: i for i, u in enumerate(sorted(acc_df['User'].unique()))}

    windows, labels, subjects = [], [], []

    for (user, device, gt), acc_grp in acc_df.groupby(['User', 'Device', 'gt']):
        gyr_grp = gyr_df[
            (gyr_df['User'] == user) &
            (gyr_df['Device'] == device) &
            (gyr_df['gt'] == gt)
        ]
        if len(acc_grp) < WINDOW_LEN or len(gyr_grp) < WINDOW_LEN:
            continue

        src_fs = _estimate_fs(acc_grp['Creation_Time'].values)

        acc_xyz = acc_grp[['x', 'y', 'z']].values.astype(np.float32)
        gyr_xyz = gyr_grp[['x', 'y', 'z']].values.astype(np.float32)

        # align lengths before resampling
        n_min = min(len(acc_xyz), len(gyr_xyz))
        acc_xyz = _resample(acc_xyz[:n_min], src_fs)
        gyr_xyz = _resample(gyr_xyz[:n_min], src_fs)

        n_min = min(len(acc_xyz), len(gyr_xyz))
        data   = np.concatenate([acc_xyz[:n_min], gyr_xyz[:n_min]], axis=1)  # (T, 6)

        wins = _slide(data)
        if len(wins) == 0:
            continue

        windows.append(wins)
        labels.extend([ACTIVITY_MAP[gt]] * len(wins))
        subjects.extend([user_map[user]] * len(wins))

    X        = np.concatenate(windows, axis=0).astype(np.float32)
    y        = np.array(labels,   dtype=np.int64)
    subjects = np.array(subjects, dtype=np.int64)
    return X, y, subjects
