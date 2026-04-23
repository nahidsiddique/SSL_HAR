"""
PAMAP2 Dataset Loader
Files: Protocol/subject10{1..9}.dat  (space-separated, 54 cols)
Column layout (0-based):
  0  timestamp
  1  activityID
  2  heart rate
  3-19  IMU hand  : temp(3), acc1_xyz(4-6), acc2_xyz(7-9), gyro_xyz(10-12), mag_xyz(13-15), orient(16-19)
  20-36 IMU chest : temp(20), acc1_xyz(21-23), acc2_xyz(24-26), gyro_xyz(27-29), ...
  37-53 IMU ankle : temp(37), acc1_xyz(38-40), acc2_xyz(41-43), gyro_xyz(44-46), ...

We use chest IMU: acc1 (cols 21-23) + gyro (cols 27-29)  → 6 channels
Source 100 Hz → downsample to 50 Hz → window at 128 samples (2.56 s)
Activities kept: sit=2, stand=3, walk=4, upstairs=12, downstairs=13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sp_signal

ACTIVITY_MAP = {
    4:  0,   # walking          → walking
    12: 1,   # ascending stairs → upstairs
    13: 2,   # descending stairs→ downstairs
    2:  3,   # sitting          → sitting
    3:  4,   # standing         → standing
}

SRC_FS     = 100
TARGET_FS  = 50
WINDOW_LEN = 128
STRIDE     = 64

# 0-based column indices for chest IMU
CHEST_ACC1_COLS  = [21, 22, 23]
CHEST_GYRO_COLS  = [27, 28, 29]
SENSOR_COLS      = CHEST_ACC1_COLS + CHEST_GYRO_COLS


def _slide(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    starts = list(range(0, n - WINDOW_LEN + 1, STRIDE))
    if not starts:
        return np.empty((0, WINDOW_LEN, 6), dtype=np.float32)
    return np.stack([data[s:s + WINDOW_LEN] for s in starts])


def _contiguous_runs(mask: np.ndarray):
    """Yield (start, end) slices of True runs."""
    in_run = False
    start  = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start  = i
            in_run = True
        elif not v and in_run:
            yield start, i
            in_run = False
    if in_run:
        yield start, len(mask)


def load_pamap2(root: str) -> tuple:
    """
    Args:
        root: path to 'PAMAP2_Dataset/' directory

    Returns:
        X        : (N, 128, 6)  float32
        y        : (N,)         int64   [0..4]
        subjects : (N,)         int64
    """
    root         = Path(root)
    protocol_dir = root / 'Protocol'

    windows, labels, subjects = [], [], []

    for subj_id, dat_file in enumerate(sorted(protocol_dir.glob('subject*.dat'))):
        df = pd.read_csv(dat_file, sep=r'\s+', header=None, na_values=['NaN', 'nan'])
        df = df.astype(float)

        act_ids = df.iloc[:, 1].values.astype(int)

        # interpolate missing sensor values
        sensor_df = df.iloc[:, SENSOR_COLS].copy()
        sensor_df = sensor_df.interpolate(method='linear', limit_direction='both')
        sensor_data = sensor_df.values.astype(np.float32)  # (T_orig, 6)

        # downsample 100 Hz → 50 Hz
        step         = SRC_FS // TARGET_FS            # = 2
        sensor_ds    = sensor_data[::step]
        act_ds       = act_ids[::step]

        for act_id, target_label in ACTIVITY_MAP.items():
            mask = act_ds == act_id
            for s, e in _contiguous_runs(mask):
                seg  = sensor_ds[s:e]
                wins = _slide(seg)
                if len(wins) == 0:
                    continue
                windows.append(wins)
                labels.extend([target_label] * len(wins))
                subjects.extend([subj_id] * len(wins))

    X        = np.concatenate(windows, axis=0).astype(np.float32)
    y        = np.array(labels,   dtype=np.int64)
    subjects = np.array(subjects, dtype=np.int64)
    return X, y, subjects
