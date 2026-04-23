"""
MotionSense Dataset Loader
Root: DeviceMotion_data/
Folder structure: DeviceMotion_data/<act>_<trial>/sub_<N>.csv
  e.g. DeviceMotion_data/dws_1/sub_1.csv

CSV columns include:
  attitude.roll, attitude.pitch, attitude.yaw
  gravity.x, gravity.y, gravity.z
  rotationRate.x, rotationRate.y, rotationRate.z
  userAcceleration.x, userAcceleration.y, userAcceleration.z

We use userAcceleration.xyz + rotationRate.xyz  → 6 channels  at 50 Hz
Activities kept: dws, ups, sit, std, wlk  (jog dropped)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re

ACTIVITY_MAP = {
    'dws': 2,   # downstairs → downstairs
    'ups': 1,   # upstairs   → upstairs
    'sit': 3,   # sitting    → sitting
    'std': 4,   # standing   → standing
    'wlk': 0,   # walking    → walking
    # 'jog' dropped
}

ACC_COLS = ['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']
GYR_COLS = ['rotationRate.x',     'rotationRate.y',     'rotationRate.z']
NEEDED   = ACC_COLS + GYR_COLS

WINDOW_LEN = 128
STRIDE     = 64


def _slide(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    starts = list(range(0, n - WINDOW_LEN + 1, STRIDE))
    if not starts:
        return np.empty((0, WINDOW_LEN, 6), dtype=np.float32)
    return np.stack([data[s:s + WINDOW_LEN] for s in starts])


def load_motionsense(root: str) -> tuple:
    """
    Args:
        root: path to 'DeviceMotion_data/' directory

    Returns:
        X        : (N, 128, 6)  float32
        y        : (N,)         int64   [0..4]
        subjects : (N,)         int64
    """
    root = Path(root)

    windows, labels, subjects = [], [], []

    for act_dir in sorted(root.iterdir()):
        if not act_dir.is_dir():
            continue

        # folder name like 'dws_1', 'ups_11' — extract activity code
        folder_name = act_dir.name
        act_code    = re.split(r'[_\-]', folder_name)[0].lower()
        if act_code not in ACTIVITY_MAP:
            continue
        label = ACTIVITY_MAP[act_code]

        for csv_file in sorted(act_dir.glob('*.csv')):
            # subject ID from filename: sub_1.csv → 1, or 1.csv → 1
            digits = re.findall(r'\d+', csv_file.stem)
            if not digits:
                continue
            subj_id = int(digits[-1])

            df = pd.read_csv(csv_file)

            # handle slight column name variations (some files have index col)
            missing = [c for c in NEEDED if c not in df.columns]
            if missing:
                continue

            data = df[NEEDED].dropna().values.astype(np.float32)
            wins = _slide(data)
            if len(wins) == 0:
                continue

            windows.append(wins)
            labels.extend([label]   * len(wins))
            subjects.extend([subj_id] * len(wins))

    X        = np.concatenate(windows, axis=0).astype(np.float32)
    y        = np.array(labels,   dtype=np.int64)
    subjects = np.array(subjects, dtype=np.int64)
    return X, y, subjects
