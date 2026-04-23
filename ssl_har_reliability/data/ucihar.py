"""
UCI HAR Dataset Loader
Raw inertial signals: body_acc_xyz + body_gyro_xyz → (N, 128, 6)
Labels: 1=walk, 2=upstairs, 3=downstairs, 4=sit, 5=stand  (6=laying dropped)
"""

import numpy as np
from pathlib import Path

# UCI label → unified 5-class label
ACTIVITY_MAP = {
    1: 0,  # WALKING        → walking
    2: 1,  # WALKING_UPSTAIRS   → upstairs
    3: 2,  # WALKING_DOWNSTAIRS → downstairs
    4: 3,  # SITTING        → sitting
    5: 4,  # STANDING       → standing
    # 6: LAYING → dropped
}

CLASS_NAMES = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
               4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}


def _load_split(root: Path, split: str):
    sig_dir = root / split / 'Inertial Signals'
    channels = []
    for sensor in ['body_acc', 'body_gyro']:
        for axis in ['x', 'y', 'z']:
            fpath = sig_dir / f'{sensor}_{axis}_{split}.txt'
            channels.append(np.loadtxt(fpath))  # (N, 128)
    X = np.stack(channels, axis=-1)  # (N, 128, 6)
    y = np.loadtxt(root / split / f'y_{split}.txt', dtype=int)
    subjects = np.loadtxt(root / split / f'subject_{split}.txt', dtype=int)
    return X, y, subjects


def load_ucihar(root: str) -> tuple:
    """
    Args:
        root: path to 'UCI HAR Dataset/' directory

    Returns:
        X        : (N, 128, 6)  float32
        y        : (N,)         int64   [0..4]
        subjects : (N,)         int64   [1..30]
    """
    root = Path(root)
    X_tr, y_tr, s_tr = _load_split(root, 'train')
    X_te, y_te, s_te = _load_split(root, 'test')

    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    subjects = np.concatenate([s_tr, s_te], axis=0)

    # drop LAYING (label 6)
    mask = np.isin(y, list(ACTIVITY_MAP.keys()))
    X, y, subjects = X[mask], y[mask], subjects[mask]

    y = np.array([ACTIVITY_MAP[int(l)] for l in y], dtype=np.int64)
    return X.astype(np.float32), y, subjects.astype(np.int64)
