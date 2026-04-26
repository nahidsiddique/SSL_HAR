# SSL-HAR Reliability

This repository contains the code for examining how reliably SSL-pretrained encoders perform for IMU-based human activity recognition (HAR) when evaluated on unseen subjects, using four public datasets (UCI-HAR, HHAR, PAMAP2, and MotionSense) and evaluating models not only on classification accuracy but also on calibration (ECE, NLL, Brier score) and conformal coverage, comparing SSL pretraining against fully supervised baselines under two subject-wise split strategies. 


## Contents

- Subject-wise evaluation code for the two main setups:
  - balanced subject-wise split: equal subject counts per dataset per fold
  - exact subject-wise split: uses each dataset's original per-subject counts
- Shared SSL and supervised training code
- Experiment runners in `scripts/`
- Reference notebooks in `notebooks/`


## Experiments

The commands are:

- `scripts/run_ssl_experiment.py`
  - `configs/subjectwise_balanced.json`
  - `configs/subjectwise_exact.json`
- `scripts/run_supervised_baseline.py`
  - `configs/supervised_baseline_balanced.json`

Together these cover the two SSL settings and the balanced supervised baseline used in the project.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

Download each dataset and point the scripts to the extracted folder:

- **UCI Human Activity Recognition (UCI-HAR)** (`--ucihar-root`): https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Heterogeneity Human Activity Recognition (HHAR)** (`--hhar-root`): https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition
- **PAMAP2 Physical Activity Monitoring (PAMAP2)** (`--pamap2-root`): https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- **MotionSense** (`--motionsense-root`): https://github.com/mmalekzadeh/motion-sense/tree/master/data


## Commands

Balanced subject-wise SSL:

```bash
python scripts/run_ssl_experiment.py \
  --config configs/subjectwise_balanced.json \
  --ucihar-root /path/to/UCI\ HAR\ Dataset \
  --hhar-root /path/to/Activity\ recognition\ exp \
  --pamap2-root /path/to/PAMAP2_Dataset \
  --motionsense-root /path/to/DeviceMotion_data
```

Exact subject-wise SSL:

```bash
python scripts/run_ssl_experiment.py \
  --config configs/subjectwise_exact.json \
  --ucihar-root /path/to/UCI\ HAR\ Dataset \
  --hhar-root /path/to/Activity\ recognition\ exp \
  --pamap2-root /path/to/PAMAP2_Dataset \
  --motionsense-root /path/to/DeviceMotion_data
```

Balanced supervised baseline:

```bash
python scripts/run_supervised_baseline.py \
  --config configs/supervised_baseline_balanced.json \
  --ucihar-root /path/to/UCI\ HAR\ Dataset \
  --hhar-root /path/to/Activity\ recognition\ exp \
  --pamap2-root /path/to/PAMAP2_Dataset \
  --motionsense-root /path/to/DeviceMotion_data
```

## Repository Layout

- `ssl_har_reliability/`: models, data, training, metrics, analysis, and code
- `scripts/`: runnable entry points for the main experiments
- `configs/`: ready-to-use experiment settings
- `docs/`: short notes on outputs and reproduction
- `notebooks/`: notebook versions kept for reference

## Outputs

By default, experiment outputs are written to `output/`. This includes summary tables, JSON files, saved checkpoints, and transfer or transition summaries.

