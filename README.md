# SSL-HAR Reliability

Code for the IMU-based human activity recognition project using UCI-HAR, HHAR, PAMAP2, and MotionSense.


## Contents

- Subject-wise evaluation code for the two main setups:
  - balanced subject-wise split
  - exact subject-wise split
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

## Dataset

Download the raw datasets separately and point the scripts to these paths:

- `--ucihar-root`: extracted `UCI HAR Dataset/`
- `--hhar-root`: extracted `Activity recognition exp/`
- `--pamap2-root`: extracted `PAMAP2_Dataset/`
- `--motionsense-root`: extracted `DeviceMotion_data/`

## Dataset URL:

- Heterogeneity Human Activity Recognition (HHAR): https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition

- UCI Human Activity Recognition (UCIHAR): https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

- PAMAP2 Physical Activity Monitoring dataset (PAMAP2): https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

- MotionSense: https://github.com/mmalekzadeh/motion-sense/tree/master/data


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

- `ssl_har_reliability/`: models, datasets, training, metrics, and analysis code
- `scripts/`: runnable entry points for the main experiments
- `configs/`: ready-to-use experiment settings
- `docs/`: short notes on outputs and reproduction
- `notebooks/reference/`: notebook versions kept for reference

## Outputs

By default, experiment outputs are written to `output/`. This includes summary tables, JSON files, saved checkpoints, and transfer or transition summaries.

