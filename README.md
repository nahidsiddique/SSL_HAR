# SSL-HAR Reliability

Code for our IMU-based human activity recognition project using UCI-HAR, HHAR, PAMAP2, and MotionSense.

The goal of this repository is simple: keep the main experiments easy to understand and easy to run.

## Included

- Python package code under `ssl_har_reliability/`
- Subject-wise evaluation code for the two main setups:
  - balanced subject-wise split
  - exact subject-wise split
- Shared SSL and supervised training code
- Experiment runners in `scripts/`
- Reference notebooks in `notebooks/reference/`

## Not Included

- Raw datasets
- Paper manuscript sources
- Dataset download automation

## Main Experiments

The main commands are:

- `scripts/run_ssl_experiment.py`
  - `configs/paper/subjectwise_balanced.json`
  - `configs/paper/subjectwise_exact.json`
- `scripts/run_supervised_baseline.py`
  - `configs/paper/supervised_baseline_balanced.json`

Together these cover the two SSL settings and the balanced supervised baseline used in the project.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Layout

Download the raw datasets separately and point the scripts to these paths:

- `--ucihar-root`: extracted `UCI HAR Dataset/`
- `--hhar-root`: extracted `Activity recognition exp/`
- `--pamap2-root`: extracted `PAMAP2_Dataset/`
- `--motionsense-root`: extracted `DeviceMotion_data/`

The scripts expect the datasets to already be extracted in their original folder structures.

## Example Commands

Balanced subject-wise SSL:

```bash
python scripts/run_ssl_experiment.py \
  --config configs/paper/subjectwise_balanced.json \
  --ucihar-root /path/to/UCI\ HAR\ Dataset \
  --hhar-root /path/to/Activity\ recognition\ exp \
  --pamap2-root /path/to/PAMAP2_Dataset \
  --motionsense-root /path/to/DeviceMotion_data
```

Exact subject-wise SSL:

```bash
python scripts/run_ssl_experiment.py \
  --config configs/paper/subjectwise_exact.json \
  --ucihar-root /path/to/UCI\ HAR\ Dataset \
  --hhar-root /path/to/Activity\ recognition\ exp \
  --pamap2-root /path/to/PAMAP2_Dataset \
  --motionsense-root /path/to/DeviceMotion_data
```

Balanced supervised baseline:

```bash
python scripts/run_supervised_baseline.py \
  --config configs/paper/supervised_baseline_balanced.json \
  --ucihar-root /path/to/UCI\ HAR\ Dataset \
  --hhar-root /path/to/Activity\ recognition\ exp \
  --pamap2-root /path/to/PAMAP2_Dataset \
  --motionsense-root /path/to/DeviceMotion_data
```

## Repository Layout

- `ssl_har_reliability/`: models, datasets, training, metrics, and analysis code
- `scripts/`: runnable entry points for the main experiments
- `configs/paper/`: ready-to-use experiment settings
- `docs/`: short notes on outputs and reproduction
- `notebooks/reference/`: notebook versions kept for reference

## Outputs

By default, experiment outputs are written to `output/`. This includes summary tables, JSON files, saved checkpoints, and transfer or transition summaries.
