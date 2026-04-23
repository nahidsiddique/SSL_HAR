# Reproduction Notes

This repository focuses on the experiments reported directly in the paper:

- balanced subject-wise SSL benchmark
- exact subject-wise SSL benchmark
- balanced subject-wise supervised baseline

## Outputs

The scripts write results into `output/` by default:

- `summary.csv`: compact metrics table
- `*.json`: per-method or per-seed details
- `checkpoints/`: saved weights
- `transition_reliability.csv`: transition vs stable reliability summary for SSL runs
- `transfer_summary.csv`: leave-one-dataset-out transfer summary for SSL runs
