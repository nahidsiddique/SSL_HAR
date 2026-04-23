#!/usr/bin/env python3
"""Run the supervised baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ssl_har_reliability.data import build_subjectwise_dataset
from ssl_har_reliability.experiment import (
    load_all_datasets,
    make_loader,
    run_supervised_baseline,
    set_seed,
    write_json,
)


DEFAULTS = {
    "protocol": "balanced",
    "seeds": [7, 19, 42],
    "batch_size": 256,
    "epochs": 50,
    "output_dir": "output/supervised_baseline",
}


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_run_overview(cfg: dict) -> None:
    print_section("Supervised Baseline")
    print("Protocol: balanced subject-wise split")
    print(f"Seeds: {', '.join(str(seed) for seed in cfg['seeds'])}")
    print(f"Device: {cfg['device']}")
    print(f"Epochs: {cfg['epochs']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Output directory: {cfg['output_dir']}")


def print_dataset_overview(datasets: dict) -> None:
    print_section("Loaded Datasets")
    for name, (X, _, subjects) in datasets.items():
        print(f"{name:12s} windows={len(X):6d}  subjects={len(set(subjects.tolist())):3d}")


def print_seed_summary(seed: int, row: dict) -> None:
    print(
        f"Seed {seed}: "
        f"acc={row['acc']:.4f}, "
        f"macro_f1={row['macro_f1']:.4f}, "
        f"ece_raw={row['ece_raw']:.4f}, "
        f"ece_ts={row['ece_ts']:.4f}"
    )


def print_output_summary(output_dir: Path) -> None:
    print_section("Saved Outputs")
    print(f"Seed summary: {output_dir / 'summary.csv'}")
    print(f"Aggregate:    {output_dir / 'aggregate.csv'}")
    print(f"Run config:   {output_dir / 'effective_config.json'}")
    print(f"Checkpoints:  {output_dir / 'checkpoints'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the balanced subject-wise supervised baseline.",
    )
    parser.add_argument("--config", help="Path to a JSON config file.")
    parser.add_argument("--protocol", choices=["balanced"], help="Currently only the balanced split is supported.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seed values to evaluate.")
    parser.add_argument("--batch-size", type=int, help="Batch size for data loaders.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--output-dir", help="Directory where results will be saved.")
    parser.add_argument("--device", help="Torch device string, for example 'cpu' or 'cuda'.")
    parser.add_argument("--ucihar-root", help="Path to the extracted UCI HAR Dataset directory.")
    parser.add_argument("--hhar-root", help="Path to the extracted HHAR Activity recognition exp directory.")
    parser.add_argument("--pamap2-root", help="Path to the extracted PAMAP2_Dataset directory.")
    parser.add_argument("--motionsense-root", help="Path to the extracted DeviceMotion_data directory.")
    return parser.parse_args()


def resolve_config(args):
    config = {}
    if args.config:
        config = json.loads(Path(args.config).read_text())

    merged = dict(DEFAULTS)
    merged.update(config)
    overrides = {
        "protocol": args.protocol,
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "output_dir": args.output_dir,
        "device": args.device,
        "ucihar_root": args.ucihar_root,
        "hhar_root": args.hhar_root,
        "pamap2_root": args.pamap2_root,
        "motionsense_root": args.motionsense_root,
    }
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    required_roots = ["ucihar_root", "hhar_root", "pamap2_root", "motionsense_root"]
    missing = [key for key in required_roots if key not in merged or not merged[key]]
    if missing:
        missing_flags = ", ".join(f"--{key.replace('_', '-')}" for key in missing)
        raise SystemExit(f"Missing dataset paths. Please provide: {missing_flags}")
    if "device" not in merged:
        merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return merged


def main():
    cfg = resolve_config(parse_args())
    print_run_overview(cfg)

    output_dir = Path(cfg["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "effective_config.json", cfg)

    datasets = load_all_datasets(
        {
            "ucihar": cfg["ucihar_root"],
            "hhar": cfg["hhar_root"],
            "pamap2": cfg["pamap2_root"],
            "motionsense": cfg["motionsense_root"],
        }
    )
    print_dataset_overview(datasets)

    rows = []
    for seed in cfg["seeds"]:
        print_section(f"Running Seed {seed}")
        set_seed(int(seed))
        splits = build_subjectwise_dataset(
            datasets,
            protocol=cfg["protocol"],
            seed=int(seed),
        )
        train_loader = make_loader(splits["train"], int(cfg["batch_size"]), shuffle=True)
        val_loader = make_loader(splits["val"], int(cfg["batch_size"]), shuffle=False)
        test_loader = make_loader(splits["test"], int(cfg["batch_size"]), shuffle=False)

        result = run_supervised_baseline(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=cfg["device"],
            n_epochs=int(cfg["epochs"]),
        )
        metrics = result["metrics"]
        row = {
            "seed": int(seed),
            "acc": metrics["raw_acc"],
            "macro_f1": metrics["raw_f1"],
            "ece_raw": metrics["raw_ece"],
            "ece_ts": metrics["ts_ece"],
            "temperature": metrics["temperature"],
            "coverage": metrics["conformal"]["empirical_coverage"],
            "avg_set_size": metrics["conformal"]["avg_set_size"],
        }
        rows.append(row)
        print_seed_summary(int(seed), row)
        write_json(output_dir / f"seed_{seed}_details.json", {"seed": int(seed), "metrics": metrics})
        torch.save(
            {
                "seed": int(seed),
                "encoder_state_dict": result["encoder"].state_dict(),
                "head_state_dict": result["head"].state_dict(),
            },
            checkpoints_dir / f"supervised_baseline_seed_{seed}.pt",
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    aggregate = summary_df.agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    print_output_summary(output_dir)


if __name__ == "__main__":
    main()
