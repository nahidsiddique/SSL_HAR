#!/usr/bin/env python3
"""Run the main SSL experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ssl_har_reliability.analysis import build_transition_reliability_table
from ssl_har_reliability.data import build_subjectwise_dataset, concatenate_unified_arrays
from ssl_har_reliability.experiment import (
    load_all_datasets,
    make_loader,
    run_ssl_method,
    run_transfer_eval_for_method,
    set_seed,
    write_json,
)


DEFAULTS = {
    "protocol": "balanced",
    "seed": 42,
    "methods": ["simclr", "tstcc", "tfc", "softclt"],
    "batch_size": 256,
    "pretrain_epochs": 100,
    "finetune_epochs": 50,
    "run_transition": True,
    "run_transfer": True,
    "output_dir": "output/ssl_experiment",
}

METHOD_LABELS = {
    "simclr": "SimCLR",
    "tstcc": "TS-TCC",
    "tfc": "TF-C",
    "softclt": "SoftCLT",
}


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method.lower(), method)


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_run_overview(cfg: dict) -> None:
    print_section("SSL Experiment")
    print(f"Protocol: {cfg['protocol']} subject-wise split")
    print(f"Methods: {', '.join(method_label(method) for method in cfg['methods'])}")
    print(f"Seed: {cfg['seed']}")
    print(f"Device: {cfg['device']}")
    print(f"Pretraining epochs: {cfg['pretrain_epochs']}")
    print(f"Fine-tuning epochs: {cfg['finetune_epochs']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Output directory: {cfg['output_dir']}")


def print_dataset_overview(datasets: dict) -> None:
    print_section("Loaded Datasets")
    for name, (X, _, subjects) in datasets.items():
        print(f"{name:12s} windows={len(X):6d}  subjects={len(set(subjects.tolist())):3d}")


def print_split_overview(split_summary: dict) -> None:
    print_section("Subject-Wise Split")
    print(
        f"train={split_summary['train_size']} "
        f"val={split_summary['val_size']} "
        f"test={split_summary['test_size']}"
    )
    print(
        f"subjects: train={split_summary['n_train_subjects']} "
        f"val={split_summary['n_val_subjects']} "
        f"test={split_summary['n_test_subjects']}"
    )


def print_method_summary(summary_rows: list[dict]) -> None:
    print("Summary")
    for row in summary_rows:
        print(
            f"  {method_label(row['method'])} {row['setting']}: "
            f"acc={row['acc']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"ece_raw={row['ece_raw']:.4f}, "
            f"ece_ts={row['ece_ts']:.4f}"
        )


def print_output_summary(output_dir: Path, wrote_transition: bool, wrote_transfer: bool) -> None:
    print_section("Saved Outputs")
    print(f"Main summary: {output_dir / 'summary.csv'}")
    print(f"Run config:    {output_dir / 'effective_config.json'}")
    print(f"Split summary: {output_dir / 'split_summary.json'}")
    print(f"Checkpoints:   {output_dir / 'checkpoints'}")
    if wrote_transition:
        print(f"Transitions:   {output_dir / 'transition_reliability.csv'}")
    if wrote_transfer:
        print(f"Transfer:      {output_dir / 'transfer_summary.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the self-supervised HAR experiments on a subject-wise split.",
    )
    parser.add_argument("--config", help="Path to a JSON config file.")
    parser.add_argument(
        "--protocol",
        choices=["balanced", "exact"],
        help="Subject-wise split protocol to run.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for split construction and training.")
    parser.add_argument(
        "--methods",
        nargs="+",
        help="One or more SSL methods: simclr, tstcc, tfc, softclt.",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for data loaders.")
    parser.add_argument("--pretrain-epochs", type=int, help="Number of SSL pretraining epochs.")
    parser.add_argument("--finetune-epochs", type=int, help="Number of supervised fine-tuning epochs.")
    parser.add_argument("--output-dir", help="Directory where results will be saved.")
    parser.add_argument("--device", help="Torch device string, for example 'cpu' or 'cuda'.")
    parser.add_argument("--ucihar-root", help="Path to the extracted UCI HAR Dataset directory.")
    parser.add_argument("--hhar-root", help="Path to the extracted HHAR Activity recognition exp directory.")
    parser.add_argument("--pamap2-root", help="Path to the extracted PAMAP2_Dataset directory.")
    parser.add_argument("--motionsense-root", help="Path to the extracted DeviceMotion_data directory.")
    parser.add_argument(
        "--run-transition",
        dest="run_transition",
        action="store_true",
        help="Compute transition-window reliability summaries.",
    )
    parser.add_argument(
        "--no-transition",
        dest="run_transition",
        action="store_false",
        help="Skip transition-window analysis.",
    )
    parser.add_argument(
        "--run-transfer",
        dest="run_transfer",
        action="store_true",
        help="Run leave-one-dataset-out transfer evaluation.",
    )
    parser.add_argument(
        "--no-transfer",
        dest="run_transfer",
        action="store_false",
        help="Skip transfer evaluation.",
    )
    parser.set_defaults(run_transition=None, run_transfer=None)
    return parser.parse_args()


def resolve_config(args):
    config = {}
    if args.config:
        config = json.loads(Path(args.config).read_text())

    merged = dict(DEFAULTS)
    merged.update(config)

    overrides = {
        "protocol": args.protocol,
        "seed": args.seed,
        "methods": args.methods,
        "batch_size": args.batch_size,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "output_dir": args.output_dir,
        "device": args.device,
        "run_transition": args.run_transition,
        "run_transfer": args.run_transfer,
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

    set_seed(int(cfg["seed"]))
    datasets = load_all_datasets(
        {
            "ucihar": cfg["ucihar_root"],
            "hhar": cfg["hhar_root"],
            "pamap2": cfg["pamap2_root"],
            "motionsense": cfg["motionsense_root"],
        }
    )
    print_dataset_overview(datasets)

    splits = build_subjectwise_dataset(
        datasets,
        protocol=cfg["protocol"],
        seed=int(cfg["seed"]),
    )

    train_loader = make_loader(splits["train"], int(cfg["batch_size"]), shuffle=True)
    val_loader = make_loader(splits["val"], int(cfg["batch_size"]), shuffle=False)
    test_loader = make_loader(splits["test"], int(cfg["batch_size"]), shuffle=False)

    split_summary = {
        "protocol": cfg["protocol"],
        "seed": int(cfg["seed"]),
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "n_train_subjects": int(len(set(splits["meta"]["train_subjects"]))),
        "n_val_subjects": int(len(set(splits["meta"]["val_subjects"]))),
        "n_test_subjects": int(len(set(splits["meta"]["test_subjects"]))),
    }
    write_json(output_dir / "split_summary.json", split_summary)
    print_split_overview(split_summary)

    summary_rows = []
    linear_outputs = {}

    for method in cfg["methods"]:
        print_section(f"Running {method_label(method)}")
        result = run_ssl_method(
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=cfg["device"],
            pretrain_epochs=int(cfg["pretrain_epochs"]),
            finetune_epochs=int(cfg["finetune_epochs"]),
        )
        summary_rows.extend(result["summary_rows"])
        linear_outputs[method] = result["linear"]
        print_method_summary(result["summary_rows"])
        write_json(output_dir / f"{method}_details.json", {
            "method": method,
            "summary_rows": result["summary_rows"],
            "pretrain_loss_history": result["pretrain_loss_history"],
        })
        torch.save(
            {
                "method": method,
                "protocol": cfg["protocol"],
                "seed": int(cfg["seed"]),
                "state_dict": result["model"].state_dict(),
            },
            checkpoints_dir / f"{method}_{cfg['protocol']}_seed{cfg['seed']}.pt",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    wrote_transition = False
    if cfg["run_transition"]:
        print_section("Transition Analysis")
        full_arrays = concatenate_unified_arrays(datasets)
        transition_df = build_transition_reliability_table(
            linear_outputs,
            full_arrays,
            splits["meta"]["test_idx"],
        )
        transition_df.to_csv(output_dir / "transition_reliability.csv", index=False)
        wrote_transition = True

    wrote_transfer = False
    if cfg["run_transfer"]:
        print_section("Transfer Evaluation")
        transfer_rows = []
        for method in cfg["methods"]:
            print(f"Evaluating transfer for {method_label(method)}")
            transfer_rows.extend(
                run_transfer_eval_for_method(
                    method=method,
                    datasets=datasets,
                    protocol=cfg["protocol"],
                    batch_size=int(cfg["batch_size"]),
                    device=cfg["device"],
                    pretrain_epochs=int(cfg["pretrain_epochs"]),
                    seed=int(cfg["seed"]),
                )
            )
        pd.DataFrame(transfer_rows).to_csv(output_dir / "transfer_summary.csv", index=False)
        wrote_transfer = True

    print_output_summary(output_dir, wrote_transition=wrote_transition, wrote_transfer=wrote_transfer)


if __name__ == "__main__":
    main()
