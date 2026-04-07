import argparse
from pathlib import Path

import pandas as pd

from train import RESULTS_DIR, ensure_output_dirs, train_model


LR_VALUES = [0.1, 0.01, 0.001, 0.0001]
BATCH_SIZES = [8, 16, 32, 64, 128]


def save_summary(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved summary to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run KMNIST experiments.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()

    baseline_summary = train_model(
        run_name="baseline",
        lr=0.001,
        batch_size=64,
        epochs=args.epochs,
        loss_name="cross_entropy",
        seed=args.seed,
        device_name=args.device,
        data_dir=args.data_dir,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    loss_summary = train_model(
        run_name="loss_label_smoothing",
        lr=0.001,
        batch_size=64,
        epochs=args.epochs,
        loss_name="label_smoothing",
        seed=args.seed,
        device_name=args.device,
        data_dir=args.data_dir,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    lr_rows = []
    for lr in LR_VALUES:
        lr_rows.append(
            train_model(
                run_name=f"lr_{lr}",
                lr=lr,
                batch_size=64,
                epochs=args.epochs,
                loss_name="cross_entropy",
                seed=args.seed,
                device_name=args.device,
                data_dir=args.data_dir,
                val_split=args.val_split,
                num_workers=args.num_workers,
            )
        )
    save_summary(RESULTS_DIR / "lr_sweep_summary.csv", lr_rows)

    batch_rows = []
    for batch_size in BATCH_SIZES:
        batch_rows.append(
            train_model(
                run_name=f"batch_{batch_size}",
                lr=0.001,
                batch_size=batch_size,
                epochs=args.epochs,
                loss_name="cross_entropy",
                seed=args.seed,
                device_name=args.device,
                data_dir=args.data_dir,
                val_split=args.val_split,
                num_workers=args.num_workers,
            )
        )
    save_summary(RESULTS_DIR / "batch_sweep_summary.csv", batch_rows)

    all_rows = [baseline_summary, loss_summary, *lr_rows, *batch_rows]
    save_summary(RESULTS_DIR / "all_experiments_summary.csv", all_rows)


if __name__ == "__main__":
    main()
