from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from mnist_otdd.cross_dataset_study import (
    TensorDigitDataset,
    compute_distilled_gaps,
    compute_gap_matrix,
    ensure_dir,
    load_dataset_pair,
    plot_distilled_cross_eval,
    plot_distilled_gap,
    resolve_device,
    set_seed,
)
from mnist_otdd.resnet50_teacher_distill import (
    SoftLabelDataset,
    collect_teacher_statistics,
    relabel_dataset,
    synthesize_dataset,
    train_student,
    train_teacher,
)


def load_or_compute_gap_matrix(train_datasets, args, device, output_dir: Path) -> pd.DataFrame:
    gap_path = output_dir / "dataset_gap_matrix.csv"
    if gap_path.exists() and not args.force_recompute_gap:
        return pd.read_csv(gap_path)
    return compute_gap_matrix(
        train_datasets,
        args.gap_sample_size,
        args.num_projections,
        args.distance_batch_size,
        device,
        output_dir,
    )


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir))
    teacher_dir = ensure_dir(output_dir / "teacher_checkpoints")
    data_root = Path(args.data_root)

    train_datasets = {}
    test_datasets = {}
    for name in args.datasets:
        train_dataset, test_dataset = load_dataset_pair(name, data_root)
        train_datasets[name] = train_dataset
        test_datasets[name] = test_dataset

    teacher_paths: dict[str, Path] = {}
    for name in args.datasets:
        checkpoint_path = teacher_dir / f"{name}_resnet50_best_test.pt"
        metrics_path = teacher_dir / f"{name}_resnet50_metrics.csv"
        if checkpoint_path.exists() and not args.retrain_teachers:
            teacher_paths[name] = checkpoint_path
            print(f"[resume] reusing teacher checkpoint for {name}: {checkpoint_path}")
            if not metrics_path.exists():
                print(f"[resume] metrics missing for {name}; checkpoint will still be reused.")
            continue
        teacher_paths[name] = train_teacher(
            name,
            train_datasets[name],
            test_datasets[name],
            teacher_dir,
            args.teacher_epochs,
            args.teacher_batch_size,
            args.teacher_learning_rate,
            args.val_split,
            args.seed,
            device,
        )

    gap_df = load_or_compute_gap_matrix(train_datasets, args, device, output_dir)
    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    gap_weights = (1.0 / mean_gaps).to_dict()
    total_gap = sum(gap_weights.values())
    gap_weights = {k: v / total_gap for k, v in gap_weights.items()}

    teachers, prototypes, spreads, confidences = collect_teacher_statistics(
        train_datasets,
        teacher_paths,
        device,
        args.teacher_stat_sample_size,
    )
    conf_total = max(sum(confidences.values()), 1e-8)
    conf_weights = {k: v / conf_total for k, v in confidences.items()}
    teacher_weights = {
        name: args.gap_weight_ratio * gap_weights[name] + (1.0 - args.gap_weight_ratio) * conf_weights[name]
        for name in args.datasets
    }
    with open(output_dir / "teacher_weight_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "gap_weights": gap_weights,
                "confidence_weights": conf_weights,
                "teacher_weights": teacher_weights,
                "teacher_paths": {k: str(v) for k, v in teacher_paths.items()},
            },
            file,
            indent=2,
        )

    synth_path = output_dir / "synth_dataset.pt"
    if synth_path.exists() and not args.rebuild_synth:
        payload = torch.load(synth_path, map_location="cpu")
        synth_dataset = TensorDigitDataset(payload["images"], payload["labels"], "ResNet50TeacherSynth")
        print(f"[resume] reusing synthetic dataset: {synth_path}")
    else:
        synth_dataset = synthesize_dataset(
            teachers,
            prototypes,
            spreads,
            teacher_weights,
            args.ipc,
            args.recover_steps,
            args.recover_lr,
            args.feature_weight,
            args.entropy_weight,
            args.seed,
            device,
        )
        torch.save({"images": synth_dataset.images, "labels": synth_dataset.labels}, synth_path)

    relabeled_path = output_dir / "relabeled_dataset.pt"
    if relabeled_path.exists() and not args.relabel_synth:
        payload = torch.load(relabeled_path, map_location="cpu")
        relabeled = SoftLabelDataset(payload["images"], payload["hard_labels"], payload["soft_labels"])
        print(f"[resume] reusing relabeled dataset: {relabeled_path}")
    else:
        relabeled = relabel_dataset(synth_dataset, teachers, teacher_weights, args.temperature, args.student_batch_size, device)
        torch.save(
            {"images": relabeled.images, "hard_labels": relabeled.hard_labels, "soft_labels": relabeled.soft_labels},
            relabeled_path,
        )

    cross_eval = train_student(
        relabeled,
        test_datasets,
        args.student_epochs,
        args.student_batch_size,
        args.student_learning_rate,
        args.alpha,
        args.temperature,
        args.seed,
        device,
        output_dir,
    )
    distilled_dataset = TensorDigitDataset(relabeled.images, relabeled.hard_labels, "ResNet50TeacherSimpleCNNStudent")
    gap_to_real = compute_distilled_gaps(
        distilled_dataset,
        train_datasets,
        args.gap_sample_size,
        args.num_projections,
        args.distance_batch_size,
        device,
        output_dir,
    )
    plot_distilled_gap(gap_to_real, output_dir / "distilled_gap_bar.png")
    plot_distilled_cross_eval(cross_eval.to_dict("records"), output_dir / "distilled_cross_accuracy.png")
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume-friendly ResNet50 teacher distillation with SimpleCNN student.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-learning-rate", type=float, default=1e-4)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--student-batch-size", type=int, default=128)
    parser.add_argument("--student-learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--gap-sample-size", type=int, default=300)
    parser.add_argument("--num-projections", type=int, default=80)
    parser.add_argument("--distance-batch-size", type=int, default=64)
    parser.add_argument("--teacher-stat-sample-size", type=int, default=2000)
    parser.add_argument("--ipc", type=int, default=20)
    parser.add_argument("--recover-steps", type=int, default=200)
    parser.add_argument("--recover-lr", type=float, default=0.03)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--gap-weight-ratio", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain-teachers", action="store_true")
    parser.add_argument("--rebuild-synth", action="store_true")
    parser.add_argument("--relabel-synth", action="store_true")
    parser.add_argument("--force-recompute-gap", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
