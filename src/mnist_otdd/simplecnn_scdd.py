from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from kmnist_oss.model import SimpleCNN
from mnist_otdd.cross_dataset_study import (
    TensorDigitDataset,
    build_loader,
    compute_distilled_gaps,
    compute_gap_matrix,
    denormalize_tensor,
    ensure_dir,
    evaluate,
    evaluate_model_on_all,
    load_dataset_pair,
    materialize_dataset,
    plot_distilled_cross_eval,
    plot_distilled_gap,
    resolve_device,
    save_figure,
    set_seed,
)


class SoftLabelDataset(Dataset):
    def __init__(self, images: torch.Tensor, hard_labels: torch.Tensor, soft_labels: torch.Tensor):
        self.images = images.float()
        self.hard_labels = hard_labels.long()
        self.soft_labels = soft_labels.float()

    def __len__(self) -> int:
        return len(self.hard_labels)

    def __getitem__(self, index: int):
        return self.images[index], self.hard_labels[index], self.soft_labels[index]


def load_teacher(model_path: Path, device: torch.device) -> SimpleCNN:
    model = SimpleCNN().to(device)
    payload = torch.load(model_path, map_location=device)
    state_dict = payload["model_state_dict"]
    remapped = {}
    for key, value in state_dict.items():
        if key == "classifier.1.weight":
            remapped["embedding.weight"] = value
        elif key == "classifier.1.bias":
            remapped["embedding.bias"] = value
        elif key == "classifier.4.weight":
            remapped["output.weight"] = value
        elif key == "classifier.4.bias":
            remapped["output.bias"] = value
        else:
            remapped[key] = value
    model.load_state_dict(remapped, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def collect_teacher_statistics(
    train_datasets: dict[str, Dataset],
    teacher_paths: dict[str, Path],
    device: torch.device,
    sample_size_per_dataset: int,
) -> tuple[dict[str, SimpleCNN], dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, float]]:
    teachers = {name: load_teacher(path, device) for name, path in teacher_paths.items()}
    embeddings_by_class = {label: [] for label in range(10)}
    confidences = {name: 0.0 for name in teachers}
    counts = {name: 0 for name in teachers}

    for name, dataset in train_datasets.items():
        images, labels = materialize_dataset(dataset)
        if len(labels) > sample_size_per_dataset:
            generator = torch.Generator().manual_seed(42 + len(name))
            chosen = torch.randperm(len(labels), generator=generator)[:sample_size_per_dataset]
            images = images[chosen]
            labels = labels[chosen]
        images = images.to(device)
        labels = labels.to(device)
        teacher = teachers[name]
        feature_dict = teacher(images, return_features=True)
        logits = feature_dict["logits"]
        probs = F.softmax(logits, dim=1)
        confidences[name] = probs.max(dim=1).values.mean().item()
        counts[name] = len(labels)
        embeddings = feature_dict["embedding"].detach().cpu()
        for label in range(10):
            class_embeddings = embeddings[labels.cpu() == label]
            if len(class_embeddings) > 0:
                embeddings_by_class[label].append(class_embeddings)

    prototypes = {}
    spreads = {}
    for label in range(10):
        class_embed = torch.cat(embeddings_by_class[label], dim=0)
        prototypes[label] = class_embed.mean(dim=0)
        spreads[label] = class_embed.std(dim=0).clamp_min(1e-3)
    return teachers, prototypes, spreads, confidences


def synthesize_dataset(
    teachers: dict[str, SimpleCNN],
    prototypes: dict[int, torch.Tensor],
    spreads: dict[int, torch.Tensor],
    teacher_weights: dict[str, float],
    ipc: int,
    steps: int,
    lr: float,
    feature_weight: float,
    entropy_weight: float,
    seed: int,
    device: torch.device,
) -> TensorDigitDataset:
    set_seed(seed)
    synth_images = []
    synth_labels = []
    teacher_names = list(teachers)
    for label in range(10):
        inputs = torch.randn(ipc, 1, 28, 28, device=device, requires_grad=True)
        optimizer = optim.Adam([inputs], lr=lr)
        target_embed = prototypes[label].to(device)
        target_spread = spreads[label].to(device)
        for _ in range(steps):
            optimizer.zero_grad()
            clamped_inputs = torch.tanh(inputs)
            logits_sum = None
            embed_loss = 0.0
            entropy_loss = 0.0
            for name in teacher_names:
                teacher = teachers[name]
                feature_dict = teacher(clamped_inputs, return_features=True)
                logits = feature_dict["logits"]
                embeddings = feature_dict["embedding"]
                weight = teacher_weights[name]
                logits_sum = logits * weight if logits_sum is None else logits_sum + logits * weight
                embed_loss = embed_loss + weight * (
                    F.mse_loss(embeddings.mean(dim=0), target_embed)
                    + 0.2 * F.mse_loss(embeddings.std(dim=0), target_spread)
                )
                probs = F.softmax(logits, dim=1)
                entropy_loss = entropy_loss + weight * (-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean())
            class_loss = F.cross_entropy(logits_sum, torch.full((ipc,), label, device=device, dtype=torch.long))
            variation_loss = (
                (clamped_inputs[:, :, 1:, :] - clamped_inputs[:, :, :-1, :]).pow(2).mean()
                + (clamped_inputs[:, :, :, 1:] - clamped_inputs[:, :, :, :-1]).pow(2).mean()
            )
            loss = class_loss + feature_weight * embed_loss + 1e-3 * variation_loss + entropy_weight * entropy_loss
            loss.backward()
            optimizer.step()
        final_images = ((torch.tanh(inputs).detach().cpu()) + 1.0) / 2.0
        final_images = (final_images - 0.5) / 0.5
        synth_images.append(final_images)
        synth_labels.append(torch.full((ipc,), label, dtype=torch.long))
    return TensorDigitDataset(torch.cat(synth_images, dim=0), torch.cat(synth_labels, dim=0), "SimpleCNNSynth")


@torch.no_grad()
def relabel_synthesized_dataset(
    dataset: TensorDigitDataset,
    teachers: dict[str, SimpleCNN],
    teacher_weights: dict[str, float],
    temperature: float,
    batch_size: int,
    device: torch.device,
) -> SoftLabelDataset:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images_out = []
    hard_labels_out = []
    soft_labels_out = []
    for images, hard_labels in loader:
        images = images.to(device)
        logits_sum = None
        for name, teacher in teachers.items():
            logits = teacher(images) / temperature
            weighted = logits * teacher_weights[name]
            logits_sum = weighted if logits_sum is None else logits_sum + weighted
        soft_labels = F.softmax(logits_sum, dim=1).cpu()
        images_out.append(images.cpu())
        hard_labels_out.append(hard_labels.cpu())
        soft_labels_out.append(soft_labels)
    return SoftLabelDataset(torch.cat(images_out), torch.cat(hard_labels_out), torch.cat(soft_labels_out))


def train_student(
    dataset: SoftLabelDataset,
    test_datasets: dict[str, Dataset],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    alpha: float,
    temperature: float,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    student = SimpleCNN().to(device)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    rows = []
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for images, hard_labels, soft_labels in train_loader:
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)
            optimizer.zero_grad()
            logits = student(images)
            kd = kl_loss(F.log_softmax(logits / temperature, dim=1), soft_labels) * (temperature ** 2)
            ce = ce_loss(logits, hard_labels)
            loss = alpha * kd + (1.0 - alpha) * ce
            loss.backward()
            optimizer.step()
            batch_size_local = hard_labels.size(0)
            total_loss += loss.item() * batch_size_local
            total_correct += (logits.argmax(dim=1) == hard_labels).sum().item()
            total_examples += batch_size_local
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "train_accuracy": total_correct / total_examples,
        }
        for test_name, dataset_test in test_datasets.items():
            test_loader = build_loader(dataset_test, batch_size=batch_size, shuffle=False, seed=seed)
            _, accuracy = evaluate(student, test_loader, ce_loss, device)
            row[f"test_accuracy_{test_name}"] = accuracy
        rows.append(row)
        print(f"[simplecnn-scdd] epoch={epoch}/{epochs} train_acc={row['train_accuracy']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "student_metrics.csv", index=False)
    cross_eval_df = pd.DataFrame(evaluate_model_on_all(student, "SimpleCNNSCDD", test_datasets, batch_size=batch_size, device=device))
    cross_eval_df.to_csv(output_dir / "student_cross_eval.csv", index=False)
    return metrics_df, cross_eval_df


def save_synth_preview(dataset: TensorDigitDataset, output_path: Path) -> None:
    count = min(100, len(dataset.images))
    cols = 10
    rows = max(1, int(np.ceil(count / cols)))
    images = denormalize_tensor(dataset.images[:count]).numpy()
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx, axis in enumerate(axes.flat):
        if idx < count:
            axis.imshow(images[idx, 0], cmap="gray")
            axis.set_title(str(int(dataset.labels[idx])), fontsize=8)
        axis.axis("off")
    save_figure(fig, output_path)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir))

    train_datasets = {}
    test_datasets = {}
    for name in args.datasets:
        train_dataset, test_dataset = load_dataset_pair(name, Path(args.data_root))
        train_datasets[name] = train_dataset
        test_datasets[name] = test_dataset

    teacher_paths = {name: Path(args.teacher_root) / f"{name}_best.pt" for name in args.datasets}
    gap_df = compute_gap_matrix(
        train_datasets=train_datasets,
        gap_sample_size=args.gap_sample_size,
        num_projections=args.num_projections,
        batch_size=args.distance_batch_size,
        device=device,
        output_dir=output_dir,
    )
    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    gap_weights = (1.0 / mean_gaps).to_dict()
    total_gap_weight = sum(gap_weights.values())
    gap_weights = {name: value / total_gap_weight for name, value in gap_weights.items()}

    teachers, prototypes, spreads, confidences = collect_teacher_statistics(
        train_datasets=train_datasets,
        teacher_paths=teacher_paths,
        device=device,
        sample_size_per_dataset=args.teacher_stat_sample_size,
    )
    confidence_total = sum(confidences.values())
    confidence_weights = {name: value / confidence_total for name, value in confidences.items()}
    teacher_weights = {
        name: args.gap_weight_ratio * gap_weights[name] + (1.0 - args.gap_weight_ratio) * confidence_weights[name]
        for name in args.datasets
    }

    synth_dataset = synthesize_dataset(
        teachers=teachers,
        prototypes=prototypes,
        spreads=spreads,
        teacher_weights=teacher_weights,
        ipc=args.ipc,
        steps=args.recover_steps,
        lr=args.recover_lr,
        feature_weight=args.feature_weight,
        entropy_weight=args.entropy_weight,
        seed=args.seed,
        device=device,
    )
    torch.save({"images": synth_dataset.images, "labels": synth_dataset.labels}, output_dir / "synth_dataset.pt")
    save_synth_preview(synth_dataset, output_dir / "synth_dataset_preview.png")

    relabeled_dataset = relabel_synthesized_dataset(
        dataset=synth_dataset,
        teachers=teachers,
        teacher_weights=teacher_weights,
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=device,
    )
    torch.save(
        {
            "images": relabeled_dataset.images,
            "hard_labels": relabeled_dataset.hard_labels,
            "soft_labels": relabeled_dataset.soft_labels,
            "teacher_weights": teacher_weights,
        },
        output_dir / "relabeled_dataset.pt",
    )

    metrics_df, cross_eval_df = train_student(
        dataset=relabeled_dataset,
        test_datasets=test_datasets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        temperature=args.temperature,
        seed=args.seed,
        device=device,
        output_dir=output_dir,
    )

    distilled_gap_df = compute_distilled_gaps(
        distilled_dataset=TensorDigitDataset(relabeled_dataset.images, relabeled_dataset.hard_labels, "SimpleCNNSCDD"),
        real_train_datasets=train_datasets,
        gap_sample_size=args.gap_sample_size,
        num_projections=args.num_projections,
        batch_size=args.distance_batch_size,
        device=device,
        output_dir=output_dir,
    )
    plot_distilled_gap(distilled_gap_df, output_dir / "simplecnn_scdd_gap_bar.png")
    plot_distilled_cross_eval(cross_eval_df.to_dict("records"), output_dir / "simplecnn_scdd_cross_accuracy.png")

    with open(output_dir / "simplecnn_scdd_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "datasets": args.datasets,
                "teacher_root": args.teacher_root,
                "teacher_weights": teacher_weights,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "ipc": args.ipc,
                "recover_steps": args.recover_steps,
                "recover_lr": args.recover_lr,
                "feature_weight": args.feature_weight,
                "entropy_weight": args.entropy_weight,
                "gap_weight_ratio": args.gap_weight_ratio,
            },
            file,
            indent=2,
        )
    metrics_df.to_csv(output_dir / "simplecnn_scdd_epoch_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimpleCNN-only SCDD-style synthesis and distillation.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--teacher-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gap-sample-size", type=int, default=1200)
    parser.add_argument("--num-projections", type=int, default=200)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--teacher-stat-sample-size", type=int, default=1500)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--recover-steps", type=int, default=150)
    parser.add_argument("--recover-lr", type=float, default=0.05)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--gap-weight-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
