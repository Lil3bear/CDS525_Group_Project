from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from mnist_otdd.resnet50_moe_proto_distill import load_teacher, to_resnet_input


class SoftLabelDataset(Dataset):
    def __init__(self, images: torch.Tensor, hard_labels: torch.Tensor, soft_labels: torch.Tensor):
        self.images = images.float()
        self.hard_labels = hard_labels.long()
        self.soft_labels = soft_labels.float()

    def __len__(self) -> int:
        return len(self.hard_labels)

    def __getitem__(self, index: int):
        return self.images[index], self.hard_labels[index], self.soft_labels[index]


def allocate_quotas(weights: dict[str, float], total: int) -> dict[str, int]:
    raw = {name: weights[name] * total for name in weights}
    quotas = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(weights, key=lambda name: raw[name] - quotas[name], reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


@torch.no_grad()
def collect_embeddings(
    teacher,
    train_datasets: dict[str, Dataset],
    sample_size_per_dataset: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for idx, (name, dataset) in enumerate(train_datasets.items()):
        images, labels = materialize_dataset(dataset)
        if len(labels) > sample_size_per_dataset:
            chosen = torch.randperm(len(labels), generator=torch.Generator().manual_seed(100 + idx))[:sample_size_per_dataset]
            images = images[chosen]
            labels = labels[chosen]
        rows = []
        probs = []
        for start in range(0, len(labels), batch_size):
            batch_images = images[start : start + batch_size].to(device)
            outputs = teacher(to_resnet_input(batch_images), return_features=True)
            rows.append(outputs["embedding"].cpu())
            probs.append(F.softmax(outputs["logits"], dim=1).cpu())
        result[name] = {
            "images": images.cpu(),
            "labels": labels.cpu(),
            "embeddings": torch.cat(rows, dim=0),
            "probs": torch.cat(probs, dim=0),
        }
    return result


def build_weighted_prototypes(
    collected: dict[str, dict[str, torch.Tensor]],
    domain_weights: dict[str, float],
) -> dict[int, torch.Tensor]:
    prototypes: dict[int, torch.Tensor] = {}
    for label in range(10):
        parts = []
        for name, payload in collected.items():
            mask = payload["labels"] == label
            if mask.sum() == 0:
                continue
            parts.append(payload["embeddings"][mask] * domain_weights[name])
        prototypes[label] = torch.cat(parts, dim=0).mean(dim=0)
    return prototypes


def select_high_fidelity_exemplars(
    collected: dict[str, dict[str, torch.Tensor]],
    domain_weights: dict[str, float],
    prototypes: dict[int, torch.Tensor],
    images_per_class: int,
    confidence_weight: float,
) -> tuple[TensorDigitDataset, pd.DataFrame]:
    quotas = allocate_quotas(domain_weights, images_per_class)
    selected_images = []
    selected_labels = []
    provenance_rows = []

    for label in range(10):
        proto = prototypes[label]
        for domain_name, payload in collected.items():
            mask = payload["labels"] == label
            if mask.sum() == 0:
                continue
            domain_images = payload["images"][mask]
            domain_embeddings = payload["embeddings"][mask]
            domain_probs = payload["probs"][mask, label]
            distances = (domain_embeddings - proto).pow(2).mean(dim=1)
            scores = -distances + confidence_weight * domain_probs
            quota = min(quotas[domain_name], len(domain_images))
            chosen = torch.topk(scores, k=quota).indices
            selected_images.append(domain_images[chosen])
            selected_labels.append(torch.full((quota,), label, dtype=torch.long))
            for idx in chosen.tolist():
                provenance_rows.append(
                    {
                        "domain": domain_name,
                        "label": label,
                        "score": float(scores[idx]),
                        "distance": float(distances[idx]),
                        "teacher_confidence": float(domain_probs[idx]),
                    }
                )

    images = torch.cat(selected_images, dim=0)
    labels = torch.cat(selected_labels, dim=0)
    provenance = pd.DataFrame(provenance_rows)
    return TensorDigitDataset(images, labels, "MoEHighFidelityExemplars"), provenance


@torch.no_grad()
def relabel_with_teacher(
    teacher,
    dataset: TensorDigitDataset,
    temperature: float,
    batch_size: int,
    device: torch.device,
) -> SoftLabelDataset:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out_images = []
    out_hard = []
    out_soft = []
    for images, hard_labels in loader:
        outputs = teacher(to_resnet_input(images.to(device)), return_features=True)
        logits = outputs["logits"] / temperature
        soft = F.softmax(logits, dim=1).cpu()
        out_images.append(images.cpu())
        out_hard.append(hard_labels.cpu())
        out_soft.append(soft)
    return SoftLabelDataset(torch.cat(out_images), torch.cat(out_hard), torch.cat(out_soft))


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
        print(f"[moe-exemplar-student] epoch={epoch}/{epochs} train_acc={row['train_accuracy']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "student_metrics.csv", index=False)
    cross_eval_df = pd.DataFrame(evaluate_model_on_all(student, "MoEExemplarStudent", test_datasets, batch_size=batch_size, device=device))
    cross_eval_df.to_csv(output_dir / "student_cross_eval.csv", index=False)
    torch.save({"model_state_dict": student.state_dict()}, output_dir / "student_last.pt")
    return metrics_df, cross_eval_df


def save_preview(dataset: TensorDigitDataset, output_path: Path) -> None:
    images = denormalize_tensor(dataset.images[:100]).numpy()
    labels = dataset.labels[:100].numpy()
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(str(int(labels[idx])), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_path)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir))
    teacher_ckpt = Path(args.teacher_checkpoint)
    teacher = load_teacher(teacher_ckpt, device)

    train_datasets = {}
    test_datasets = {}
    for name in args.datasets:
        train_dataset, test_dataset = load_dataset_pair(name, Path(args.data_root))
        train_datasets[name] = train_dataset
        test_datasets[name] = test_dataset

    gap_df = compute_gap_matrix(train_datasets, args.gap_sample_size, args.num_projections, args.distance_batch_size, device, output_dir)
    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    inv_gap = (1.0 / mean_gaps).to_dict()
    total = sum(inv_gap.values())
    domain_weights = {name: inv_gap[name] / total for name in args.datasets}

    collected = collect_embeddings(teacher, train_datasets, args.prototype_sample_size, args.batch_size, device)
    prototypes = build_weighted_prototypes(collected, domain_weights)
    exemplar_dataset, provenance = select_high_fidelity_exemplars(
        collected,
        domain_weights,
        prototypes,
        args.images_per_class,
        args.confidence_weight,
    )
    provenance.to_csv(output_dir / "selected_exemplar_provenance.csv", index=False)
    torch.save({"images": exemplar_dataset.images, "labels": exemplar_dataset.labels}, output_dir / "exemplar_distilled_dataset.pt")
    save_preview(exemplar_dataset, output_dir / "exemplar_preview.png")

    relabeled = relabel_with_teacher(teacher, exemplar_dataset, args.temperature, args.batch_size, device)
    torch.save(
        {"images": relabeled.images, "hard_labels": relabeled.hard_labels, "soft_labels": relabeled.soft_labels},
        output_dir / "exemplar_relabeled_dataset.pt",
    )

    _, cross_eval = train_student(
        relabeled,
        test_datasets,
        args.student_epochs,
        args.batch_size,
        args.student_learning_rate,
        args.alpha,
        args.temperature,
        args.seed,
        device,
        output_dir,
    )
    gap_to_real = compute_distilled_gaps(
        TensorDigitDataset(relabeled.images, relabeled.hard_labels, "MoEHighFidelityExemplars"),
        train_datasets,
        args.gap_sample_size,
        args.num_projections,
        args.distance_batch_size,
        device,
        output_dir,
    )
    plot_distilled_gap(gap_to_real, output_dir / "exemplar_gap_bar.png")
    plot_distilled_cross_eval(cross_eval.to_dict("records"), output_dir / "exemplar_cross_accuracy.png")

    summary = {
        "teacher_checkpoint": str(teacher_ckpt),
        "student_mean_cross_accuracy": float(cross_eval["test_accuracy"].mean()),
        "student_min_cross_accuracy": float(cross_eval["test_accuracy"].min()),
        "student_max_cross_accuracy": float(cross_eval["test_accuracy"].max()),
        "images_per_class": args.images_per_class,
        "domain_weights": domain_weights,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-fidelity MoE exemplar distillation.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--student-learning-rate", type=float, default=1e-3)
    parser.add_argument("--prototype-sample-size", type=int, default=5000)
    parser.add_argument("--images-per-class", type=int, default=100)
    parser.add_argument("--confidence-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gap-sample-size", type=int, default=300)
    parser.add_argument("--num-projections", type=int, default=80)
    parser.add_argument("--distance-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
