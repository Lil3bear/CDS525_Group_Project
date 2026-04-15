from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def build_selected_fusion_dataset(
    train_datasets: dict[str, Dataset],
    gap_df: pd.DataFrame,
    ipc: int,
    source_sample_size: int,
    seed: int,
    output_dir: Path,
) -> TensorDigitDataset:
    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    centrality = (1.0 / mean_gaps).to_dict()
    total_weight = sum(centrality.values())
    normalized_weights = {name: value / total_weight for name, value in centrality.items()}

    dataset_images = {}
    dataset_labels = {}
    for name, dataset in train_datasets.items():
        images, labels = materialize_dataset(dataset)
        if len(labels) > source_sample_size:
            generator = torch.Generator().manual_seed(seed + len(name))
            subset, _ = torch.utils.data.random_split(dataset, [source_sample_size, len(dataset) - source_sample_size], generator=generator)
            images, labels = materialize_dataset(subset)
        dataset_images[name] = images
        dataset_labels[name] = labels

    selected_images = []
    selected_labels = []
    provenance = []
    for label in range(10):
        class_images_all = []
        class_weights = []
        for name in train_datasets:
            class_images = dataset_images[name][dataset_labels[name] == label]
            if len(class_images) == 0:
                continue
            keep = max(1, int(round(ipc * len(train_datasets) * normalized_weights[name])))
            if len(class_images) > keep:
                generator = torch.Generator().manual_seed(seed + label * 31 + len(name))
                perm = torch.randperm(len(class_images), generator=generator)[:keep]
                class_images = class_images[perm]
            class_images_all.append(class_images)
            class_weights.extend([normalized_weights[name]] * len(class_images))
            provenance.append({"dataset": name, "label": label, "count": int(len(class_images)), "weight": normalized_weights[name]})
        pool = torch.cat(class_images_all, dim=0)
        flat = denormalize_tensor(pool).view(len(pool), -1).cpu().numpy()
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=ipc, random_state=seed, n_init=10).fit(flat)
        centers = kmeans.cluster_centers_
        chosen = []
        for center in centers:
            distances = np.linalg.norm(flat - center[None, :], axis=1)
            candidate = int(np.argmin(distances))
            if candidate in chosen:
                order = np.argsort(distances)
                for fallback in order:
                    if int(fallback) not in chosen:
                        candidate = int(fallback)
                        break
            chosen.append(candidate)
        selected_images.append(pool[chosen].cpu())
        selected_labels.append(torch.full((ipc,), label, dtype=torch.long))

    pd.DataFrame(provenance).to_csv(output_dir / "selected_fusion_provenance.csv", index=False)
    return TensorDigitDataset(torch.cat(selected_images, dim=0), torch.cat(selected_labels, dim=0), "SelectedFusionDistilled")


@torch.no_grad()
def relabel_with_teacher_ensemble(
    distilled_dataset: TensorDigitDataset,
    teacher_paths: dict[str, Path],
    teacher_weights: dict[str, float],
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> SoftLabelDataset:
    teachers = {name: load_teacher(path, device) for name, path in teacher_paths.items()}
    loader = DataLoader(distilled_dataset, batch_size=batch_size, shuffle=False)
    image_batches = []
    hard_batches = []
    soft_batches = []
    for images, hard_labels in loader:
        images = images.to(device)
        logits_sum = None
        for name, teacher in teachers.items():
            logits = teacher(images) / temperature
            weighted = logits * teacher_weights[name]
            logits_sum = weighted if logits_sum is None else logits_sum + weighted
        soft_labels = F.softmax(logits_sum, dim=1).cpu()
        image_batches.append(images.cpu())
        hard_batches.append(hard_labels.cpu())
        soft_batches.append(soft_labels)
    return SoftLabelDataset(torch.cat(image_batches), torch.cat(hard_batches), torch.cat(soft_batches))


def train_student_on_soft_labels(
    dataset: SoftLabelDataset,
    test_datasets: dict[str, Dataset],
    device: torch.device,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    alpha: float,
    seed: int,
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
        total_examples = 0
        total_correct = 0
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
            total_examples += batch_size_local
            total_correct += (logits.argmax(dim=1) == hard_labels).sum().item()

        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "train_accuracy": total_correct / total_examples,
        }
        loss_fn = nn.CrossEntropyLoss()
        for test_name, test_dataset in test_datasets.items():
            test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)
            _, test_accuracy = evaluate(student, test_loader, loss_fn, device)
            row[f"test_accuracy_{test_name}"] = test_accuracy
        rows.append(row)
        print(f"[scdd-student] epoch={epoch}/{epochs} train_acc={row['train_accuracy']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "scdd_student_metrics.csv", index=False)
    cross_eval = pd.DataFrame(
        evaluate_model_on_all(student, "SCDDStyleDistilled", test_datasets, batch_size=batch_size, device=device)
    )
    cross_eval.to_csv(output_dir / "scdd_student_cross_eval.csv", index=False)
    return metrics_df, cross_eval


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

    gap_df = compute_gap_matrix(
        train_datasets=train_datasets,
        gap_sample_size=args.gap_sample_size,
        num_projections=args.num_projections,
        batch_size=args.distance_batch_size,
        device=device,
        output_dir=output_dir,
    )

    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    teacher_weights_raw = (1.0 / mean_gaps).to_dict()
    total = sum(teacher_weights_raw.values())
    teacher_weights = {name: value / total for name, value in teacher_weights_raw.items()}

    teacher_paths = {
        name: Path(args.teacher_root) / f"{name}_best.pt"
        for name in args.datasets
    }
    distilled_dataset = build_selected_fusion_dataset(
        train_datasets=train_datasets,
        gap_df=gap_df,
        ipc=args.distill_ipc,
        source_sample_size=args.distill_source_sample_size,
        seed=args.seed,
        output_dir=output_dir,
    )
    soft_dataset = relabel_with_teacher_ensemble(
        distilled_dataset=distilled_dataset,
        teacher_paths=teacher_paths,
        teacher_weights=teacher_weights,
        device=device,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
    torch.save(
        {
            "images": soft_dataset.images,
            "hard_labels": soft_dataset.hard_labels,
            "soft_labels": soft_dataset.soft_labels,
            "teacher_weights": teacher_weights,
        },
        output_dir / "scdd_soft_distilled_dataset.pt",
    )

    metrics_df, cross_eval_df = train_student_on_soft_labels(
        dataset=soft_dataset,
        test_datasets=test_datasets,
        device=device,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        alpha=args.alpha,
        seed=args.seed,
    )
    gap_to_real = compute_distilled_gaps(
        distilled_dataset=TensorDigitDataset(soft_dataset.images, soft_dataset.hard_labels, "SCDDStyleDistilled"),
        real_train_datasets=train_datasets,
        gap_sample_size=args.gap_sample_size,
        num_projections=args.num_projections,
        batch_size=args.distance_batch_size,
        device=device,
        output_dir=output_dir,
    )
    plot_distilled_gap(gap_to_real, output_dir / "scdd_distilled_gap_bar.png")
    plot_distilled_cross_eval(cross_eval_df.to_dict("records"), output_dir / "scdd_distilled_cross_accuracy.png")
    with open(output_dir / "scdd_config.json", "w", encoding="utf-8") as file:
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
                "distill_ipc": args.distill_ipc,
                "distill_source_sample_size": args.distill_source_sample_size,
                "gap_sample_size": args.gap_sample_size,
                "num_projections": args.num_projections,
            },
            file,
            indent=2,
        )
    metrics_df.to_csv(output_dir / "scdd_epoch_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCDD-style multi-teacher relabel distillation for MNIST-family datasets.")
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
    parser.add_argument("--distill-ipc", type=int, default=20)
    parser.add_argument("--distill-source-sample-size", type=int, default=1500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
