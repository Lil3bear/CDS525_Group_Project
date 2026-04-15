from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.transforms import functional as TF

from kmnist_oss.model import SimpleCNN
from mnist_otdd.cross_dataset_study import (
    TensorDigitDataset,
    build_loader,
    compute_distilled_gaps,
    compute_gap_matrix,
    ensure_dir,
    evaluate_model_on_all,
    load_dataset_pair,
    materialize_dataset,
    plot_distilled_cross_eval,
    plot_distilled_gap,
    resolve_device,
    set_seed,
    split_train_val,
)


IMAGENET_MEAN = ResNet50_Weights.IMAGENET1K_V2.transforms().mean
IMAGENET_STD = ResNet50_Weights.IMAGENET1K_V2.transforms().std


class DomainLabeledDataset(Dataset):
    def __init__(self, base_dataset: Dataset, domain_index: int):
        self.base_dataset = base_dataset
        self.domain_index = domain_index

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        return image, label, self.domain_index


def to_resnet_input(images: torch.Tensor) -> torch.Tensor:
    images = (images * 0.5) + 0.5
    images = images.repeat(1, 3, 1, 1)
    images = TF.resize(images, [224, 224])
    images = TF.normalize(images, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return images


class SharedResNet50MoETeacher(nn.Module):
    def __init__(self, domain_names: list[str], shared_weight: float = 0.5):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.domain_names = domain_names
        self.shared_head = nn.Linear(in_features, 10)
        self.expert_heads = nn.ModuleDict({name: nn.Linear(in_features, 10) for name in domain_names})
        self.gate = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, len(domain_names)),
        )
        self.shared_weight = shared_weight
        self.feature_dim = in_features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        embedding = self.backbone(x)
        shared_logits = self.shared_head(embedding)
        expert_logits = torch.stack([self.expert_heads[name](embedding) for name in self.domain_names], dim=1)
        gate_logits = self.gate(embedding)
        gate_weights = F.softmax(gate_logits, dim=1)
        mixture_logits = (expert_logits * gate_weights.unsqueeze(-1)).sum(dim=1)
        combined_logits = self.shared_weight * shared_logits + (1.0 - self.shared_weight) * mixture_logits
        if return_features:
            return {
                "embedding": embedding,
                "shared_logits": shared_logits,
                "expert_logits": expert_logits,
                "gate_logits": gate_logits,
                "gate_weights": gate_weights,
                "mixture_logits": mixture_logits,
                "logits": combined_logits,
            }
        return combined_logits


@torch.no_grad()
def evaluate_teacher_on_dataset(model: SharedResNet50MoETeacher, dataset: Dataset, batch_size: int, device: torch.device) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = to_resnet_input(images.to(device))
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        batch_size_local = labels.size(0)
        total_loss += loss.item() * batch_size_local
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size_local
    return total_loss / total_examples, total_correct / total_examples


def batch_alignment_loss(embeddings: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    device = embeddings.device
    loss = torch.zeros((), device=device)
    count = 0
    for label in labels.unique():
        label_mask = labels == label
        label_embeddings = embeddings[label_mask]
        if label_embeddings.size(0) < 2:
            continue
        global_mean = label_embeddings.mean(dim=0, keepdim=True)
        for domain in domains[label_mask].unique():
            domain_mask = label_mask & (domains == domain)
            domain_embeddings = embeddings[domain_mask]
            if domain_embeddings.size(0) == 0:
                continue
            loss = loss + F.mse_loss(domain_embeddings.mean(dim=0, keepdim=True), global_mean)
            count += 1
    if count == 0:
        return torch.zeros((), device=device)
    return loss / count


def train_teacher(
    train_datasets: dict[str, Dataset],
    test_datasets: dict[str, Dataset],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    alignment_weight: float,
    gate_weight: float,
    shared_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[Path, pd.DataFrame]:
    domain_names = list(train_datasets)
    train_parts = []
    val_parts = []
    for index, name in enumerate(domain_names):
        train_split, val_split_dataset = split_train_val(train_datasets[name], val_split=val_split, seed=seed + index)
        train_parts.append(DomainLabeledDataset(train_split, index))
        val_parts.append(DomainLabeledDataset(val_split_dataset, index))

    merged_train = ConcatDataset(train_parts)
    train_loader = DataLoader(
        merged_train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = SharedResNet50MoETeacher(domain_names=domain_names, shared_weight=shared_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    best_score = -1.0
    best_checkpoint = output_dir / "shared_resnet50_moe_teacher_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for images, labels, domain_indices in train_loader:
            images = to_resnet_input(images.to(device))
            labels = labels.to(device)
            domain_indices = domain_indices.to(device)

            optimizer.zero_grad()
            outputs = model(images, return_features=True)
            batch_indices = torch.arange(labels.size(0), device=device)
            domain_logits = outputs["expert_logits"][batch_indices, domain_indices]
            shared_loss = loss_fn(outputs["shared_logits"], labels)
            expert_loss = loss_fn(domain_logits, labels)
            combined_loss = loss_fn(outputs["logits"], labels)
            gate_loss = loss_fn(outputs["gate_logits"], domain_indices)
            align_loss = batch_alignment_loss(outputs["embedding"], labels, domain_indices)
            loss = combined_loss + 0.5 * shared_loss + 0.5 * expert_loss + gate_weight * gate_loss + alignment_weight * align_loss
            loss.backward()
            optimizer.step()

            batch_size_local = labels.size(0)
            total_loss += loss.item() * batch_size_local
            total_correct += (outputs["logits"].argmax(dim=1) == labels).sum().item()
            total_examples += batch_size_local

        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "train_accuracy": total_correct / total_examples,
        }
        mean_test_acc = 0.0
        for name, dataset in test_datasets.items():
            _, test_acc = evaluate_teacher_on_dataset(model, dataset, batch_size=batch_size, device=device)
            row[f"test_accuracy_{name}"] = test_acc
            mean_test_acc += test_acc
        mean_test_acc /= len(test_datasets)
        row["test_accuracy_mean"] = mean_test_acc
        rows.append(row)
        if mean_test_acc > best_score:
            best_score = mean_test_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "domain_names": domain_names}, best_checkpoint)
        print(f"[teacher] epoch={epoch}/{epochs} train={row['train_accuracy']:.4f} mean_test={mean_test_acc:.4f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "teacher_metrics.csv", index=False)
    return best_checkpoint, metrics


def load_teacher(checkpoint_path: Path, device: torch.device) -> SharedResNet50MoETeacher:
    payload = torch.load(checkpoint_path, map_location=device)
    model = SharedResNet50MoETeacher(domain_names=payload["domain_names"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def collect_prototypes(
    teacher: SharedResNet50MoETeacher,
    train_datasets: dict[str, Dataset],
    dataset_weights: dict[str, float],
    sample_size_per_dataset: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, float]]:
    sums = {label: None for label in range(10)}
    sq_sums = {label: None for label in range(10)}
    counts = {label: 0.0 for label in range(10)}
    confidences = {}

    for idx, (name, dataset) in enumerate(train_datasets.items()):
        images, labels = materialize_dataset(dataset)
        if len(labels) > sample_size_per_dataset:
            chosen = torch.randperm(len(labels), generator=torch.Generator().manual_seed(100 + idx))[:sample_size_per_dataset]
            images = images[chosen]
            labels = labels[chosen]
        loader = DataLoader(TensorDigitDataset(images, labels, name), batch_size=batch_size, shuffle=False)
        domain_confidences = []
        for batch_images, batch_labels in loader:
            inputs = to_resnet_input(batch_images.to(device))
            outputs = teacher(inputs, return_features=True)
            embeddings = outputs["embedding"].cpu()
            probs = F.softmax(outputs["logits"], dim=1).cpu()
            domain_confidences.append(probs.max(dim=1).values.mean())
            for label in range(10):
                mask = batch_labels == label
                if mask.sum() == 0:
                    continue
                weight = dataset_weights[name]
                selected = embeddings[mask]
                weighted_selected = selected * weight
                if sums[label] is None:
                    sums[label] = weighted_selected.sum(dim=0)
                    sq_sums[label] = (weighted_selected * selected).sum(dim=0)
                else:
                    sums[label] += weighted_selected.sum(dim=0)
                    sq_sums[label] += (weighted_selected * selected).sum(dim=0)
                counts[label] += float(mask.sum()) * weight
        confidences[name] = float(torch.stack(domain_confidences).mean()) if domain_confidences else 0.0

    prototypes = {}
    spreads = {}
    for label in range(10):
        prototypes[label] = sums[label] / max(counts[label], 1e-6)
        variance = (sq_sums[label] / max(counts[label], 1e-6)) - prototypes[label].pow(2)
        spreads[label] = variance.clamp_min(1e-6).sqrt()
    return prototypes, spreads, confidences


def synthesize_prototype_dataset(
    teacher: SharedResNet50MoETeacher,
    prototypes: dict[int, torch.Tensor],
    spreads: dict[int, torch.Tensor],
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
    for label in range(10):
        raw = torch.randn(ipc, 1, 28, 28, device=device, requires_grad=True)
        optimizer = optim.Adam([raw], lr=lr)
        target_embed = prototypes[label].to(device)
        target_spread = spreads[label].to(device)
        target_labels = torch.full((ipc,), label, dtype=torch.long, device=device)
        for _ in range(steps):
            optimizer.zero_grad()
            synth = torch.tanh(raw)
            outputs = teacher(to_resnet_input(synth), return_features=True)
            logits = outputs["logits"]
            embeddings = outputs["embedding"]
            class_loss = F.cross_entropy(logits, target_labels)
            feature_loss = F.mse_loss(embeddings.mean(dim=0), target_embed) + 0.2 * F.mse_loss(embeddings.std(dim=0), target_spread)
            probs = F.softmax(logits, dim=1)
            entropy_loss = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
            tv_loss = (
                (synth[:, :, 1:, :] - synth[:, :, :-1, :]).pow(2).mean()
                + (synth[:, :, :, 1:] - synth[:, :, :, :-1]).pow(2).mean()
            )
            loss = class_loss + feature_weight * feature_loss + entropy_weight * entropy_loss + 1e-3 * tv_loss
            loss.backward()
            optimizer.step()
        final_images = torch.tanh(raw).detach().cpu()
        synth_images.append(final_images)
        synth_labels.append(torch.full((ipc,), label, dtype=torch.long))
    return TensorDigitDataset(torch.cat(synth_images, dim=0), torch.cat(synth_labels, dim=0), "ResNet50MoEPrototype")


class SoftLabelDataset(Dataset):
    def __init__(self, images: torch.Tensor, hard_labels: torch.Tensor, soft_labels: torch.Tensor):
        self.images = images.float()
        self.hard_labels = hard_labels.long()
        self.soft_labels = soft_labels.float()

    def __len__(self) -> int:
        return len(self.hard_labels)

    def __getitem__(self, index: int):
        return self.images[index], self.hard_labels[index], self.soft_labels[index]


@torch.no_grad()
def relabel_with_teacher(
    dataset: TensorDigitDataset,
    teacher: SharedResNet50MoETeacher,
    temperature: float,
    batch_size: int,
    device: torch.device,
) -> SoftLabelDataset:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images_out = []
    hard_labels_out = []
    soft_labels_out = []
    for images, hard_labels in loader:
        logits = teacher(to_resnet_input(images.to(device))) / temperature
        soft_labels = F.softmax(logits, dim=1).cpu()
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
        row = {"epoch": epoch, "train_loss": total_loss / total_examples, "train_accuracy": total_correct / total_examples}
        for name, test_dataset in test_datasets.items():
            test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)
            loss, acc = evaluate_simplecnn(student, test_loader, ce_loss, device)
            row[f"test_loss_{name}"] = loss
            row[f"test_accuracy_{name}"] = acc
        rows.append(row)
        print(f"[student] epoch={epoch}/{epochs} train_acc={row['train_accuracy']:.4f}")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "student_metrics.csv", index=False)
    cross_eval = pd.DataFrame(evaluate_model_on_all(student, "ResNet50MoEProtoStudent", test_datasets, batch_size=batch_size, device=device))
    cross_eval.to_csv(output_dir / "student_cross_eval.csv", index=False)
    torch.save({"model_state_dict": student.state_dict()}, output_dir / "student_best_last.pt")
    return metrics, cross_eval


@torch.no_grad()
def evaluate_simplecnn(model: SimpleCNN, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        batch_size_local = labels.size(0)
        total_loss += loss.item() * batch_size_local
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size_local
    return total_loss / total_examples, total_correct / total_examples


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir))
    data_root = Path(args.data_root)

    train_datasets = {}
    test_datasets = {}
    for name in args.datasets:
        train_dataset, test_dataset = load_dataset_pair(name, data_root)
        train_datasets[name] = train_dataset
        test_datasets[name] = test_dataset

    teacher_ckpt, teacher_metrics = train_teacher(
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        output_dir=output_dir,
        epochs=args.teacher_epochs,
        batch_size=args.teacher_batch_size,
        learning_rate=args.teacher_learning_rate,
        val_split=args.val_split,
        alignment_weight=args.alignment_weight,
        gate_weight=args.gate_weight,
        shared_weight=args.shared_weight,
        seed=args.seed,
        device=device,
    )

    teacher = load_teacher(teacher_ckpt, device)
    gap_df = compute_gap_matrix(train_datasets, args.gap_sample_size, args.num_projections, args.distance_batch_size, device, output_dir)
    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    gap_weights = (1.0 / mean_gaps).to_dict()
    gap_total = sum(gap_weights.values())
    gap_weights = {name: value / gap_total for name, value in gap_weights.items()}

    prototypes, spreads, confidences = collect_prototypes(
        teacher=teacher,
        train_datasets=train_datasets,
        dataset_weights=gap_weights,
        sample_size_per_dataset=args.prototype_sample_size,
        batch_size=args.teacher_batch_size,
        device=device,
    )
    confidence_total = max(sum(confidences.values()), 1e-8)
    confidence_weights = {name: value / confidence_total for name, value in confidences.items()}
    proto_weights = {
        name: args.prototype_gap_weight * gap_weights[name] + (1.0 - args.prototype_gap_weight) * confidence_weights[name]
        for name in args.datasets
    }
    with open(output_dir / "teacher_weight_summary.json", "w", encoding="utf-8") as file:
        json.dump({"gap_weights": gap_weights, "confidence_weights": confidence_weights, "prototype_weights": proto_weights}, file, indent=2)

    prototypes, spreads, _ = collect_prototypes(
        teacher=teacher,
        train_datasets=train_datasets,
        dataset_weights=proto_weights,
        sample_size_per_dataset=args.prototype_sample_size,
        batch_size=args.teacher_batch_size,
        device=device,
    )
    synth_dataset = synthesize_prototype_dataset(
        teacher=teacher,
        prototypes=prototypes,
        spreads=spreads,
        ipc=args.ipc,
        steps=args.recover_steps,
        lr=args.recover_lr,
        feature_weight=args.feature_weight,
        entropy_weight=args.entropy_weight,
        seed=args.seed,
        device=device,
    )
    torch.save({"images": synth_dataset.images, "labels": synth_dataset.labels}, output_dir / "prototype_synth_dataset.pt")

    relabeled = relabel_with_teacher(
        dataset=synth_dataset,
        teacher=teacher,
        temperature=args.temperature,
        batch_size=args.student_batch_size,
        device=device,
    )
    torch.save({"images": relabeled.images, "hard_labels": relabeled.hard_labels, "soft_labels": relabeled.soft_labels}, output_dir / "prototype_relabeled_dataset.pt")

    _, cross_eval = train_student(
        dataset=relabeled,
        test_datasets=test_datasets,
        epochs=args.student_epochs,
        batch_size=args.student_batch_size,
        learning_rate=args.student_learning_rate,
        alpha=args.alpha,
        temperature=args.temperature,
        seed=args.seed,
        device=device,
        output_dir=output_dir,
    )
    distilled_gap = compute_distilled_gaps(
        TensorDigitDataset(relabeled.images, relabeled.hard_labels, "ResNet50MoEPrototype"),
        train_datasets,
        args.gap_sample_size,
        args.num_projections,
        args.distance_batch_size,
        device,
        output_dir,
    )
    plot_distilled_gap(distilled_gap, output_dir / "prototype_distilled_gap_bar.png")
    plot_distilled_cross_eval(cross_eval.to_dict("records"), output_dir / "prototype_distilled_cross_accuracy.png")

    summary = {
        "teacher_checkpoint": str(teacher_ckpt),
        "teacher_best_mean_test_accuracy": float(teacher_metrics["test_accuracy_mean"].max()),
        "student_mean_cross_accuracy": float(cross_eval["test_accuracy"].mean()),
        "student_min_cross_accuracy": float(cross_eval["test_accuracy"].min()),
        "student_max_cross_accuracy": float(cross_eval["test_accuracy"].max()),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared ResNet-50 MoE teacher with feature-space prototype distillation.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--teacher-epochs", type=int, default=8)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--teacher-learning-rate", type=float, default=1e-4)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--student-batch-size", type=int, default=128)
    parser.add_argument("--student-learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--gap-sample-size", type=int, default=300)
    parser.add_argument("--num-projections", type=int, default=80)
    parser.add_argument("--distance-batch-size", type=int, default=64)
    parser.add_argument("--prototype-sample-size", type=int, default=2000)
    parser.add_argument("--ipc", type=int, default=30)
    parser.add_argument("--recover-steps", type=int, default=250)
    parser.add_argument("--recover-lr", type=float, default=0.03)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--gate-weight", type=float, default=0.2)
    parser.add_argument("--shared-weight", type=float, default=0.5)
    parser.add_argument("--prototype-gap-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
