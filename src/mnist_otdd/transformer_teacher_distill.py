from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from torchvision.models import ViT_L_16_Weights
from torchvision.transforms import functional as TF

from kmnist_oss.model import SimpleCNN
from mnist_otdd.cross_dataset_study import (
    TensorDigitDataset,
    build_loader,
    compute_distilled_gaps,
    compute_gap_matrix,
    ensure_dir,
    evaluate,
    evaluate_model_on_all,
    load_dataset_pair,
    materialize_dataset,
    plot_distilled_cross_eval,
    plot_distilled_gap,
    resolve_device,
    set_seed,
    split_train_val,
    train_one_epoch,
)


class ViTDatasetWrapper(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        image = (image * 0.5) + 0.5
        image = image.repeat(3, 1, 1)
        image = TF.resize(image, [224, 224])
        image = TF.normalize(image, mean=ViT_L_16_Weights.IMAGENET1K_V1.transforms().mean, std=ViT_L_16_Weights.IMAGENET1K_V1.transforms().std)
        return image, label


class SoftLabelDataset(Dataset):
    def __init__(self, images: torch.Tensor, hard_labels: torch.Tensor, soft_labels: torch.Tensor):
        self.images = images.float()
        self.hard_labels = hard_labels.long()
        self.soft_labels = soft_labels.float()

    def __len__(self) -> int:
        return len(self.hard_labels)

    def __getitem__(self, index: int):
        return self.images[index], self.hard_labels[index], self.soft_labels[index]


class ViTTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        in_features = backbone.heads.head.in_features
        backbone.heads.head = nn.Linear(in_features, 10)
        self.backbone = backbone
        self.feature_dim = in_features

    def forward(self, x, return_features: bool = False):
        x = self.backbone._process_input(x)
        n = x.shape[0]
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone.encoder(x)
        cls = x[:, 0]
        logits = self.backbone.heads(cls)
        if return_features:
            return {"embedding": cls, "logits": logits}
        return logits


def train_vit_teacher(
    dataset_name: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    seed: int,
    device: torch.device,
) -> Path:
    wrapped_train = ViTDatasetWrapper(train_dataset)
    wrapped_test = ViTDatasetWrapper(test_dataset)
    train_split, val_split_dataset = split_train_val(wrapped_train, val_split=val_split, seed=seed)
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_split_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(wrapped_test, batch_size=batch_size, shuffle=False)

    model = ViTTeacher().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    best_test = -1.0
    best_checkpoint = output_dir / f"{dataset_name}_vit_best_test.pt"
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        rows.append(
            {
                "dataset": dataset_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        if test_accuracy > best_test:
            best_test = test_accuracy
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_checkpoint)
        print(f"[vit:{dataset_name}] epoch={epoch}/{epochs} val={val_accuracy:.4f} test={test_accuracy:.4f}")
    pd.DataFrame(rows).to_csv(output_dir / f"{dataset_name}_vit_metrics.csv", index=False)
    return best_checkpoint


@torch.no_grad()
def collect_teacher_statistics(
    train_datasets: dict[str, Dataset],
    teacher_paths: dict[str, Path],
    device: torch.device,
    sample_size_per_dataset: int,
) -> tuple[dict[str, ViTTeacher], dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, float]]:
    teachers = {}
    for name, path in teacher_paths.items():
        teacher = ViTTeacher().to(device)
        payload = torch.load(path, map_location=device)
        teacher.load_state_dict(payload["model_state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teachers[name] = teacher

    embeddings_by_class = {label: [] for label in range(10)}
    confidences = {}
    for name, dataset in train_datasets.items():
        images, labels = materialize_dataset(dataset)
        if len(labels) > sample_size_per_dataset:
            idx = torch.randperm(len(labels), generator=torch.Generator().manual_seed(42 + len(name)))[:sample_size_per_dataset]
            images = images[idx]
            labels = labels[idx]
        wrapped = TensorDigitDataset(images, labels, name)
        loader = DataLoader(ViTDatasetWrapper(wrapped), batch_size=64, shuffle=False)
        teacher = teachers[name]
        batch_embeds = []
        batch_labels = []
        confs = []
        for batch_images, batch_labels_tensor in loader:
            batch_images = batch_images.to(device)
            outputs = teacher(batch_images, return_features=True)
            probs = F.softmax(outputs["logits"], dim=1)
            confs.append(probs.max(dim=1).values.cpu())
            batch_embeds.append(outputs["embedding"].cpu())
            batch_labels.append(batch_labels_tensor)
        embeddings = torch.cat(batch_embeds, dim=0)
        labels_cpu = torch.cat(batch_labels, dim=0)
        confidences[name] = torch.cat(confs).mean().item()
        for label in range(10):
            class_embeddings = embeddings[labels_cpu == label]
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
    teachers: dict[str, ViTTeacher],
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
    for label in range(10):
        raw = torch.randn(ipc, 1, 28, 28, device=device, requires_grad=True)
        optimizer = optim.Adam([raw], lr=lr)
        target_embed = prototypes[label].to(device)
        target_spread = spreads[label].to(device)
        for _ in range(steps):
            optimizer.zero_grad()
            images = ((torch.tanh(raw) + 1.0) / 2.0).repeat(1, 3, 1, 1)
            images = TF.resize(images, [224, 224])
            images = TF.normalize(images, mean=ViT_L_16_Weights.IMAGENET1K_V1.transforms().mean, std=ViT_L_16_Weights.IMAGENET1K_V1.transforms().std)
            logits_sum = None
            embed_loss = 0.0
            entropy_loss = 0.0
            for name, teacher in teachers.items():
                outputs = teacher(images, return_features=True)
                weight = teacher_weights[name]
                logits_sum = outputs["logits"] * weight if logits_sum is None else logits_sum + outputs["logits"] * weight
                embed_loss = embed_loss + weight * (
                    F.mse_loss(outputs["embedding"].mean(dim=0), target_embed)
                    + 0.2 * F.mse_loss(outputs["embedding"].std(dim=0), target_spread)
                )
                probs = F.softmax(outputs["logits"], dim=1)
                entropy_loss = entropy_loss + weight * (-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean())
            class_loss = F.cross_entropy(logits_sum, torch.full((ipc,), label, device=device, dtype=torch.long))
            tv_loss = (
                (raw[:, :, 1:, :] - raw[:, :, :-1, :]).pow(2).mean()
                + (raw[:, :, :, 1:] - raw[:, :, :, :-1]).pow(2).mean()
            )
            loss = class_loss + feature_weight * embed_loss + entropy_weight * entropy_loss + 1e-3 * tv_loss
            loss.backward()
            optimizer.step()
        final = ((torch.tanh(raw).detach().cpu()) + 1.0) / 2.0
        final = (final - 0.5) / 0.5
        synth_images.append(final)
        synth_labels.append(torch.full((ipc,), label, dtype=torch.long))
    return TensorDigitDataset(torch.cat(synth_images, dim=0), torch.cat(synth_labels, dim=0), "ViTTeacherSynth")


@torch.no_grad()
def relabel_dataset(dataset: TensorDigitDataset, teachers: dict[str, ViTTeacher], teacher_weights: dict[str, float], temperature: float, batch_size: int, device: torch.device) -> SoftLabelDataset:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images_out, hard_out, soft_out = [], [], []
    for images, hard_labels in loader:
        vit_images = ((images.to(device) * 0.5) + 0.5).repeat(1, 3, 1, 1)
        vit_images = TF.resize(vit_images, [224, 224])
        vit_images = TF.normalize(vit_images, mean=ViT_L_16_Weights.IMAGENET1K_V1.transforms().mean, std=ViT_L_16_Weights.IMAGENET1K_V1.transforms().std)
        logits_sum = None
        for name, teacher in teachers.items():
            logits = teacher(vit_images) / temperature
            weighted = logits * teacher_weights[name]
            logits_sum = weighted if logits_sum is None else logits_sum + weighted
        soft_labels = F.softmax(logits_sum, dim=1).cpu()
        images_out.append(images.cpu())
        hard_out.append(hard_labels.cpu())
        soft_out.append(soft_labels)
    return SoftLabelDataset(torch.cat(images_out), torch.cat(hard_out), torch.cat(soft_out))


def train_student(dataset: SoftLabelDataset, test_datasets: dict[str, Dataset], epochs: int, batch_size: int, learning_rate: float, alpha: float, temperature: float, seed: int, device: torch.device, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            total_loss += loss.item() * hard_labels.size(0)
            total_correct += (logits.argmax(dim=1) == hard_labels).sum().item()
            total_examples += hard_labels.size(0)
        row = {"epoch": epoch, "train_loss": total_loss / total_examples, "train_accuracy": total_correct / total_examples}
        for name, test_dataset in test_datasets.items():
            test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)
            _, acc = evaluate(student, test_loader, ce_loss, device)
            row[f"test_accuracy_{name}"] = acc
        rows.append(row)
        print(f"[student] epoch={epoch}/{epochs} train_acc={row['train_accuracy']:.4f}")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "student_metrics.csv", index=False)
    cross_eval = pd.DataFrame(evaluate_model_on_all(student, "ViTTeacherSimpleCNNStudent", test_datasets, batch_size=batch_size, device=device))
    cross_eval.to_csv(output_dir / "student_cross_eval.csv", index=False)
    return metrics, cross_eval


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

    teacher_paths = {}
    for name in args.datasets:
        teacher_paths[name] = train_vit_teacher(
            dataset_name=name,
            train_dataset=train_datasets[name],
            test_dataset=test_datasets[name],
            output_dir=teacher_dir,
            epochs=args.teacher_epochs,
            batch_size=args.teacher_batch_size,
            learning_rate=args.teacher_learning_rate,
            val_split=args.val_split,
            seed=args.seed,
            device=device,
        )

    gap_df = compute_gap_matrix(train_datasets, args.gap_sample_size, args.num_projections, args.distance_batch_size, device, output_dir)
    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    gap_weights = (1.0 / mean_gaps).to_dict()
    total_gap = sum(gap_weights.values())
    gap_weights = {k: v / total_gap for k, v in gap_weights.items()}

    teachers, prototypes, spreads, confidences = collect_teacher_statistics(train_datasets, teacher_paths, device, args.teacher_stat_sample_size)
    conf_total = sum(confidences.values())
    conf_weights = {k: v / conf_total for k, v in confidences.items()}
    teacher_weights = {name: args.gap_weight_ratio * gap_weights[name] + (1.0 - args.gap_weight_ratio) * conf_weights[name] for name in args.datasets}

    synth_dataset = synthesize_dataset(teachers, prototypes, spreads, teacher_weights, args.ipc, args.recover_steps, args.recover_lr, args.feature_weight, args.entropy_weight, args.seed, device)
    torch.save({"images": synth_dataset.images, "labels": synth_dataset.labels}, output_dir / "synth_dataset.pt")
    relabeled = relabel_dataset(synth_dataset, teachers, teacher_weights, args.temperature, args.student_batch_size, device)
    torch.save({"images": relabeled.images, "hard_labels": relabeled.hard_labels, "soft_labels": relabeled.soft_labels}, output_dir / "relabeled_dataset.pt")

    _, cross_eval = train_student(relabeled, test_datasets, args.student_epochs, args.student_batch_size, args.student_learning_rate, args.alpha, args.temperature, args.seed, device, output_dir)
    gap_to_real = compute_distilled_gaps(TensorDigitDataset(relabeled.images, relabeled.hard_labels, "ViTTeacherSimpleCNNStudent"), train_datasets, args.gap_sample_size, args.num_projections, args.distance_batch_size, device, output_dir)
    plot_distilled_gap(gap_to_real, output_dir / "distilled_gap_bar.png")
    plot_distilled_cross_eval(cross_eval.to_dict("records"), output_dir / "distilled_cross_accuracy.png")
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer teacher distillation with SimpleCNN student.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
