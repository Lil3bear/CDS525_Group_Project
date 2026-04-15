from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE


def savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a distilled dataset payload.")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(input_path, map_location="cpu")
    images = payload["images"].float()
    hard_labels = payload["hard_labels"].long()
    soft_labels = payload["soft_labels"].float()

    confidence = soft_labels.max(dim=1).values
    entropy = -(soft_labels.clamp_min(1e-8) * soft_labels.clamp_min(1e-8).log()).sum(dim=1)

    summary = pd.DataFrame(
        [
            {"metric": "num_samples", "value": int(images.shape[0])},
            {"metric": "channels", "value": int(images.shape[1])},
            {"metric": "height", "value": int(images.shape[2])},
            {"metric": "width", "value": int(images.shape[3])},
            {"metric": "pixel_min", "value": float(images.min())},
            {"metric": "pixel_max", "value": float(images.max())},
            {"metric": "pixel_mean", "value": float(images.mean())},
            {"metric": "pixel_std", "value": float(images.std())},
            {"metric": "soft_conf_mean", "value": float(confidence.mean())},
            {"metric": "soft_conf_std", "value": float(confidence.std())},
            {"metric": "soft_entropy_mean", "value": float(entropy.mean())},
            {"metric": "soft_entropy_std", "value": float(entropy.std())},
        ]
    )
    summary.to_csv(output_dir / "distilled_dataset_summary.csv", index=False)

    label_counts = pd.DataFrame(
        {
            "label": list(range(10)),
            "count": [int((hard_labels == label).sum()) for label in range(10)],
            "pred_count": [int((soft_labels.argmax(dim=1) == label).sum()) for label in range(10)],
        }
    )
    label_counts.to_csv(output_dir / "distilled_label_counts.csv", index=False)

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            ax.imshow(images[idx, 0].numpy(), cmap="gray")
            ax.set_title(f"{int(hard_labels[idx])}/{int(soft_labels[idx].argmax())}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Distilled samples: hard/pred", fontsize=12)
    savefig(fig, output_dir / "distilled_grid_100.png")

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for label in range(10):
        indices = torch.where(hard_labels == label)[0][:10]
        for col in range(10):
            ax = axes[label, col]
            if col < len(indices):
                idx = int(indices[col])
                ax.imshow(images[idx, 0].numpy(), cmap="gray")
                ax.set_title(f"{float(confidence[idx]):.2f}", fontsize=7)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(str(label), rotation=0, labelpad=12, fontsize=10)
    fig.suptitle("Per-class distilled samples; title=soft confidence", fontsize=12)
    savefig(fig, output_dir / "distilled_by_class.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confidence.numpy(), bins=20, color="#2a6f97", edgecolor="white")
    ax.set_title("Soft-label confidence distribution")
    ax.set_xlabel("max soft-label probability")
    ax.set_ylabel("count")
    savefig(fig, output_dir / "distilled_confidence_hist.png")

    flat_images = images.view(images.shape[0], -1).numpy()
    perplexity = min(30, max(5, len(flat_images) // 20))
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity)
    coords = tsne.fit_transform(flat_images)
    tsne_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": hard_labels.numpy(),
            "pred": soft_labels.argmax(dim=1).numpy(),
            "confidence": confidence.numpy(),
        }
    )
    tsne_df.to_csv(output_dir / "distilled_tsne.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(tsne_df["x"], tsne_df["y"], c=tsne_df["label"], cmap="tab10", s=18, alpha=0.85)
    ax.set_title("t-SNE of distilled images colored by hard label")
    fig.colorbar(scatter, ax=ax)
    savefig(fig, output_dir / "distilled_tsne.png")

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
