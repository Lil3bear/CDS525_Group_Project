import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from kmnist_oss.data import get_dataloaders
from kmnist_oss.model import SimpleCNN
from kmnist_oss.train import FIGURES_DIR, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the first 100 test predictions.")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best_baseline.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    _, _, test_loader = get_dataloaders(
        batch_size=100,
        data_dir=args.data_dir,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    images, labels = next(iter(test_loader))
    images = images[:100].to(device)
    labels = labels[:100]

    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(10, 10, figsize=(18, 18))

    for index, axis in enumerate(axes.flat):
        image = images[index].cpu().squeeze().numpy()
        image = (image * 0.5) + 0.5
        axis.imshow(image, cmap="gray")
        axis.set_title(
            f"P:{predictions[index].item()} T:{labels[index].item()}",
            fontsize=7,
        )
        axis.axis("off")

    fig.suptitle("First 100 KMNIST Test Predictions", fontsize=18)
    output_path = FIGURES_DIR / "first_100_test_predictions.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
