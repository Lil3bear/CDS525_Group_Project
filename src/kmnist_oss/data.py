from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5
KMNIST_NPZ_FILENAMES = {
    "train_images": "kmnist-train-imgs.npz",
    "train_labels": "kmnist-train-labels.npz",
    "test_images": "kmnist-test-imgs.npz",
    "test_labels": "kmnist-test-labels.npz",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_kmnist_npz_dir(data_dir: str = "data") -> Path:
    return Path(data_dir) / "KMNIST" / "npz"


def ensure_kmnist_npz_dir(data_dir: str = "data") -> Path:
    npz_dir = get_kmnist_npz_dir(data_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    return npz_dir


def load_npz_array(path: Path, kind: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing KMNIST {kind} file: {path}")

    with np.load(path) as data:
        if len(data.files) == 0:
            raise ValueError(f"No arrays found in {path}")
        if len(data.files) == 1:
            return data[data.files[0]]

        if kind == "images":
            preferred_keys = ["images", "imgs", "x", "data", "arr_0"]
        else:
            preferred_keys = ["labels", "targets", "y", "data", "arr_0"]

        for key in preferred_keys:
            if key in data.files:
                return data[key]

        return data[data.files[0]]


def prepare_images(images: np.ndarray) -> torch.Tensor:
    images = np.asarray(images, dtype=np.float32)

    if images.ndim == 3:
        images = images[:, None, :, :]
    elif images.ndim == 4:
        if images.shape[1] == 1:
            pass
        elif images.shape[-1] == 1:
            images = np.transpose(images, (0, 3, 1, 2))
        else:
            raise ValueError(f"Unsupported image shape: {images.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {images.shape}")

    if images.max() > 1.0:
        images = images / 255.0

    images = (images - NORMALIZE_MEAN) / NORMALIZE_STD
    return torch.from_numpy(images)


def prepare_labels(labels: np.ndarray) -> torch.Tensor:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    return torch.from_numpy(labels)


class KMNISTNPZDataset(Dataset):
    def __init__(self, images_path: Path, labels_path: Path):
        images = load_npz_array(images_path, kind="images")
        labels = load_npz_array(labels_path, kind="labels")

        self.images = prepare_images(images)
        self.labels = prepare_labels(labels)

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Image and label count mismatch: {len(self.images)} images vs {len(self.labels)} labels"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def build_datasets(data_dir: str = "data", val_split: float = 0.1, seed: int = 42):
    npz_dir = ensure_kmnist_npz_dir(data_dir)

    train_images_path = npz_dir / KMNIST_NPZ_FILENAMES["train_images"]
    train_labels_path = npz_dir / KMNIST_NPZ_FILENAMES["train_labels"]
    test_images_path = npz_dir / KMNIST_NPZ_FILENAMES["test_images"]
    test_labels_path = npz_dir / KMNIST_NPZ_FILENAMES["test_labels"]

    full_train_dataset = KMNISTNPZDataset(train_images_path, train_labels_path)
    test_dataset = KMNISTNPZDataset(test_images_path, test_labels_path)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size: int = 64,
    data_dir: str = "data",
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
):
    set_seed(seed)
    train_dataset, val_dataset, test_dataset = build_datasets(
        data_dir=data_dir,
        val_split=val_split,
        seed=seed,
    )

    pin_memory = torch.cuda.is_available()
    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
