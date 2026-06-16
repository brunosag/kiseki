from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets


def load_mnist(
    data_dir: Path | str,
    batch_size: int,
    seed: int,
    *,
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    raw_dataset = datasets.MNIST(root=Path(data_dir), train=True, download=download)
    dataset = TensorDataset(
        raw_dataset.data.unsqueeze(1).float().div(255.0),
        torch.as_tensor(raw_dataset.targets, dtype=torch.long),
    )
    return make_regular_train_val_loaders(
        dataset,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_regular_train_val_loaders(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    seed: int,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    if len(dataset) < 60000:
        raise ValueError("MNIST train split must contain at least 60000 examples")

    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [50000, 10000], generator=split_generator)

    shuffle_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def cycle_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


@dataclass(slots=True)
class DataLoaderFactory:
    data_dir: Path = Path("data")
    download: bool = True
    num_workers: int = 0
    pin_memory: bool = False

    def mnist(self, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
        return load_mnist(
            self.data_dir,
            batch_size=batch_size,
            seed=seed,
            download=self.download,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
