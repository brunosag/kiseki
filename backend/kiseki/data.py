from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, default_collate, random_split
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


def load_mnist_test(
    data_dir: Path | str,
    *,
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    raw_dataset = datasets.MNIST(root=Path(data_dir), train=False, download=download)
    dataset = TensorDataset(
        raw_dataset.data.unsqueeze(1).float().div(255.0),
        torch.as_tensor(raw_dataset.targets, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
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


class DeterministicBatchStream:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        batch_size: int,
        seed: int,
        train_indices: list[int] | None = None,
        validation_indices: list[int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed + 1)
        self.train_indices = train_indices or dataset_indices(dataset)
        self.validation_indices = validation_indices or []
        self.current_epoch = -1
        self.current_permutation: torch.Tensor | None = None
        self.next_batch_offset = 0

    def __iter__(self) -> "DeterministicBatchStream":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.dataset) == 0:
            raise StopIteration

        if (
            self.current_permutation is None
            or self.next_batch_offset >= len(self.current_permutation)
        ):
            self.current_epoch += 1
            self.current_permutation = torch.randperm(len(self.dataset), generator=self.generator)
            self.next_batch_offset = 0

        end = min(self.next_batch_offset + self.batch_size, len(self.current_permutation))
        batch_positions = self.current_permutation[self.next_batch_offset : end]
        self.next_batch_offset = end
        batch = [self.dataset[int(position)] for position in batch_positions.tolist()]
        return default_collate(batch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "train_indices": self.train_indices,
            "validation_indices": self.validation_indices,
            "current_epoch": self.current_epoch,
            "current_permutation": self.current_permutation,
            "next_batch_offset": self.next_batch_offset,
            "batch_size": self.batch_size,
            "shuffle_generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.current_epoch = int(state.get("current_epoch", -1))
        self.current_permutation = state.get("current_permutation")
        self.next_batch_offset = int(state.get("next_batch_offset", 0))
        self.batch_size = int(state.get("batch_size", self.batch_size))
        self.train_indices = list(state.get("train_indices", self.train_indices))
        self.validation_indices = list(state.get("validation_indices", self.validation_indices))
        generator_state = state.get("shuffle_generator_state")
        if generator_state is not None:
            self.generator.set_state(generator_state)


def deterministic_batch_stream(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    batch_size: int,
    seed: int,
) -> DeterministicBatchStream:
    return DeterministicBatchStream(
        train_loader.dataset,
        batch_size=batch_size,
        seed=seed,
        train_indices=dataset_indices(train_loader.dataset),
        validation_indices=dataset_indices(val_loader.dataset),
    )


def dataset_indices(dataset: torch.utils.data.Dataset) -> list[int]:
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return list(range(len(dataset)))
    return [int(index) for index in indices]


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

    def mnist_test(self) -> DataLoader:
        return load_mnist_test(
            self.data_dir,
            download=self.download,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
