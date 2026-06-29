from typing import Literal

DatasetName = Literal["mnist", "fashion_mnist", "cifar10"]
RealDatasetName = DatasetName

REAL_DATASETS: tuple[RealDatasetName, ...] = ("mnist", "fashion_mnist", "cifar10")
