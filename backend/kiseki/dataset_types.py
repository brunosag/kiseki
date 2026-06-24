from typing import Literal

DatasetName = Literal["mnist", "cifar10"]
RealDatasetName = DatasetName

REAL_DATASETS: tuple[RealDatasetName, ...] = ("mnist", "cifar10")

