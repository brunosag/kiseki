"""Kuzushiji-MNIST: 70k cursive hiragana, 28x28 grayscale (Clanuwat et al., 2018).

A drop-in for MNIST at the same cost that is much harder: optimizers that tie near 99% on
MNIST separate here.
"""

from pathlib import Path

from kiseki.datasets.base import DatasetInfo, RawFile, RawSplits
from kiseki.datasets.idx import parse_idx_dir

_URL = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/"

INFO = DatasetInfo(
    name="kmnist",
    title="Kuzushiji-MNIST",
    image_shape=(28, 28, 1),
    class_names=(
        "o (お)",
        "ki (き)",
        "su (す)",
        "tsu (つ)",
        "na (な)",
        "ha (は)",
        "ma (ま)",
        "ya (や)",
        "re (れ)",
        "wo (を)",
    ),
    train_size=60000,
    validation_size=10000,
    default_model="small_cnn",
    homepage="https://github.com/rois-codh/kmnist",
    files=(
        RawFile(
            "train-images-idx3-ubyte.gz",
            _URL + "train-images-idx3-ubyte.gz",
            "51467d22d8cc72929e2a028a0428f2086b092bb31cfb79c69cc0a90ce135fde4",
        ),
        RawFile(
            "train-labels-idx1-ubyte.gz",
            _URL + "train-labels-idx1-ubyte.gz",
            "e38f9ebcd0f3ebcdec7fc8eabdcdaef93bb0df8ea12bee65224341c8183d8e17",
        ),
        RawFile(
            "t10k-images-idx3-ubyte.gz",
            _URL + "t10k-images-idx3-ubyte.gz",
            "edd7a857845ad6bb1d0ba43fe7e794d164fe2dce499a1694695a792adfac43c5",
        ),
        RawFile(
            "t10k-labels-idx1-ubyte.gz",
            _URL + "t10k-labels-idx1-ubyte.gz",
            "20bb9a0ef54c7db3efc55a92eef5582c109615df22683c380526788f98e42a1c",
        ),
    ),
)


def parse(raw_dir: Path) -> RawSplits:
    return parse_idx_dir(raw_dir)
