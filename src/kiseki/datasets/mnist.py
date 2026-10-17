"""MNIST: 70k handwritten digits, 28x28 grayscale (LeCun et al., 1998)."""

from pathlib import Path

from kiseki.datasets.base import DatasetInfo, RawFile, RawSplits
from kiseki.datasets.idx import parse_idx_dir

_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

INFO = DatasetInfo(
    name="mnist",
    title="MNIST",
    image_shape=(28, 28, 1),
    class_names=tuple(str(digit) for digit in range(10)),
    train_size=60000,
    validation_size=10000,
    default_model="small_cnn",
    homepage="https://yann.lecun.com/exdb/mnist/",
    files=(
        RawFile("train-images-idx3-ubyte.gz", _URL + "train-images-idx3-ubyte.gz",
                "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
        RawFile("train-labels-idx1-ubyte.gz", _URL + "train-labels-idx1-ubyte.gz",
                "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
        RawFile("t10k-images-idx3-ubyte.gz", _URL + "t10k-images-idx3-ubyte.gz",
                "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"),
        RawFile("t10k-labels-idx1-ubyte.gz", _URL + "t10k-labels-idx1-ubyte.gz",
                "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
    ),
)


def parse(raw_dir: Path) -> RawSplits:
    return parse_idx_dir(raw_dir)
