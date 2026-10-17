"""Fashion-MNIST: 70k Zalando article images, 28x28 grayscale (Xiao et al., 2017)."""

from pathlib import Path

from kiseki.datasets.base import DatasetInfo, RawFile, RawSplits
from kiseki.datasets.idx import parse_idx_dir

_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

INFO = DatasetInfo(
    name="fashion_mnist",
    title="Fashion-MNIST",
    image_shape=(28, 28, 1),
    class_names=(
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ),
    train_size=60000,
    validation_size=10000,
    default_model="wide_cnn",
    homepage="https://github.com/zalandoresearch/fashion-mnist",
    files=(
        RawFile("train-images-idx3-ubyte.gz", _URL + "train-images-idx3-ubyte.gz",
                "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"),
        RawFile("train-labels-idx1-ubyte.gz", _URL + "train-labels-idx1-ubyte.gz",
                "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"),
        RawFile("t10k-images-idx3-ubyte.gz", _URL + "t10k-images-idx3-ubyte.gz",
                "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073"),
        RawFile("t10k-labels-idx1-ubyte.gz", _URL + "t10k-labels-idx1-ubyte.gz",
                "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5"),
    ),
)


def parse(raw_dir: Path) -> RawSplits:
    return parse_idx_dir(raw_dir)
