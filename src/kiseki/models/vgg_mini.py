"""Six 3x3 convolutions in three VGG-style stages (16, 32, 64 channels), average-pooled to
2x2, then a 128-unit hidden layer: about 106k weights on 32x32x3 inputs."""

from kiseki.models.layers import AvgPool, Conv, Dense, Flatten, MaxPool, Network, ReLU, network


def build(input_shape: tuple[int, int, int], classes: int) -> Network:
    height = input_shape[0]
    return network(
        "vgg_mini",
        input_shape,
        classes,
        [
            Conv(16, 3, "SAME"),
            ReLU(),
            Conv(16, 3, "SAME"),
            ReLU(),
            MaxPool(2),
            Conv(32, 3, "SAME"),
            ReLU(),
            Conv(32, 3, "SAME"),
            ReLU(),
            MaxPool(2),
            Conv(64, 3, "SAME"),
            ReLU(),
            Conv(64, 3, "SAME"),
            ReLU(),
            AvgPool(height // 8),
            Flatten(),
            Dense(128),
            ReLU(),
            Dense(classes),
        ],
    )
