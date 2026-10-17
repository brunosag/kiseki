"""Two 3x3 convolutions (8, 16 channels) and a 120-unit hidden layer: about 50k weights on
28x28 inputs. Cheap enough for populations of thousands."""

from kiseki.models.layers import Conv, Dense, Flatten, MaxPool, Network, ReLU, network


def build(input_shape: tuple[int, int, int], classes: int) -> Network:
    return network("small_cnn", input_shape, classes, [
        Conv(8, 3), ReLU(), MaxPool(2),
        Conv(16, 3), ReLU(), MaxPool(2),
        Flatten(), Dense(120), ReLU(), Dense(classes),
    ])
