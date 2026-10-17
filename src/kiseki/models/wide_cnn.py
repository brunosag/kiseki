"""Two 5x5 same-padded convolutions (32, 64 channels) straight into the classifier: about
83k weights on 28x28 inputs, ~45x the compute of small_cnn."""

from kiseki.models.layers import Conv, Dense, Flatten, MaxPool, Network, ReLU, network


def build(input_shape: tuple[int, int, int], classes: int) -> Network:
    return network("wide_cnn", input_shape, classes, [
        Conv(32, 5, "SAME"), ReLU(), MaxPool(2),
        Conv(64, 5, "SAME"), ReLU(), MaxPool(2),
        Flatten(), Dense(classes),
    ])
