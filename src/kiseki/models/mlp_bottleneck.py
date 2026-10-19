"""A 16-unit linear bottleneck before one 128-unit ReLU layer: Flatten → Dense(16) →
Dense(128) → ReLU → Dense(classes); about 16k weights on 28x28 inputs."""

from kiseki.models.layers import Dense, Flatten, Network, ReLU, network


def build(input_shape: tuple[int, int, int], classes: int) -> Network:
    return network(
        "mlp_bottleneck",
        input_shape,
        classes,
        [Flatten(), Dense(16), Dense(128), ReLU(), Dense(classes)],
    )
