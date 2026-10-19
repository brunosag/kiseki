"""Model catalog. A model is a JAX module here declaring its layers; see `layers.py`.

Names are listed without importing JAX so the core can validate specs. The worker calls
`load(name)` to get the `Network` (init, apply, taps and LRP all derive from the layers).
"""

import importlib

NAMES = ("small_cnn", "wide_cnn", "vgg_mini", "mlp_bottleneck")


def load(name: str, input_shape: tuple[int, int, int], classes: int):
    if name not in NAMES:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(NAMES)}")
    module = importlib.import_module(f"kiseki.models.{name}")
    return module.build(input_shape, classes)
