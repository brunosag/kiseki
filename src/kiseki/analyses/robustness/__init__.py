from typing import Annotated

import msgspec
from msgspec import Meta

TITLE = "Robustness to input corruptions"
SUMMARY = "Accuracy under Gaussian noise, brightness shifts, contrast loss and centre occlusion."


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    noise: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
    brightness: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)
    contrast: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)
    occlusion: Annotated[
        tuple[float, ...],
        Meta(description="Side of the centre square, as a share of the image side"),
    ] = (0.0, 0.125, 0.25, 0.375, 0.5)
