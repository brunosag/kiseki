from typing import Annotated, Literal

import msgspec
from msgspec import Meta

TITLE = "Loss landscape"
SUMMARY = ("The loss around each model on random filter-normalized planes (Li et al. 2018), "
           "and along the straight line between two models: whether they sit in one basin "
           "(linear mode connectivity) and how sharp each minimum is.")


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    samples: Annotated[int, Meta(ge=100, description="Images the loss is measured on")] = 2000
    data: Literal["train", "test"] = "train"
    resolution: Annotated[int, Meta(ge=5, le=101, description="Grid points per axis of each plane")] = 31
    radius: Annotated[float, Meta(gt=0, description="Plane half-width, in filter-normalized units")] = 1.0
    line_points: Annotated[int, Meta(ge=3, le=401)] = 41
    line_margin: Annotated[float, Meta(ge=0, le=1, description="Extend the line this far past both models")] = 0.25
