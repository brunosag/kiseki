from typing import Annotated, Literal

import msgspec
from msgspec import Meta

TITLE = "Layer-wise relevance propagation"
SUMMARY = ("Which pixels each model's decision rests on: LRP heatmaps (z^B, gamma, epsilon "
           "rules) for sample images chosen where the models disagree, and class averages.")


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    gallery: Annotated[int, Meta(ge=0, le=200, description="Sample images in the gallery")] = 24
    per_class: Annotated[int, Meta(ge=0, description="Images averaged per class (0: none)")] = 500
    target: Literal["label", "prediction"] = "label"
    gamma: Annotated[float, Meta(ge=0)] = 0.25
    epsilon: Annotated[float, Meta(gt=0)] = 0.25
