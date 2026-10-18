from typing import Annotated

import msgspec
from msgspec import Meta

TITLE = "Weights"
SUMMARY = "Per-layer weight norms and distributions, and distances between the models' weights."


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    bins: Annotated[int, Meta(ge=5, le=200, description="Histogram bins")] = 40
