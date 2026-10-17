from typing import Annotated

import msgspec
from msgspec import Meta

TITLE = "Activations and representation similarity"
SUMMARY = ("Per-layer activation statistics (dead units, sparsity, quantiles) and linear CKA "
           "similarity between every pair of layers of two models.")


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    samples: Annotated[int, Meta(ge=100, description="Images used (CKA cost grows as samples²)")] = 2000
    bins: Annotated[int, Meta(ge=5, le=200)] = 40
