from typing import Annotated

import msgspec
from msgspec import Meta

TITLE = "Calibration"
SUMMARY = "Whether confidence matches accuracy: ECE, Brier score, NLL, reliability diagram."


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    bins: Annotated[int, Meta(ge=2, le=100, description="Equal-width confidence bins")] = 15
