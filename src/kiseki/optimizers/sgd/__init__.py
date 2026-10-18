"""Minibatch stochastic gradient descent with optional momentum and weight decay."""

from typing import Annotated

import msgspec
from msgspec import Meta

TITLE = "SGD"
SUMMARY = "Minibatch SGD on the cross-entropy loss (gradient baseline)."


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    learning_rate: Annotated[float, Meta(gt=0, description="Step size η")] = 0.01
    momentum: Annotated[float, Meta(ge=0, lt=1, description="Heavy-ball momentum")] = 0.0
    weight_decay: Annotated[float, Meta(ge=0, description="L2 coefficient")] = 0.0
