"""Adam with decoupled weight decay (AdamW; Kingma & Ba 2015, Loshchilov & Hutter 2019).

`l2` instead adds a coupled L2 term to the gradient before the moments, as Adam's
original weight decay (and `torch.optim.Adam(weight_decay=...)`) does, and as OpenAI-ES
adds its `weight_decay` to the gradient estimate.
"""

from typing import Annotated

import msgspec
from msgspec import Meta

TITLE = "Adam"
SUMMARY = "Adam on the cross-entropy loss, with decoupled weight decay (AdamW)."


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    learning_rate: Annotated[float, Meta(gt=0, description="Step size η")] = 0.001
    beta1: Annotated[float, Meta(ge=0, lt=1, description="First-moment decay")] = 0.9
    beta2: Annotated[float, Meta(ge=0, lt=1, description="Second-moment decay")] = 0.999
    epsilon: Annotated[float, Meta(gt=0, description="Denominator floor")] = 1e-8
    weight_decay: Annotated[
        float, Meta(ge=0, description="Decoupled decay λ: each step θ -= η·λ·θ")
    ] = 0.0
    l2: Annotated[float, Meta(ge=0, description="Coupled L2: adds l2·θ to the gradient")] = 0.0
