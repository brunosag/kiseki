"""OpenAI-ES: natural evolution strategy with mirrored sampling (Salimans et al., 2017).

Estimates the gradient of the expected loss from rank-shaped scores of Gaussian
perturbations, then takes an Adam step. Noise is regenerated from its key instead of
stored, so memory does not grow with the population.
"""

from typing import Annotated

import msgspec
from msgspec import Meta

TITLE = "OpenAI-ES"
SUMMARY = (
    "Evolution strategy: a gradient estimate from mirrored Gaussian perturbations, "
    "Adam step (Salimans et al. 2017)."
)


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    population_size: Annotated[
        int, Meta(ge=2, multiple_of=2, description="Perturbations per step, in mirrored pairs")
    ] = 1000
    sigma: Annotated[float, Meta(gt=0, description="Perturbation scale σ")] = 0.02
    learning_rate: Annotated[float, Meta(gt=0, description="Adam step size")] = 0.01
    weight_decay: Annotated[float, Meta(ge=0, description="L2 coefficient")] = 0.005
    beta1: Annotated[float, Meta(ge=0, lt=1, description="Adam first-moment decay")] = 0.9
    beta2: Annotated[float, Meta(ge=0, lt=1, description="Adam second-moment decay")] = 0.999
