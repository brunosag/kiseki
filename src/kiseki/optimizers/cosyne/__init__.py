"""CoSyNE: Cooperative Synapse Neuroevolution (Gomez, Schmidhuber & Miikkulainen, JMLR 2008).

Tournament crossover and Gaussian mutation make children; each weight position is also
permuted across the worse-ranked part of the population, recombining synapses.
"""

from typing import Annotated

import msgspec
from msgspec import Meta

SUMMARY = "Neuroevolution by crossover, mutation and per-weight permutation (Gomez et al. 2008)."


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    population_size: Annotated[int, Meta(ge=8, description="Networks per generation N")] = 1000
    tournament_size: Annotated[int, Meta(ge=1, description="Contestants per tournament")] = 4
    mutation_stdev: Annotated[float, Meta(ge=0, description="Gaussian mutation σ")] = 0.03
    mutation_probability: Annotated[
        float, Meta(ge=0, le=1, description="Chance each child weight mutates")
    ] = 1.0
    elitism_ratio: Annotated[
        float, Meta(ge=0, le=1, description="Share of best networks carried over unchanged")
    ] = 0.01
    permute_all: Annotated[
        bool, Meta(description="Permute every weight position instead of rank-weighted ones")
    ] = False
    sbx_eta: Annotated[
        float | None,
        Meta(description="Simulated binary crossover η; null uses one-point crossover"),
    ] = None
