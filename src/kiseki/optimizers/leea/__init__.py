"""LEEA: Limited Evaluation Evolutionary Algorithm (Morse & Stanley, GECCO 2016).

Each generation scores every network on one minibatch; offspring inherit a decayed share
of their parents' fitness so selection integrates evidence across batches.
"""

from typing import Annotated

import msgspec
from msgspec import Meta

SUMMARY = "Genetic algorithm with minibatch fitness and fitness inheritance (Morse & Stanley 2016)."


class Params(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    population_size: Annotated[int, Meta(ge=4, description="Networks per generation N")] = 1000
    mutation_probability: Annotated[
        float, Meta(ge=0, le=1, description="Chance each weight of an asexual child mutates")
    ] = 0.04
    initial_mutation_step: Annotated[
        float, Meta(ge=0, description="Initial half-width of uniform mutations η₀")
    ] = 0.03
    mutation_decay: Annotated[
        float, Meta(gt=0, le=1, description="Factor γ applied to the step when validation stalls")
    ] = 0.99
    validation_patience: Annotated[
        int, Meta(ge=1, description="Validations without improvement before decaying the step")
    ] = 5
    retention_fraction: Annotated[
        float, Meta(gt=0, le=1, description="Share of the population eligible as parents ρ")
    ] = 0.4
    crossover_fraction: Annotated[
        float, Meta(ge=0, le=1, description="Share of children made by crossover")
    ] = 0.5
    fitness_decay: Annotated[
        float, Meta(ge=0, le=1, description="Share λ of inherited fitness lost each generation")
    ] = 0.2
