import math

import torch

from kiseki.models import CNN2C2DMNIST, count_parameters
from kiseki.optimizers import (
    LEEAConfig,
    LEEARunner,
    SGDConfig,
    SGDRunner,
    copy_vector_to_model,
    flatten_parameters,
)


def synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))


def test_sgd_step_is_finite() -> None:
    model = CNN2C2DMNIST()
    runner = SGDRunner(model, SGDConfig(eta=0.01))
    loss = runner.step(*synthetic_batch())

    assert math.isfinite(loss)


def test_leea_step_is_finite() -> None:
    model = CNN2C2DMNIST()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=6, mutation_probability=0.1, initial_mutation_step=0.01),
        device=torch.device("cpu"),
        seed=7,
    )
    loss = runner.step(*synthetic_batch())

    assert math.isfinite(loss)


def test_leea_vectorized_reproduction_keeps_finite_population() -> None:
    model = CNN2C2DMNIST()
    runner = LEEARunner(
        model,
        LEEAConfig(
            population_size=8,
            mutation_probability=0.2,
            initial_mutation_step=0.01,
            retention_fraction=0.5,
        ),
        device=torch.device("cpu"),
        seed=11,
    )
    losses = torch.linspace(0.1, 1.0, steps=runner.population.shape[0])

    runner._select_and_reproduce(losses)

    assert runner.population.shape == (8, count_parameters(model))
    assert runner.inherited_fitness.shape == (8,)
    assert torch.isfinite(runner.population).all()
    assert torch.isfinite(runner.inherited_fitness).all()


def test_flatten_and_copy_support_channels_last_parameters() -> None:
    model = CNN2C2DMNIST().to(memory_format=torch.channels_last)
    vector = flatten_parameters(model)
    replacement = torch.linspace(-0.5, 0.5, steps=vector.numel(), dtype=vector.dtype)

    copy_vector_to_model(replacement, model)

    assert vector.numel() == count_parameters(model)
    assert model.conv1.weight.is_contiguous(memory_format=torch.channels_last)
    assert torch.allclose(flatten_parameters(model), replacement)
