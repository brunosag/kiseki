import math

import torch

from kiseki.models import CNN2C2DMNIST
from kiseki.optimizers import LEEAConfig, LEEARunner, SGDConfig, SGDRunner


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
