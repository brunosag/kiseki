import math

import torch
from torch import nn

import kiseki.optimizers as optimizers
from kiseki.models import CNN2C2DMNIST, count_parameters
from kiseki.optimizers import (
    LEEAConfig,
    LEEARunner,
    SGDConfig,
    SGDRunner,
    copy_vector_to_model,
    flatten_parameters,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))


def tiny_batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(5, 3), torch.randint(0, 2, (5,))


def test_sgd_step_is_finite() -> None:
    model = CNN2C2DMNIST()
    runner = SGDRunner(model, SGDConfig(eta=0.01))
    loss = runner.step(*synthetic_batch())

    assert math.isfinite(loss)


def test_leea_step_is_finite() -> None:
    model = CNN2C2DMNIST()
    runner = LEEARunner(
        model,
        LEEAConfig(
            population_size=6,
            mutation_probability=0.1,
            initial_mutation_step=0.01,
            evaluation_chunk_size=2,
        ),
        device=torch.device("cpu"),
        seed=7,
    )
    loss = runner.step(*synthetic_batch())

    assert math.isfinite(loss)


def test_leea_generic_evaluator_supports_other_models() -> None:
    runner = LEEARunner(
        TinyClassifier(),
        LEEAConfig(population_size=4, evaluation_chunk_size=2),
        device=torch.device("cpu"),
        seed=7,
    )

    assert math.isfinite(runner.step(*tiny_batch()))


def test_leea_profile_records_phase_timings() -> None:
    runner = LEEARunner(
        CNN2C2DMNIST(),
        LEEAConfig(population_size=4, evaluation_chunk_size=2, profile=True),
        device=torch.device("cpu"),
        seed=7,
    )

    runner.step(*synthetic_batch())

    assert runner.last_profile is not None
    for key in (
        "evaluation_seconds",
        "fitness_selection_seconds",
        "reproduction_seconds",
        "scheduler_model_copy_seconds",
        "total_step_seconds",
    ):
        assert key in runner.last_profile
        assert runner.last_profile[key] >= 0.0
    assert runner.last_profile["total_step_seconds"] >= runner.last_profile["evaluation_seconds"]


def test_leea_reports_current_generation_loss_not_global_best(monkeypatch) -> None:
    model = TinyClassifier()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=4, mutation_probability=0.0, evaluation_chunk_size=2),
        device=torch.device("cpu"),
        seed=7,
    )
    loss_sequences = [
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([5.0, 6.0, 7.0, 8.0]),
    ]

    def fake_evaluate_population(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return loss_sequences.pop(0)

    monkeypatch.setattr(runner, "_evaluate_population", fake_evaluate_population)
    monkeypatch.setattr(runner, "_reproduce", lambda parent_indices: runner.population.clone())

    first_loss = runner.step(*tiny_batch())
    second_loss = runner.step(*tiny_batch())

    assert first_loss == 1.0
    assert second_loss == 5.0


def test_leea_fitness_inheritance_matches_parent_roles() -> None:
    model = TinyClassifier()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=4, crossover_fraction=0.5, fitness_decay=0.2),
        device=torch.device("cpu"),
        seed=11,
    )
    runner.is_first_step = False
    runner.inherited_fitness = torch.tensor([10.0, 20.0, 30.0, 40.0])
    runner.asexual_parent_indices = torch.tensor([1, 3])
    runner.sexual_parent_1_indices = torch.tensor([0, 2])
    runner.sexual_parent_2_indices = torch.tensor([2, 3])

    fitness = torch.ones(4)
    runner._inherit_fitness(fitness)

    assert torch.allclose(
        fitness,
        torch.tensor(
            [
                1.0 + 20.0 * 0.8,
                1.0 + 40.0 * 0.8,
                1.0 + (10.0 + 30.0) * 0.4,
                1.0 + (30.0 + 40.0) * 0.4,
            ]
        ),
    )


def test_leea_selects_weighted_parents_from_elite_wheel_without_sexual_collisions() -> None:
    model = TinyClassifier()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=6, retention_fraction=0.5, crossover_fraction=0.5),
        device=torch.device("cpu"),
        seed=13,
    )
    fitness = torch.tensor([0.1, 0.2, 0.3, 10.0, 20.0, 30.0])

    asexual, sexual_1, sexual_2 = runner._select_parents(fitness)
    elite = {3, 4, 5}

    assert set(asexual.tolist()).issubset(elite)
    assert set(sexual_1.tolist()).issubset(elite)
    assert set(sexual_2.tolist()).issubset(elite)
    assert torch.all(sexual_1 != sexual_2)


def test_leea_asexual_mutation_is_uniform_and_bounded() -> None:
    model = TinyClassifier()
    runner = LEEARunner(
        model,
        LEEAConfig(
            population_size=4,
            mutation_probability=1.0,
            initial_mutation_step=0.25,
            crossover_fraction=0.0,
        ),
        device=torch.device("cpu"),
        seed=17,
    )
    runner.population.zero_()

    children = runner._reproduce_asexual(torch.tensor([0, 1, 2, 3]))

    assert torch.all(children <= 0.25)
    assert torch.all(children >= -0.25)
    assert not torch.allclose(children, torch.zeros_like(children))


def test_leea_asexual_mutation_rate_is_approximately_configured() -> None:
    model = TinyClassifier()
    runner = LEEARunner(
        model,
        LEEAConfig(
            population_size=64,
            mutation_probability=0.25,
            initial_mutation_step=0.25,
            crossover_fraction=0.0,
        ),
        device=torch.device("cpu"),
        seed=18,
    )
    runner.population.zero_()

    children = runner._reproduce_asexual(torch.arange(runner.population_size))
    mutation_rate = float((children != 0.0).float().mean())

    assert 0.15 <= mutation_rate <= 0.35
    assert torch.all(children <= 0.25)
    assert torch.all(children >= -0.25)


def test_leea_sexual_reproduction_crosses_without_mutation() -> None:
    model = TinyClassifier()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=4, crossover_fraction=1.0),
        device=torch.device("cpu"),
        seed=19,
    )
    runner.population[0].fill_(0.0)
    runner.population[1].fill_(1.0)

    children = runner._reproduce_sexual(torch.tensor([0, 0]), torch.tensor([1, 1]))

    assert set(torch.unique(children).tolist()).issubset({0.0, 1.0})


def test_leea_scheduler_decays_only_after_validation_patience() -> None:
    model = TinyClassifier()
    runner = LEEARunner(
        model,
        LEEAConfig(
            population_size=4,
            initial_mutation_step=1.0,
            mutation_decay=0.5,
            validation_patience=2,
        ),
        device=torch.device("cpu"),
        seed=23,
    )

    runner.update_scheduler(is_best=False)
    assert runner.mutation_step == 1.0

    runner.update_scheduler(is_best=False)
    assert runner.mutation_step == 0.5
    assert runner.validation_patience == 0

    runner.update_scheduler(is_best=False)
    runner.update_scheduler(is_best=True)
    assert runner.mutation_step == 0.5
    assert runner.validation_patience == 0


def test_leea_chunked_evaluator_matches_individual_loop() -> None:
    model = CNN2C2DMNIST()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=5, evaluation_chunk_size=2),
        device=torch.device("cpu"),
        seed=29,
    )
    inputs, targets = synthetic_batch()

    chunked_losses = runner._evaluate_population(inputs, targets)
    loop_losses = runner._evaluate_population_loop(inputs, targets)

    assert torch.allclose(chunked_losses, loop_losses, atol=1e-5, rtol=1e-5)


def test_leea_chunk_sizes_produce_equivalent_losses() -> None:
    model = CNN2C2DMNIST()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=5, evaluation_chunk_size=1),
        device=torch.device("cpu"),
        seed=31,
    )
    inputs, targets = synthetic_batch()

    losses_chunk_1 = runner._evaluate_population(inputs, targets)
    runner.manual_evaluation_chunk_size = 3
    runner.evaluation_chunk_size = 3
    losses_chunk_3 = runner._evaluate_population(inputs, targets)

    assert torch.allclose(losses_chunk_1, losses_chunk_3, atol=1e-5, rtol=1e-5)


def test_leea_auto_chunk_size_uses_available_memory(monkeypatch) -> None:
    monkeypatch.setattr(optimizers, "available_memory_bytes", lambda device: 4096)

    chunk_size = optimizers.estimate_leea_chunk_size(
        device=torch.device("cpu"),
        population_size=100,
        parameter_bytes=128,
        input_bytes=128,
    )

    assert chunk_size == 1


def test_leea_auto_chunk_size_recomputes_for_batch_shape(monkeypatch) -> None:
    model = TinyClassifier()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=20),
        device=torch.device("cpu"),
        seed=37,
    )
    monkeypatch.setattr(optimizers, "available_memory_bytes", lambda device: 1_000_000)

    small_inputs = torch.randn(2, 3)
    small_targets = torch.randint(0, 2, (2,))
    large_inputs = torch.randn(64, 3)
    large_targets = torch.randint(0, 2, (64,))

    small_chunk = runner._resolve_evaluation_chunk_size(small_inputs, small_targets)
    large_chunk = runner._resolve_evaluation_chunk_size(large_inputs, large_targets)

    assert small_chunk > large_chunk


def test_leea_manual_chunk_size_overrides_auto(monkeypatch) -> None:
    model = TinyClassifier()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=20, evaluation_chunk_size=3),
        device=torch.device("cpu"),
        seed=41,
    )
    monkeypatch.setattr(optimizers, "available_memory_bytes", lambda device: 1_000_000)

    assert runner._resolve_evaluation_chunk_size(torch.randn(2, 3), torch.randint(0, 2, (2,))) == 3


def test_leea_chunk_autotune_grows_past_conservative_estimate() -> None:
    attempted_chunk_sizes = []

    def can_evaluate(chunk_size: int) -> bool:
        attempted_chunk_sizes.append(chunk_size)
        return chunk_size <= 80

    chunk_size = optimizers.find_largest_safe_chunk_size(
        population_size=1000,
        initial_chunk_size=19,
        can_evaluate=can_evaluate,
    )

    assert chunk_size == 80
    assert max(attempted_chunk_sizes) > 80


def test_leea_reproduction_keeps_finite_population() -> None:
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
    fitness = runner._compute_fitness(losses)
    parent_indices = runner._select_parents(fitness)

    runner.population = runner._reproduce(parent_indices)
    runner.inherited_fitness = fitness

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
