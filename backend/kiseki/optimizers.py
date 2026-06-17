from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from .schemas import ETA, ETA_0, GAMMA, LAMBDA, P_M, RHO, RHO_X, TAU_PAT


class OptimizerRunner(Protocol):
    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        ...


@dataclass(slots=True)
class SGDConfig:
    eta: float = 0.01


@dataclass(slots=True)
class LEEAConfig:
    population_size: int = 200
    mutation_probability: float = 0.04
    initial_mutation_step: float = 0.03
    mutation_decay: float = 0.99
    retention_fraction: float = 0.4
    crossover_fraction: float = 0.5
    fitness_decay: float = 0.2
    validation_patience: int = 25


class SGDRunner:
    def __init__(self, model: nn.Module, config: SGDConfig) -> None:
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config.eta)

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())


class LEEARunner:
    def __init__(
        self,
        model: nn.Module,
        config: LEEAConfig,
        *,
        device: torch.device,
        seed: int,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.mutation_step = config.initial_mutation_step
        self.parameter_shapes = [parameter.shape for parameter in model.parameters()]
        self.parameter_sizes = [parameter.numel() for parameter in model.parameters()]
        self.num_parameters = sum(self.parameter_sizes)

        base = flatten_parameters(model).to(device)
        population_size = max(2, config.population_size)
        self.population = base.repeat(population_size, 1)
        self.population[1:] += torch.randn(
            (population_size - 1, self.num_parameters),
            generator=self.generator,
            device=device,
        ) * config.initial_mutation_step
        self.inherited_fitness = torch.zeros(population_size, device=device)
        self.best_vector = base.clone()
        self.best_loss = float("inf")

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.eval()
        losses = torch.empty(self.population.shape[0], device=self.device)

        with torch.no_grad():
            for index, vector in enumerate(self.population):
                copy_vector_to_model(vector, self.model)
                logits = self.model(inputs)
                losses[index] = F.cross_entropy(logits, targets)

        finite_losses = torch.nan_to_num(losses, nan=float("inf"), posinf=float("inf"))
        current_best_loss, current_best_index = finite_losses.min(dim=0)
        if float(current_best_loss) < self.best_loss:
            self.best_loss = float(current_best_loss.detach().cpu())
            self.best_vector = self.population[int(current_best_index)].detach().clone()

        copy_vector_to_model(self.best_vector, self.model)
        self._select_and_reproduce(finite_losses)
        self.mutation_step *= self.config.mutation_decay
        return self.best_loss

    def _select_and_reproduce(self, losses: torch.Tensor) -> None:
        fitness = -losses + (1.0 - self.config.fitness_decay) * self.inherited_fitness
        population_size = self.population.shape[0]
        retain_count = min(
            population_size,
            max(2, int(round(population_size * self.config.retention_fraction))),
        )
        retained_indices = torch.topk(fitness, k=retain_count).indices
        retained = self.population[retained_indices].clone()
        retained_fitness = fitness[retained_indices].clone()

        next_population = torch.empty_like(self.population)
        next_fitness = torch.empty_like(self.inherited_fitness)
        next_population[:retain_count] = retained
        next_fitness[:retain_count] = retained_fitness

        child_count = population_size - retain_count
        if child_count > 0:
            parent_a = torch.randint(
                retain_count,
                (child_count,),
                generator=self.generator,
                device=self.device,
            )
            parent_b = torch.randint(
                retain_count,
                (child_count,),
                generator=self.generator,
                device=self.device,
            )
            children = retained[parent_a].clone()
            child_fitness = retained_fitness[parent_a].clone()

            crossover_rows = (
                torch.rand(child_count, generator=self.generator, device=self.device)
                < self.config.crossover_fraction
            )
            if bool(crossover_rows.any()):
                crossover_mask = (
                    torch.rand(
                        (child_count, self.num_parameters),
                        generator=self.generator,
                        device=self.device,
                    )
                    < 0.5
                )
                crossed_children = torch.where(crossover_mask, children, retained[parent_b])
                children = torch.where(crossover_rows[:, None], crossed_children, children)
                averaged_fitness = 0.5 * (retained_fitness[parent_a] + retained_fitness[parent_b])
                child_fitness = torch.where(crossover_rows, averaged_fitness, child_fitness)

            mutation_mask = (
                torch.rand(
                    (child_count, self.num_parameters),
                    generator=self.generator,
                    device=self.device,
                )
                < self.config.mutation_probability
            )
            if bool(mutation_mask.any()):
                noise = torch.randn(
                    (child_count, self.num_parameters),
                    generator=self.generator,
                    device=self.device,
                ) * self.mutation_step
                children = children + noise * mutation_mask.to(children.dtype)

            next_population[retain_count:] = children
            next_fitness[retain_count:] = child_fitness

        self.population = next_population
        self.inherited_fitness = next_fitness


def random_scalar(generator: torch.Generator, device: torch.device) -> float:
    return float(torch.rand((), generator=generator, device=device).cpu())


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    chunks = [parameter.detach().reshape(-1) for parameter in model.parameters()]
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks).detach()


def copy_vector_to_model(vector: torch.Tensor, model: nn.Module) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            width = parameter.numel()
            source = vector[offset : offset + width].reshape(parameter.shape)
            if source.device != parameter.device or source.dtype != parameter.dtype:
                source = source.to(device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(source)
            offset += width


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_optimizer_runner(
    optimizer_name: str,
    model: nn.Module,
    raw_params: dict[str, float],
    *,
    device: torch.device,
    seed: int,
) -> OptimizerRunner:
    if optimizer_name == "SGD":
        return SGDRunner(
            model,
            SGDConfig(eta=float(raw_params.get(ETA, 0.01))),
        )
    if optimizer_name == "LEEA":
        config = LEEAConfig(
            population_size=max(2, int(raw_params.get("N", 200))),
            mutation_probability=clamp(float(raw_params.get(P_M, 0.04)), 0.0, 1.0),
            initial_mutation_step=max(0.0, float(raw_params.get(ETA_0, 0.03))),
            mutation_decay=max(0.0, float(raw_params.get(GAMMA, 0.99))),
            retention_fraction=clamp(float(raw_params.get(RHO, 0.4)), 0.01, 1.0),
            crossover_fraction=clamp(float(raw_params.get(RHO_X, 0.5)), 0.0, 1.0),
            fitness_decay=clamp(float(raw_params.get(LAMBDA, 0.2)), 0.0, 1.0),
            validation_patience=max(1, int(raw_params.get(TAU_PAT, 25))),
        )
        return LEEARunner(model, config, device=device, seed=seed)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
