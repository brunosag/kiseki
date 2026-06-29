import os
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn
from torch.func import functional_call, vmap
from torch.nn import functional as F

from .schemas import (
    COSYNE_P_M,
    ETA,
    ETA_0,
    ETA_SBX,
    GAMMA,
    LAMBDA,
    NUM_CHILDREN,
    PERMUTE_ALL,
    P_M,
    RHO,
    RHO_E,
    RHO_X,
    SIGMA_M,
    TAU_PAT,
    TOURNAMENT_SIZE,
)

CUDA_REPRODUCTION_WORK_CHUNK_BYTES = 64 * 1024 * 1024
CPU_REPRODUCTION_WORK_CHUNK_BYTES = 256 * 1024 * 1024


class OptimizerRunner(Protocol):
    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...


@dataclass(slots=True)
class SGDConfig:
    eta: float = 0.01


@dataclass(slots=True)
class LEEAConfig:
    population_size: int = 1000
    mutation_probability: float = 0.04
    initial_mutation_step: float = 0.03
    mutation_decay: float = 0.99
    retention_fraction: float = 0.4
    crossover_fraction: float = 0.5
    fitness_decay: float = 0.2
    validation_patience: int = 5
    evaluation_chunk_size: int | None = None
    profile: bool = False


@dataclass(slots=True)
class CosyneConfig:
    population_size: int = 1000
    tournament_size: int = 4
    mutation_stdev: float = 0.03
    mutation_probability: float = 1.0
    permute_all: bool = False
    elitism_ratio: float = 0.01
    sbx_eta: float | None = None
    num_children: int | None = None
    evaluation_chunk_size: int | None = None


class SGDRunner:
    def __init__(self, model: nn.Module, config: SGDConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config.eta)

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {"eta": self.config.eta},
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)


class PopulationEvaluatorMixin:
    model: nn.Module
    buffers: dict[str, torch.Tensor]
    device: torch.device
    evaluation_chunk_size: int | None
    generator: torch.Generator
    manual_evaluation_chunk_size: int | None
    num_parameters: int
    parameter_specs: list[tuple[str, torch.Size, int]]
    population: torch.Tensor
    population_size: int
    profile_enabled: bool
    _auto_chunk_signature: tuple[tuple[int, ...], torch.dtype, str] | None

    def _evaluate_population(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._evaluate_vectors(self.population, inputs, targets)

    def _evaluate_vectors(
        self,
        vectors: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        population_size = int(vectors.shape[0])
        chunk_size = self._resolve_evaluation_chunk_size(
            inputs,
            targets,
            population_size=population_size,
            vectors=vectors,
        )
        losses = torch.empty(population_size, device=self.device)
        with torch.no_grad():
            start = 0
            while start < population_size:
                end = min(start + chunk_size, population_size)
                chunk = vectors[start:end]
                try:
                    losses[start:end] = self._evaluate_population_chunk(chunk, inputs, targets)
                except RuntimeError as exc:
                    if self.device.type == "cuda" and chunk_size > 1 and is_cuda_out_of_memory(exc):
                        torch.cuda.empty_cache()
                        chunk_size = max(1, chunk_size // 2)
                        self.evaluation_chunk_size = chunk_size
                        continue
                    raise
                start = end
        return losses

    def _resolve_evaluation_chunk_size(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        population_size: int | None = None,
        vectors: torch.Tensor | None = None,
    ) -> int:
        population_size = self.population_size if population_size is None else population_size
        vectors = self.population if vectors is None else vectors
        if self.manual_evaluation_chunk_size is not None and self.manual_evaluation_chunk_size > 0:
            return min(population_size, self.manual_evaluation_chunk_size)

        signature = (tuple(inputs.shape), inputs.dtype, self.device.type)
        if self.evaluation_chunk_size is None or self._auto_chunk_signature != signature:
            estimated_chunk_size = estimate_leea_chunk_size(
                device=self.device,
                population_size=population_size,
                parameter_bytes=self.num_parameters * vectors.element_size(),
                input_bytes=inputs.numel() * inputs.element_size(),
            )
            if self.device.type == "cuda":
                self.evaluation_chunk_size = self._autotune_cuda_chunk_size(
                    inputs,
                    targets,
                    estimated_chunk_size,
                    population_size=population_size,
                    vectors=vectors,
                )
            else:
                self.evaluation_chunk_size = estimated_chunk_size
            self._auto_chunk_signature = signature

        return min(population_size, max(1, self.evaluation_chunk_size))

    def _autotune_cuda_chunk_size(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        initial_chunk_size: int,
        *,
        population_size: int | None = None,
        vectors: torch.Tensor | None = None,
    ) -> int:
        population_size = self.population_size if population_size is None else population_size
        vectors = self.population if vectors is None else vectors
        return find_largest_safe_chunk_size(
            population_size=population_size,
            initial_chunk_size=initial_chunk_size,
            can_evaluate=lambda chunk_size: self._can_evaluate_chunk_size(
                chunk_size,
                inputs,
                targets,
                vectors=vectors,
            ),
        )

    def _can_evaluate_chunk_size(
        self,
        chunk_size: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        vectors: torch.Tensor | None = None,
    ) -> bool:
        vectors = self.population if vectors is None else vectors
        try:
            with torch.no_grad():
                losses = self._evaluate_population_chunk(
                    vectors[:chunk_size],
                    inputs,
                    targets,
                )
                if self.profile_enabled and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                del losses
        except RuntimeError as exc:
            if self.device.type == "cuda" and is_cuda_out_of_memory(exc):
                torch.cuda.empty_cache()
                return False
            raise

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return True

    def _evaluate_population_chunk(
        self,
        chunk: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self._chunk_losses_generic(chunk, inputs, targets)

    def _chunk_losses_eager(
        self,
        chunk: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self._chunk_losses_generic(chunk, inputs, targets)

    def _chunk_losses_generic(
        self,
        chunk: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        params = self._unflatten_population_chunk(chunk)

        def call_model(one_params: dict[str, torch.Tensor]) -> torch.Tensor:
            return functional_call(self.model, (one_params, self.buffers), (inputs,))

        logits = vmap(call_model)(params)
        return population_cross_entropy_losses(logits, targets)

    def _evaluate_population_loop(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        losses = torch.empty(self.population_size, device=self.device)
        with torch.no_grad():
            for index, vector in enumerate(self.population):
                copy_vector_to_model(vector, self.model)
                logits = self.model(inputs)
                losses[index] = F.cross_entropy(logits, targets)
        return losses

    def _unflatten_population_chunk(self, chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        params = {}
        offset = 0
        for name, shape, size in self.parameter_specs:
            params[name] = chunk[:, offset : offset + size].reshape(chunk.shape[0], *shape)
            offset += size
        return params


class LEEARunner(PopulationEvaluatorMixin):
    def __init__(
        self,
        model: nn.Module,
        config: LEEAConfig,
        *,
        device: torch.device,
        seed: int,
    ) -> None:
        self.model = model
        self.model.eval()
        self.config = config
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.mutation_step = config.initial_mutation_step
        self.validation_patience = 0
        self.is_first_step = True
        self.profile_enabled = config.profile
        self.last_profile: dict[str, float] | None = None

        self.parameter_specs = [
            (name, parameter.shape, parameter.numel())
            for name, parameter in model.named_parameters()
        ]
        self.parameter_shapes = [shape for _, shape, _ in self.parameter_specs]
        self.parameter_sizes = [size for _, _, size in self.parameter_specs]
        self.num_parameters = sum(self.parameter_sizes)
        self.buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

        self.population_size = max(2, config.population_size)
        self.asexual_count, self.sexual_count = split_reproduction_counts(
            self.population_size,
            config.crossover_fraction,
        )
        self.manual_evaluation_chunk_size = config.evaluation_chunk_size
        self.evaluation_chunk_size = config.evaluation_chunk_size
        self._auto_chunk_signature: tuple[tuple[int, ...], torch.dtype, str] | None = None

        self.population = initialize_population(model, self.population_size, seed, device)
        self._next_population = torch.empty_like(self.population)
        self.inherited_fitness = torch.zeros(self.population_size, device=device)
        self.asexual_parent_indices = torch.empty(
            self.asexual_count,
            dtype=torch.long,
            device=device,
        )
        self.sexual_parent_1_indices = torch.empty(
            self.sexual_count,
            dtype=torch.long,
            device=device,
        )
        self.sexual_parent_2_indices = torch.empty(
            self.sexual_count,
            dtype=torch.long,
            device=device,
        )
        copy_vector_to_model(self.population[0], self.model)

    def state_dict(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "next_population": self._next_population,
            "inherited_fitness": self.inherited_fitness,
            "asexual_parent_indices": self.asexual_parent_indices,
            "sexual_parent_1_indices": self.sexual_parent_1_indices,
            "sexual_parent_2_indices": self.sexual_parent_2_indices,
            "mutation_step": self.mutation_step,
            "validation_patience": self.validation_patience,
            "is_first_step": self.is_first_step,
            "manual_evaluation_chunk_size": self.manual_evaluation_chunk_size,
            "evaluation_chunk_size": self.evaluation_chunk_size,
            "auto_chunk_signature": self._auto_chunk_signature,
            "generator_state": self.generator.get_state(),
            "last_profile": self.last_profile,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.population = state["population"].to(self.device)
        self._next_population = state.get(
            "next_population",
            torch.empty_like(self.population),
        ).to(self.device)
        self.inherited_fitness = state["inherited_fitness"].to(self.device)
        self.asexual_parent_indices = state["asexual_parent_indices"].to(self.device)
        self.sexual_parent_1_indices = state["sexual_parent_1_indices"].to(self.device)
        self.sexual_parent_2_indices = state["sexual_parent_2_indices"].to(self.device)
        self.population_size = int(self.population.shape[0])
        self.mutation_step = float(state["mutation_step"])
        self.validation_patience = int(state["validation_patience"])
        self.is_first_step = bool(state["is_first_step"])
        self.manual_evaluation_chunk_size = state.get("manual_evaluation_chunk_size")
        self.evaluation_chunk_size = state.get("evaluation_chunk_size")
        self._auto_chunk_signature = state.get("auto_chunk_signature")
        self.last_profile = state.get("last_profile")

        generator_state = state.get("generator_state")
        if generator_state is not None:
            self.generator.set_state(generator_state)

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        if self.profile_enabled:
            return self._step_profiled(inputs, targets)

        self.model.eval()
        losses = self._evaluate_population(inputs, targets)
        finite_losses = torch.nan_to_num(losses, nan=float("inf"), posinf=float("inf"))
        current_best_loss, current_best_index = finite_losses.min(dim=0)
        best_vector = self.population[int(current_best_index.detach().cpu())]

        fitness = self._compute_fitness(finite_losses)
        self._inherit_fitness(fitness)
        parent_indices = self._select_parents(fitness)
        next_population = self._reproduce(parent_indices)

        self.inherited_fitness = fitness
        (
            self.asexual_parent_indices,
            self.sexual_parent_1_indices,
            self.sexual_parent_2_indices,
        ) = parent_indices
        self.is_first_step = False

        copy_vector_to_model(best_vector, self.model)
        self.population, self._next_population = next_population, self.population
        return float(current_best_loss.detach().cpu())

    def _step_profiled(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.eval()
        started_at = self._profile_checkpoint()
        losses = self._evaluate_population(inputs, targets)
        evaluated_at = self._profile_checkpoint()

        finite_losses = torch.nan_to_num(losses, nan=float("inf"), posinf=float("inf"))
        current_best_loss, current_best_index = finite_losses.min(dim=0)
        best_vector = self.population[int(current_best_index.detach().cpu())]
        fitness = self._compute_fitness(finite_losses)
        self._inherit_fitness(fitness)
        parent_indices = self._select_parents(fitness)
        selected_at = self._profile_checkpoint()

        next_population = self._reproduce(parent_indices)
        reproduced_at = self._profile_checkpoint()

        self.inherited_fitness = fitness
        (
            self.asexual_parent_indices,
            self.sexual_parent_1_indices,
            self.sexual_parent_2_indices,
        ) = parent_indices
        self.is_first_step = False
        copy_vector_to_model(best_vector, self.model)
        self.population, self._next_population = next_population, self.population
        copied_at = self._profile_checkpoint()

        self.last_profile = {
            "evaluation_seconds": evaluated_at - started_at,
            "fitness_selection_seconds": selected_at - evaluated_at,
            "reproduction_seconds": reproduced_at - selected_at,
            "scheduler_model_copy_seconds": copied_at - reproduced_at,
            "total_step_seconds": copied_at - started_at,
        }
        return float(current_best_loss.detach().cpu())

    def _profile_checkpoint(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def update_scheduler(self, is_best: bool) -> None:
        if is_best:
            self.validation_patience = 0
            return

        self.validation_patience += 1
        if self.validation_patience >= self.config.validation_patience:
            self.mutation_step *= self.config.mutation_decay
            self.validation_patience = 0

    def _compute_fitness(self, losses: torch.Tensor) -> torch.Tensor:
        return torch.where(
            torch.isfinite(losses),
            1.0 / (1.0 + losses),
            torch.zeros_like(losses),
        )

    def _inherit_fitness(self, fitness: torch.Tensor) -> None:
        if self.is_first_step:
            return

        decay = 1.0 - self.config.fitness_decay
        if self.asexual_count:
            fitness[: self.asexual_count] += (
                self.inherited_fitness[self.asexual_parent_indices] * decay
            )
        if self.sexual_count:
            sexual_start = self.asexual_count
            fitness[sexual_start:] += (
                self.inherited_fitness[self.sexual_parent_1_indices]
                + self.inherited_fitness[self.sexual_parent_2_indices]
            ) * (0.5 * decay)

    def _select_parents(
        self,
        fitness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        elite_count = self._elite_count()
        wheel = torch.topk(fitness, k=elite_count).indices
        weights = torch.clamp(fitness[wheel], min=0.0)
        if not bool(torch.isfinite(weights).all()) or float(weights.sum().detach().cpu()) <= 0.0:
            weights = torch.ones_like(weights)

        asexual = self._sample_from_wheel(wheel, weights, self.asexual_count)
        sexual_1 = self._sample_from_wheel(wheel, weights, self.sexual_count)
        sexual_2 = self._sample_from_wheel(wheel, weights, self.sexual_count)

        if elite_count > 1 and self.sexual_count:
            collisions = sexual_1 == sexual_2
            while bool(collisions.any()):
                sexual_2[collisions] = self._sample_from_wheel(
                    wheel,
                    weights,
                    int(collisions.sum().detach().cpu()),
                )
                collisions = sexual_1 == sexual_2

        return asexual, sexual_1, sexual_2

    def _elite_count(self) -> int:
        minimum = 2 if self.sexual_count else 1
        count = int(round(self.population_size * self.config.retention_fraction))
        return min(self.population_size, max(minimum, count))

    def _sample_from_wheel(
        self,
        wheel: torch.Tensor,
        weights: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        positions = torch.multinomial(
            weights,
            count,
            replacement=True,
            generator=self.generator,
        )
        return wheel[positions]

    def _reproduce(
        self,
        parent_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        asexual, sexual_1, sexual_2 = parent_indices
        next_population = self._next_population
        if next_population.shape != self.population.shape:
            next_population = torch.empty_like(self.population)
            self._next_population = next_population
        elif next_population.data_ptr() == self.population.data_ptr():
            next_population = torch.empty_like(self.population)
            self._next_population = next_population

        if self.asexual_count:
            self._reproduce_asexual(asexual, out=next_population[: self.asexual_count])

        if self.sexual_count:
            sexual_start = self.asexual_count
            self._reproduce_sexual(
                sexual_1,
                sexual_2,
                out=next_population[sexual_start:],
            )

        return next_population

    def _reproduce_asexual(
        self,
        parent_indices: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        children = out
        if children is None:
            children = torch.empty(
                (parent_indices.numel(), self.num_parameters),
                device=self.device,
                dtype=self.population.dtype,
            )

        chunk_rows = reproduction_chunk_size(
            self.num_parameters,
            self.population.element_size(),
            self.device,
        )
        for start in range(0, parent_indices.numel(), chunk_rows):
            end = min(start + chunk_rows, parent_indices.numel())
            child_slice = children[start:end]
            child_slice.copy_(self.population[parent_indices[start:end]])
            self._mutate_asexual_children(child_slice)

        return children

    def _mutate_asexual_children(self, children: torch.Tensor) -> None:
        if (
            children.numel() == 0
            or self.config.mutation_probability <= 0.0
            or self.mutation_step == 0.0
        ):
            return

        mutation_mask = (
            torch.rand(
                children.shape,
                generator=self.generator,
                device=self.device,
            )
            < self.config.mutation_probability
        )
        mutation_coordinates = mutation_mask.nonzero(as_tuple=True)
        mutation_count = mutation_coordinates[0].numel()
        if mutation_count == 0:
            return

        mutation = (
            2.0
            * torch.rand(
                (mutation_count,),
                generator=self.generator,
                device=self.device,
            )
            - 1.0
        ) * self.mutation_step
        children[mutation_coordinates] += mutation.to(dtype=children.dtype)

    def _reproduce_sexual(
        self,
        parent_1_indices: torch.Tensor,
        parent_2_indices: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        children = out
        if children is None:
            children = torch.empty(
                (parent_1_indices.numel(), self.num_parameters),
                device=self.device,
                dtype=self.population.dtype,
            )

        chunk_rows = reproduction_chunk_size(
            self.num_parameters,
            self.population.element_size(),
            self.device,
        )
        for start in range(0, parent_1_indices.numel(), chunk_rows):
            end = min(start + chunk_rows, parent_1_indices.numel())
            parent_1 = self.population[parent_1_indices[start:end]]
            parent_2 = self.population[parent_2_indices[start:end]]
            crossover_mask = (
                torch.rand(
                    parent_1.shape,
                    generator=self.generator,
                    device=self.device,
                )
                < 0.5
            )
            torch.where(crossover_mask, parent_1, parent_2, out=children[start:end])

        return children


class CosyneRunner(PopulationEvaluatorMixin):
    def __init__(
        self,
        model: nn.Module,
        config: CosyneConfig,
        *,
        device: torch.device,
        seed: int,
    ) -> None:
        self.model = model
        self.model.eval()
        self.config = config
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.profile_enabled = False

        self.parameter_specs = [
            (name, parameter.shape, parameter.numel())
            for name, parameter in model.named_parameters()
        ]
        self.parameter_shapes = [shape for _, shape, _ in self.parameter_specs]
        self.parameter_sizes = [size for _, _, size in self.parameter_specs]
        self.num_parameters = sum(self.parameter_sizes)
        self.buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

        self.population_size = max(4, int(config.population_size))
        self.manual_evaluation_chunk_size = config.evaluation_chunk_size
        self.evaluation_chunk_size = config.evaluation_chunk_size
        self._auto_chunk_signature: tuple[tuple[int, ...], torch.dtype, str] | None = None

        self.population = initialize_population(model, self.population_size, seed, device)
        self.population_losses = torch.full(
            (self.population_size,),
            float("inf"),
            device=device,
            dtype=self.population.dtype,
        )
        self.is_first_generation = True
        copy_vector_to_model(self.population[0], self.model)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "population_size": self.config.population_size,
                "tournament_size": self.config.tournament_size,
                "mutation_stdev": self.config.mutation_stdev,
                "mutation_probability": self.config.mutation_probability,
                "permute_all": self.config.permute_all,
                "elitism_ratio": self.config.elitism_ratio,
                "sbx_eta": self.config.sbx_eta,
                "num_children": self.config.num_children,
            },
            "population": self.population,
            "population_losses": self.population_losses,
            "is_first_generation": self.is_first_generation,
            "manual_evaluation_chunk_size": self.manual_evaluation_chunk_size,
            "evaluation_chunk_size": self.evaluation_chunk_size,
            "auto_chunk_signature": self._auto_chunk_signature,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.population = state["population"].to(self.device)
        self.population_size = int(self.population.shape[0])
        population_losses = state.get("population_losses")
        if population_losses is None:
            self.population_losses = torch.full(
                (self.population_size,),
                float("inf"),
                device=self.device,
                dtype=self.population.dtype,
            )
        else:
            self.population_losses = population_losses.to(self.device)
        self.is_first_generation = bool(state.get("is_first_generation", population_losses is None))
        self.manual_evaluation_chunk_size = state.get("manual_evaluation_chunk_size")
        self.evaluation_chunk_size = state.get("evaluation_chunk_size")
        self._auto_chunk_signature = state.get("auto_chunk_signature")

        generator_state = state.get("generator_state")
        if generator_state is not None:
            self.generator.set_state(generator_state)

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.eval()
        self.population_losses = self._evaluate_population(inputs, targets)
        self.is_first_generation = False

        parent_losses = torch.nan_to_num(
            self.population_losses,
            nan=float("inf"),
            posinf=float("inf"),
        )
        sorted_indices = torch.argsort(parent_losses)
        num_elites = self._num_elites()
        num_parents = int(self.population_size / 4)
        num_relevant = max(num_elites, num_parents)
        relevant_indices = sorted_indices[:num_relevant]
        sorted_relevant = self.population[relevant_indices]
        sorted_relevant_losses = parent_losses[relevant_indices]

        to_merge: list[torch.Tensor] = []
        if num_elites >= 1:
            to_merge.append(sorted_relevant[:num_elites].clone())

        parents = sorted_relevant[:num_parents]
        parent_subset_losses = sorted_relevant_losses[:num_parents]
        children = self._crossover(parents, parent_subset_losses)
        children = self._mutate(children)

        permuted = self._cosyne_permutation(self.population, parent_losses)
        to_merge.extend([children, permuted])

        merged_population = torch.cat(to_merge, dim=0)
        merged_losses = self._evaluate_vectors(merged_population, inputs, targets)
        finite_merged_losses = torch.nan_to_num(
            merged_losses,
            nan=float("inf"),
            posinf=float("inf"),
        )
        selected_indices = torch.argsort(finite_merged_losses)[: self.population_size]

        self.population = merged_population[selected_indices].clone()
        self.population_losses = merged_losses[selected_indices].clone()
        best_loss = finite_merged_losses[selected_indices[0]]
        copy_vector_to_model(self.population[0], self.model)
        return float(best_loss.detach().cpu())

    def _num_elites(self) -> int:
        return min(
            self.population_size,
            max(0, int(self.population_size * self.config.elitism_ratio)),
        )

    def _crossover(self, parents: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
        parent_1, parent_2 = self._tournament_parent_pairs(parents, losses)
        if self.config.sbx_eta is None:
            return self._one_point_crossover(parent_1, parent_2)
        return self._simulated_binary_crossover(parent_1, parent_2, self.config.sbx_eta)

    def _tournament_parent_pairs(
        self,
        candidates: torch.Tensor,
        losses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tournaments = self._num_tournaments(len(candidates))
        if num_tournaments == 0:
            empty = candidates[:0]
            return empty, empty

        ranks = centered_ranks(losses, higher_is_better=False)
        tournament_indices = torch.randint(
            len(candidates),
            (num_tournaments, max(1, int(self.config.tournament_size))),
            generator=self.generator,
            device=self.device,
        )
        tournament_ranks = ranks[tournament_indices]
        tournament_rows = torch.arange(0, num_tournaments, device=self.device)
        selected = tournament_indices[
            tournament_rows,
            torch.argmax(tournament_ranks, dim=-1),
        ]
        split_point = int(len(selected) / 2)
        return candidates[selected][:split_point], candidates[selected][split_point:]

    def _num_tournaments(self, candidate_count: int) -> int:
        if candidate_count <= 0:
            return 0

        if self.config.num_children is None:
            f = candidate_count * 2.0
            result1 = math.ceil(f)
            result2 = math.floor(f)
            if result1 == result2:
                result = result1
                if (result % 2) != 0:
                    result += 1
            else:
                result = result1 if (result1 % 2) == 0 else result2
            return max(0, result)

        num_children = int(self.config.num_children)
        if (num_children % 2) != 0:
            raise ValueError("num_children must be even")
        return max(0, num_children)

    def _one_point_crossover(
        self,
        parents_1: torch.Tensor,
        parents_2: torch.Tensor,
    ) -> torch.Tensor:
        if parents_1.numel() == 0:
            return torch.empty(
                (0, self.num_parameters),
                device=self.device,
                dtype=self.population.dtype,
            )

        num_pairings = parents_1.shape[0]
        solution_length = parents_1.shape[1]
        if solution_length <= 1:
            return torch.cat([parents_1, parents_2], dim=0)

        gene_indices = (
            torch.arange(0, solution_length, device=self.device)
            .unsqueeze(0)
            .expand(num_pairings, solution_length)
        )
        crossover_point = (
            torch.randint(
                solution_length - 1,
                (num_pairings, 1),
                generator=self.generator,
                device=self.device,
            )
            + 1
        )
        crossover_mask = gene_indices >= crossover_point
        children_1 = torch.where(crossover_mask, parents_1, parents_2)
        children_2 = torch.where(crossover_mask, parents_2, parents_1)
        return torch.cat([children_1, children_2], dim=0)

    def _simulated_binary_crossover(
        self,
        parents_1: torch.Tensor,
        parents_2: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        if parents_1.numel() == 0:
            return torch.empty(
                (0, self.num_parameters),
                device=self.device,
                dtype=self.population.dtype,
            )

        u = torch.rand(
            parents_1.shape,
            generator=self.generator,
            device=self.device,
            dtype=parents_1.dtype,
        )
        betas = (2 * u).pow(1.0 / (eta + 1.0))
        upper_mask = u > 0.5
        betas[upper_mask] = (1.0 / (2 * (1.0 - u[upper_mask]))).pow(
            1.0 / (eta + 1.0)
        )
        children_1 = 0.5 * ((1 + betas) * parents_1 + (1 - betas) * parents_2)
        children_2 = 0.5 * ((1 + betas) * parents_2 + (1 - betas) * parents_1)
        return torch.cat([children_1, children_2], dim=0)

    def _mutate(self, children: torch.Tensor) -> torch.Tensor:
        if (
            children.numel() == 0
            or self.config.mutation_stdev <= 0.0
            or self.config.mutation_probability <= 0.0
        ):
            return children

        mutated = children.clone()
        mutation_mask = (
            torch.rand(
                mutated.shape,
                generator=self.generator,
                device=self.device,
            )
            <= self.config.mutation_probability
        )
        mutation_coordinates = mutation_mask.nonzero(as_tuple=True)
        mutation_count = mutation_coordinates[0].numel()
        if mutation_count == 0:
            return mutated

        mutation = torch.randn(
            (mutation_count,),
            generator=self.generator,
            device=self.device,
            dtype=mutated.dtype,
        ) * self.config.mutation_stdev
        mutated[mutation_coordinates] += mutation
        return mutated

    def _cosyne_permutation(
        self,
        population: torch.Tensor,
        losses: torch.Tensor,
    ) -> torch.Tensor:
        solution_length = population.shape[1]
        if solution_length == 0:
            return population.clone()

        if self.config.permute_all:
            prob_permute = torch.ones_like(population)
        else:
            ranks = centered_ranks(losses, higher_is_better=False)
            prob_permute = (
                1 - (ranks + 0.5).pow(1 / float(solution_length))
            ).unsqueeze(1).expand(population.shape[0], solution_length)

        perm_mask = (
            torch.rand(
                prob_permute.shape,
                generator=self.generator,
                device=self.device,
                dtype=prob_permute.dtype,
            )
            <= prob_permute
        )
        perm_mask_sorted = torch.sort(
            perm_mask.to(torch.long),
            descending=True,
            dim=0,
        )[0].to(torch.bool)

        perm_rand = torch.rand(
            prob_permute.shape,
            generator=self.generator,
            device=self.device,
            dtype=prob_permute.dtype,
        )
        perm_rand[torch.logical_not(perm_mask)] = 1.0
        permutations = torch.argsort(perm_rand, dim=0)

        perm_sort = (
            torch.arange(0, perm_mask.shape[0], device=self.device)
            .unsqueeze(-1)
            .repeat(1, perm_mask.shape[1])
        )
        perm_sort[torch.logical_not(perm_mask)] += perm_mask.shape[0] + 1
        perm_sort = torch.sort(perm_sort, dim=0)[0]

        _, permutation_columns = torch.nonzero(perm_mask_sorted, as_tuple=True)
        permutation_origin_indices = perm_sort[perm_mask_sorted]
        permutation_target_indices = permutations[perm_mask_sorted]

        new_population = population.clone()
        new_population[permutation_origin_indices, permutation_columns] = new_population[
            permutation_target_indices,
            permutation_columns,
        ]
        return new_population


def population_cross_entropy_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    expanded_targets = targets.expand(logits.shape[0], -1)
    per_sample_losses = F.cross_entropy(
        logits.transpose(1, 2),
        expanded_targets,
        reduction="none",
    )
    return per_sample_losses.mean(dim=1)


def centered_ranks(fitnesses: torch.Tensor, *, higher_is_better: bool = True) -> torch.Tensor:
    values = fitnesses.reshape(-1)
    count = len(values)
    if count <= 1:
        return torch.zeros_like(fitnesses)

    indices = values.argsort(descending=(not higher_is_better))
    weights = (
        torch.arange(count, dtype=values.dtype, device=values.device) / (count - 1)
    ) - 0.5
    ranks = torch.empty_like(values)
    ranks[indices] = weights
    return ranks.reshape(*(fitnesses.shape))


def split_reproduction_counts(population_size: int, crossover_fraction: float) -> tuple[int, int]:
    sexual_count = min(population_size, max(0, int(round(crossover_fraction * population_size))))
    return population_size - sexual_count, sexual_count


def reproduction_chunk_size(
    num_parameters: int,
    element_size: int,
    device: torch.device,
) -> int:
    target_bytes = (
        CUDA_REPRODUCTION_WORK_CHUNK_BYTES
        if device.type == "cuda"
        else CPU_REPRODUCTION_WORK_CHUNK_BYTES
    )
    bytes_per_individual = max(1, num_parameters * element_size)
    return max(1, target_bytes // bytes_per_individual)


def find_largest_safe_chunk_size(
    *,
    population_size: int,
    initial_chunk_size: int,
    can_evaluate: Callable[[int], bool],
) -> int:
    candidate = min(population_size, max(1, initial_chunk_size))
    if not can_evaluate(candidate):
        return binary_search_safe_chunk_size(1, candidate - 1, 1, can_evaluate)

    best = candidate
    high = min(population_size, max(candidate + 1, candidate * 2))
    failed_high: int | None = None

    while best < population_size:
        if can_evaluate(high):
            best = high
            if best == population_size:
                return best
            high = min(population_size, max(high + 1, high * 2))
            continue

        failed_high = high
        break

    if failed_high is None:
        return best

    return binary_search_safe_chunk_size(best + 1, failed_high - 1, best, can_evaluate)


def binary_search_safe_chunk_size(
    low: int,
    high: int,
    best: int,
    can_evaluate: Callable[[int], bool],
) -> int:
    while low <= high:
        midpoint = (low + high) // 2
        if can_evaluate(midpoint):
            best = midpoint
            low = midpoint + 1
        else:
            high = midpoint - 1
    return max(1, best)


def is_cuda_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cublas_status_alloc_failed" in message


def estimate_leea_chunk_size(
    *,
    device: torch.device,
    population_size: int,
    parameter_bytes: int,
    input_bytes: int,
) -> int:
    available_bytes = available_memory_bytes(device)
    if available_bytes is None:
        return min(population_size, fallback_leea_chunk_size(device))

    memory_fraction = 0.35 if device.type == "cuda" else 0.25
    budget_bytes = max(1, int(available_bytes * memory_fraction))
    # vmap batches model activations across population members. The multiplier
    # is intentionally conservative for convolutional models without tracing.
    estimated_bytes_per_individual = max(1, parameter_bytes + input_bytes * 32)
    memory_limited_chunk = max(1, budget_bytes // estimated_bytes_per_individual)
    return min(population_size, max_leea_chunk_size(device), memory_limited_chunk)


def available_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except (RuntimeError, TypeError):
            return None
        return int(free_bytes)

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None

    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None
    return pages * page_size


def fallback_leea_chunk_size(device: torch.device) -> int:
    if device.type == "cuda":
        return 16
    return 8


def max_leea_chunk_size(device: torch.device) -> int:
    if device.type == "cuda":
        return 64
    return 128


def initialize_population(
    model: nn.Module,
    population_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    original_state = clone_state_dict_on_device(model)
    population = []
    with torch.random.fork_rng(devices=fork_rng_devices(device)):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        for _ in range(population_size):
            reset_model_parameters(model)
            population.append(flatten_parameters(model).to(device))

    model.load_state_dict(original_state)
    return torch.stack(population).to(device)


def fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    if device.index is not None:
        return [device.index]
    return [torch.cuda.current_device()]


def reset_model_parameters(model: nn.Module) -> None:
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        reset_parameters()
        return

    for module in model.modules():
        if module is model:
            continue
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()


def clone_state_dict_on_device(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


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
    raw_params: dict[str, float | bool],
    *,
    device: torch.device,
    seed: int,
    leea_evaluation_chunk_size: int | None = None,
    leea_profile: bool = False,
) -> OptimizerRunner:
    if optimizer_name == "SGD":
        return SGDRunner(
            model,
            SGDConfig(eta=float(raw_params.get(ETA, 0.01))),
        )
    if optimizer_name == "LEEA":
        config = LEEAConfig(
            population_size=max(2, int(raw_params.get("N", 1000))),
            mutation_probability=clamp(float(raw_params.get(P_M, 0.04)), 0.0, 1.0),
            initial_mutation_step=max(0.0, float(raw_params.get(ETA_0, 0.03))),
            mutation_decay=max(0.0, float(raw_params.get(GAMMA, 0.99))),
            retention_fraction=clamp(float(raw_params.get(RHO, 0.4)), 0.01, 1.0),
            crossover_fraction=clamp(float(raw_params.get(RHO_X, 0.5)), 0.0, 1.0),
            fitness_decay=clamp(float(raw_params.get(LAMBDA, 0.2)), 0.0, 1.0),
            validation_patience=max(1, int(raw_params.get(TAU_PAT, 5))),
            evaluation_chunk_size=leea_evaluation_chunk_size
            if leea_evaluation_chunk_size is not None
            else leea_chunk_size_from_env(),
            profile=leea_profile,
        )
        return LEEARunner(model, config, device=device, seed=seed)
    if optimizer_name == "CoSyNE":
        eta_sbx = max(0.0, float(raw_params.get(ETA_SBX, 0)))
        num_children = max(0, int(raw_params.get(NUM_CHILDREN, 0)))
        config = CosyneConfig(
            population_size=max(4, int(raw_params.get("N", 1000))),
            tournament_size=max(1, int(raw_params.get(TOURNAMENT_SIZE, 4))),
            mutation_stdev=max(0.0, float(raw_params.get(SIGMA_M, 0.03))),
            mutation_probability=clamp(float(raw_params.get(COSYNE_P_M, 1.0)), 0.0, 1.0),
            permute_all=bool_param(raw_params.get(PERMUTE_ALL, False)),
            elitism_ratio=clamp(float(raw_params.get(RHO_E, 0.01)), 0.0, 1.0),
            sbx_eta=None if eta_sbx == 0.0 else eta_sbx,
            num_children=None if num_children == 0 else num_children,
            evaluation_chunk_size=leea_evaluation_chunk_size
            if leea_evaluation_chunk_size is not None
            else leea_chunk_size_from_env(),
        )
        return CosyneRunner(model, config, device=device, seed=seed)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def leea_chunk_size_from_env() -> int | None:
    value = os.environ.get("KISEKI_LEEA_CHUNK_SIZE")
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bool_param(value: float | bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
