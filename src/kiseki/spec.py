"""The run contract: what a training run is asked to do, frozen into spec.json at creation.

`RunSpec` is the only input a worker gets. `normalize` fills defaults (the dataset's model,
every optimizer parameter) so a stored spec is complete and two specs compare by value.
"""

import secrets
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import msgspec
from msgspec import Meta

from kiseki import datasets, models, optimizers

Precision = Literal["fp32", "bf16"]


class RunSpec(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    dataset: str = "mnist"
    model: Annotated[str, Meta(description="Empty means the dataset's default model")] = ""
    optimizer: str = "leea"
    params: dict[str, Any] = {}
    seed: int = 42
    batch_size: Annotated[int, Meta(ge=1)] = 512
    iterations: Annotated[
        int, Meta(ge=1, description="Steps to train (generations, for evolution)")
    ] = 100_000
    target_accuracy: (
        Annotated[float, Meta(gt=0, le=1, description="Stop once validation accuracy reaches it")]
        | None
    ) = None
    validation_interval: Annotated[int, Meta(ge=1)] = 10
    validation_size: (
        Annotated[int, Meta(ge=1, description="Null uses the dataset's default split")] | None
    ) = None
    split_seed: Annotated[
        int, Meta(description="Seed of the train/validation split, shared across runs")
    ] = 0
    checkpoint_interval: Annotated[
        int, Meta(ge=1, description="Steps between resumable checkpoints")
    ] = 1000
    precision: Annotated[
        Precision, Meta(description="Arithmetic for population scoring; weights stay fp32")
    ] = "fp32"
    deterministic: Annotated[
        bool, Meta(description="Bit-reproducible GPU kernels, at some speed cost")
    ] = False
    chunk_size: (
        Annotated[int, Meta(ge=1, description="Networks scored at once; null picks one")] | None
    ) = None
    start_from: Annotated[
        str,
        Meta(
            description="Empty starts from a fresh initialisation; else a saved network "
            "RUN:best|final|step-N of a run with this dataset, model and split"
        ),
    ] = ""
    label: str = ""


def normalize(spec: RunSpec) -> RunSpec:
    info = datasets.info(spec.dataset)
    model = spec.model or info.default_model
    if model not in models.NAMES:
        raise ValueError(f"unknown model {model!r}; known: {', '.join(models.NAMES)}")
    params = optimizers.normalize_params(spec.optimizer, spec.params)
    validation_size = spec.validation_size or info.validation_size
    if info.train_size - validation_size < spec.batch_size:
        raise ValueError("validation_size leaves fewer training images than one batch")
    return msgspec.structs.replace(
        spec, model=model, params=params, validation_size=validation_size
    )


def from_overrides(base: RunSpec, overrides: dict[str, Any]) -> RunSpec:
    """Apply flat overrides: RunSpec fields by name, optimizer params by name or params.<x>."""
    fields = set(RunSpec.__struct_fields__)
    top: dict[str, Any] = {}
    params = dict(base.params)
    for key, value in overrides.items():
        name = key.removeprefix("params.")
        if key in fields and key != "params":
            top[key] = value
        elif (
            name
            in optimizers.params_type(overrides.get("optimizer", base.optimizer)).__struct_fields__
        ):
            params[name] = value
        else:
            raise KeyError(f"unknown setting {key!r}")
    if "optimizer" in top and top["optimizer"] != base.optimizer:
        known = optimizers.params_type(top["optimizer"]).__struct_fields__
        params = {k: v for k, v in params.items() if k in known}
    merged = msgspec.to_builtins(base) | top | {"params": params}
    return normalize(msgspec.convert(merged, RunSpec, strict=False))


def new_run_id(spec: RunSpec, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{spec.dataset}-{spec.optimizer}-{secrets.token_hex(2)}"


class ModelRef(msgspec.Struct, frozen=True):
    """A trained network inside a run: `<run_id>:<name>` with name best, final or step-<n>."""

    run_id: str
    name: str = "best"

    @classmethod
    def parse(cls, text: str) -> ModelRef:
        run_id, _, name = text.partition(":")
        return cls(run_id=run_id, name=name or "best")

    def __str__(self) -> str:
        return f"{self.run_id}:{self.name}"


def encode(value: Any) -> bytes:
    return msgspec.json.format(msgspec.json.encode(value), indent=2) + b"\n"
