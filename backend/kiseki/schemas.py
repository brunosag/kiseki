from typing import Literal

from pydantic import BaseModel, Field, field_validator


ETA = "\u03b7"
P_M = "p\u2098"
ETA_0 = "\u03b7\u2080"
GAMMA = "\u03b3"
RHO = "\u03c1"
RHO_X = "\u03c1\u2093"
LAMBDA = "\u03bb"
TAU_PAT = "\u03c4_pat"


class SelectOption(BaseModel):
    label: str
    value: str


class ConfigField(BaseModel):
    type: Literal["select", "number"]
    label: str
    options: list[SelectOption] | None = None
    step: float | None = None
    default: str | float | int | None = None


class OptimizerParamField(BaseModel):
    key: str
    label: str
    type: Literal["number"] = "number"
    default: float
    step: float
    desc: str


class ExperimentConfig(BaseModel):
    dataset: Literal["mnist"] = "mnist"
    device: Literal["cpu", "gpu"] = "cpu"
    seed: int = 42
    batch_size: int = 1000
    iterations: int = 100000
    target_acc: float = 100.0
    optimizer: Literal["LEEA", "SGD"] = "LEEA"

    @field_validator("batch_size", "iterations")
    @classmethod
    def require_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("target_acc")
    @classmethod
    def require_accuracy_range(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("must be between 0 and 100")
        return value


class StartExperimentRequest(BaseModel):
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)
    opt_params: dict[str, dict[str, float]] = Field(default_factory=dict)


class AccuracyPoint(BaseModel):
    i: int
    value: float


class TrainingHistory(BaseModel):
    loss: list[float] = Field(default_factory=list)
    acc: list[AccuracyPoint] = Field(default_factory=list)


class ExperimentStatus(BaseModel):
    is_running: bool = False
    current_step: int = 0
    current_loss: float = 0.0
    best_acc: float = 0.0
    requested_device: str = "cpu"
    device: str = "cpu"
    device_name: str = "cpu"
    history: TrainingHistory = Field(default_factory=TrainingHistory)
    error: str | None = None


class SchemaResponse(BaseModel):
    config_schema: dict[str, ConfigField]
    optimizers_schema: dict[str, list[OptimizerParamField]]


CONFIG_SCHEMA: dict[str, ConfigField] = {
    "dataset": ConfigField(
        type="select",
        label="Dataset",
        options=[SelectOption(label="MNIST", value="mnist")],
    ),
    "device": ConfigField(
        type="select",
        label="Device",
        options=[SelectOption(label="CPU", value="cpu"), SelectOption(label="GPU", value="gpu")],
    ),
    "seed": ConfigField(type="number", label="Seed", step=1, default=42),
    "batch_size": ConfigField(type="number", label="Batch size", step=1, default=1000),
    "iterations": ConfigField(type="number", label="Iterations", step=1, default=100000),
    "target_acc": ConfigField(type="number", label="Target accuracy", step=0.01, default=100.0),
    "optimizer": ConfigField(
        type="select",
        label="Optimizer",
        options=[SelectOption(label="LEEA", value="LEEA"), SelectOption(label="SGD", value="SGD")],
    ),
}

OPTIMIZERS_SCHEMA: dict[str, list[OptimizerParamField]] = {
    "SGD": [
        OptimizerParamField(
            key=ETA,
            label=r"\eta",
            default=0.01,
            step=0.01,
            desc="Learning rate",
        ),
    ],
    "LEEA": [
        OptimizerParamField(key="N", label="N", default=200, step=1, desc="Population size"),
        OptimizerParamField(
            key=P_M,
            label=r"p_{\mathrm{m}}",
            default=0.04,
            step=0.01,
            desc="Mutation probability",
        ),
        OptimizerParamField(
            key=ETA_0,
            label=r"\eta_0",
            default=0.03,
            step=0.01,
            desc="Initial mutation step size",
        ),
        OptimizerParamField(
            key=GAMMA,
            label=r"\gamma",
            default=0.99,
            step=0.01,
            desc="Mutation decay factor",
        ),
        OptimizerParamField(
            key=RHO,
            label=r"\rho",
            default=0.4,
            step=0.01,
            desc="Retention fraction",
        ),
        OptimizerParamField(
            key=RHO_X,
            label=r"\rho_{\mathrm{x}}",
            default=0.5,
            step=0.01,
            desc="Crossover fraction",
        ),
        OptimizerParamField(
            key=LAMBDA,
            label=r"\lambda",
            default=0.2,
            step=0.01,
            desc="Fitness decay coefficient",
        ),
        OptimizerParamField(
            key=TAU_PAT,
            label=r"\tau_{\mathrm{pat}}",
            default=25,
            step=1,
            desc="Validation patience threshold",
        ),
    ],
}


def schema_response() -> SchemaResponse:
    return SchemaResponse(config_schema=CONFIG_SCHEMA, optimizers_schema=OPTIMIZERS_SCHEMA)
