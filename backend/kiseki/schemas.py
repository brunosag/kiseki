from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    type: Literal["select", "number", "boolean"]
    label: str
    options: list[SelectOption] | None = None
    step: float | None = None
    default: str | bool | float | int | None = None


class OptimizerParamField(BaseModel):
    key: str
    label: str
    type: Literal["number"] = "number"
    default: float
    step: float
    desc: str


class ExperimentConfig(BaseModel):
    dataset: Literal["mnist"] = "mnist"
    device: Literal["cpu", "gpu"] = "gpu"
    seed: int = 42
    batch_size: int = 512
    iterations: int = 100000
    target_acc: float = 100.0
    optimizer: Literal["LEEA", "SGD"] = "LEEA"
    deterministic: bool = False
    checkpoint_interval: int = 50

    @field_validator("batch_size", "iterations")
    @classmethod
    def require_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("checkpoint_interval")
    @classmethod
    def require_nonnegative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value

    @field_validator("target_acc")
    @classmethod
    def require_accuracy_range(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("must be between 0 and 100")
        return value


CheckpointKind = Literal["latest", "best"]
CheckpointListMode = Literal["training", "analysis"]


class CheckpointSelection(BaseModel):
    run_id: str
    kind: CheckpointKind


class TSNEParams(BaseModel):
    perplexity: float = 30.0
    max_iter: int = 1000
    learning_rate_mode: Literal["auto", "numeric"] = "auto"
    learning_rate: float | None = None
    angle: float = 0.5
    pca_components: int = 50
    seed: int | None = None
    use_pca: bool = True

    @field_validator("perplexity")
    @classmethod
    def require_perplexity_range(cls, value: float) -> float:
        if not 5 <= value <= 50:
            raise ValueError("must be between 5 and 50")
        return value

    @field_validator("max_iter")
    @classmethod
    def require_minimum_iterations(cls, value: int) -> int:
        if value < 250:
            raise ValueError("must be at least 250")
        return value

    @field_validator("angle")
    @classmethod
    def require_angle_range(cls, value: float) -> float:
        if not 0.2 <= value <= 0.8:
            raise ValueError("must be between 0.2 and 0.8")
        return value

    @field_validator("pca_components")
    @classmethod
    def require_pca_component_range(cls, value: int) -> int:
        if not 2 <= value <= 120:
            raise ValueError("must be between 2 and 120")
        return value

    @model_validator(mode="after")
    def require_positive_numeric_learning_rate(self) -> "TSNEParams":
        if self.learning_rate_mode == "auto":
            return self
        if self.learning_rate is None or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive when learning_rate_mode is numeric")
        return self


class TSNEAnalysisRequest(BaseModel):
    checkpoint: CheckpointSelection
    params: TSNEParams = Field(default_factory=TSNEParams)


class TSNEPoint(BaseModel):
    x: float
    y: float
    label: int
    prediction: int
    correct: bool


class StartExperimentRequest(BaseModel):
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)
    opt_params: dict[str, dict[str, float]] = Field(default_factory=dict)
    checkpoint: CheckpointSelection | None = None


class ExperimentControlsUpdate(BaseModel):
    iterations: int | None = None
    target_acc: float | None = None
    checkpoint_interval: int | None = None

    @field_validator("iterations")
    @classmethod
    def require_positive_iterations(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("checkpoint_interval")
    @classmethod
    def require_nonnegative_checkpoint_interval(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("must be non-negative")
        return value

    @field_validator("target_acc")
    @classmethod
    def require_target_accuracy_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 100.0:
            raise ValueError("must be between 0 and 100")
        return value


class CheckpointSummary(BaseModel):
    run_id: str
    kind: CheckpointKind
    saved_at: str
    step: int
    optimizer: Literal["LEEA", "SGD"]
    dataset: Literal["mnist"]
    seed: int
    requested_device: str | None = None
    device: str
    device_name: str | None = None
    deterministic: bool
    accuracy: float | None = None
    best_acc: float | None = None
    current_loss: float | None = None
    total_elapsed_seconds: float | None = None
    reproducibility_mode: str = "best_effort"
    reproducibility_status: str
    compatibility_warnings: list[str] = Field(default_factory=list)
    config: ExperimentConfig
    optimizer_params: dict[str, dict[str, float]] = Field(default_factory=dict)


class TSNEAnalysisResponse(BaseModel):
    checkpoint: CheckpointSummary
    params: TSNEParams
    points: list[TSNEPoint]


class AccuracyPoint(BaseModel):
    i: int
    value: float


class MutationStepPoint(BaseModel):
    i: int
    value: float


class TrainingHistory(BaseModel):
    loss: list[float] = Field(default_factory=list)
    acc: list[AccuracyPoint] = Field(default_factory=list)
    mutation_step: list[MutationStepPoint] = Field(default_factory=list)


class ExperimentStatus(BaseModel):
    is_running: bool = False
    is_paused: bool = False
    pause_requested: bool = False
    optimizer: Literal["LEEA", "SGD"] | None = None
    run_id: str | None = None
    current_step: int = 0
    current_loss: float = 0.0
    current_mutation_step: float | None = None
    best_acc: float = 0.0
    total_elapsed_seconds: float = 0.0
    last_iteration_seconds: float = 0.0
    requested_device: str = "cpu"
    device: str = "cpu"
    device_name: str = "cpu"
    history: TrainingHistory = Field(default_factory=TrainingHistory)
    error: str | None = None
    last_checkpoint_step: int | None = None
    last_checkpoint_acc: float | None = None
    last_checkpoint_saved_at: str | None = None
    checkpoint_path: str | None = None
    best_checkpoint_acc: float | None = None
    best_checkpoint_step: int | None = None
    best_checkpoint_saved_at: str | None = None
    best_checkpoint_path: str | None = None
    reproducibility_mode: str = "best_effort"
    checkpoint_warnings: list[str] = Field(default_factory=list)


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
        default="gpu",
        options=[SelectOption(label="CPU", value="cpu"), SelectOption(label="GPU", value="gpu")],
    ),
    "seed": ConfigField(type="number", label="Seed", step=1, default=42),
    "batch_size": ConfigField(type="number", label="Batch size", step=1, default=512),
    "iterations": ConfigField(type="number", label="Iterations", step=1, default=100000),
    "target_acc": ConfigField(type="number", label="Target accuracy", step=0.01, default=100.0),
    "deterministic": ConfigField(type="boolean", label="Deterministic", default=False),
    "checkpoint_interval": ConfigField(
        type="number",
        label="Checkpoint interval",
        step=1,
        default=50,
    ),
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
        OptimizerParamField(key="N", label="N", default=1000, step=1, desc="Population size"),
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
            default=5,
            step=1,
            desc="Validation patience threshold",
        ),
    ],
}


def schema_response() -> SchemaResponse:
    return SchemaResponse(config_schema=CONFIG_SCHEMA, optimizers_schema=OPTIMIZERS_SCHEMA)
