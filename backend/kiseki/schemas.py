from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .dataset_types import DatasetName


ETA = "\u03b7"
P_M = "p\u2098"
ETA_0 = "\u03b7\u2080"
GAMMA = "\u03b3"
RHO = "\u03c1"
RHO_X = "\u03c1\u2093"
LAMBDA = "\u03bb"
TAU_PAT = "\u03c4_pat"
TOURNAMENT_SIZE = "k"
SIGMA_M = "sigma_m"
COSYNE_P_M = "p_m"
PERMUTE_ALL = "permute_all"
RHO_E = "rho_e"
ETA_SBX = "eta_sbx"
NUM_CHILDREN = "num_children"
OptimizerName = Literal["LEEA", "SGD", "CoSyNE"]


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
    type: Literal["number", "boolean"] = "number"
    default: float | bool
    step: float | None = None
    desc: str


class ExperimentConfig(BaseModel):
    dataset: DatasetName = "mnist"
    device: Literal["cpu", "gpu"] = "gpu"
    seed: int = 42
    batch_size: int = 512
    iterations: int = 100000
    target_acc: float = 100.0
    optimizer: OptimizerName = "LEEA"
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


class AnalysisComparisonParams(BaseModel):
    tsne_perplexity: float = 30.0
    tsne_max_iter: int = 1000
    tsne_learning_rate_mode: Literal["auto", "numeric"] = "auto"
    tsne_learning_rate: float | None = None
    tsne_angle: float = 0.5
    tsne_pca_components: int = 50
    tsne_seed: int | None = None
    calibration_bins: int = 15
    lrp_gallery_sample_count: int = 24
    robustness_noise_levels: list[float] = Field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])
    robustness_brightness_levels: list[float] = Field(default_factory=lambda: [0.0, 0.1, 0.2])
    robustness_cutout_levels: list[float] = Field(default_factory=lambda: [0.0, 0.125, 0.25])

    @field_validator("tsne_perplexity")
    @classmethod
    def require_tsne_perplexity_range(cls, value: float) -> float:
        if not 5 <= value <= 50:
            raise ValueError("must be between 5 and 50")
        return value

    @field_validator("tsne_max_iter")
    @classmethod
    def require_minimum_tsne_iterations(cls, value: int) -> int:
        if value < 250:
            raise ValueError("must be at least 250")
        return value

    @field_validator("tsne_angle")
    @classmethod
    def require_tsne_angle_range(cls, value: float) -> float:
        if not 0.2 <= value <= 0.8:
            raise ValueError("must be between 0.2 and 0.8")
        return value

    @field_validator("tsne_pca_components")
    @classmethod
    def require_tsne_pca_component_range(cls, value: int) -> int:
        if not 2 <= value <= 120:
            raise ValueError("must be between 2 and 120")
        return value

    @field_validator("calibration_bins")
    @classmethod
    def require_calibration_bin_range(cls, value: int) -> int:
        if not 5 <= value <= 50:
            raise ValueError("must be between 5 and 50")
        return value

    @field_validator("lrp_gallery_sample_count")
    @classmethod
    def require_lrp_gallery_sample_count_range(cls, value: int) -> int:
        if not 1 <= value <= 60:
            raise ValueError("must be between 1 and 60")
        return value

    @field_validator(
        "robustness_noise_levels",
        "robustness_brightness_levels",
        "robustness_cutout_levels",
    )
    @classmethod
    def require_nonnegative_robustness_levels(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("must include at least one level")
        for level in value:
            if level < 0:
                raise ValueError("levels must be non-negative")
        return value

    @model_validator(mode="after")
    def require_positive_numeric_tsne_learning_rate(self) -> "AnalysisComparisonParams":
        if self.tsne_learning_rate_mode == "auto":
            return self
        if self.tsne_learning_rate is None or self.tsne_learning_rate <= 0:
            raise ValueError(
                "tsne_learning_rate must be positive when tsne_learning_rate_mode is numeric"
            )
        return self


class AnalysisComparisonJobRequest(BaseModel):
    left: CheckpointSelection
    right: CheckpointSelection
    params: AnalysisComparisonParams = Field(default_factory=AnalysisComparisonParams)
    force_recompute: bool = False


class AnalysisProgressEvent(BaseModel):
    stage: str
    message: str
    progress: float


class AnalysisTableRow(BaseModel):
    label: str
    left: str
    right: str
    comparable: bool | None = None


class AnalysisCurvePoint(BaseModel):
    i: int
    value: float


class AnalysisCurves(BaseModel):
    training_loss: list[float] = Field(default_factory=list)
    validation_accuracy: list[AnalysisCurvePoint] = Field(default_factory=list)
    training_accuracy: list[AnalysisCurvePoint] = Field(default_factory=list)
    validation_loss: list[AnalysisCurvePoint] = Field(default_factory=list)


class AnalysisPerClassMetric(BaseModel):
    label: int
    name: str
    precision: float
    recall: float
    f1: float
    support: int


class AnalysisCalibrationBin(BaseModel):
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float


class AnalysisCalibration(BaseModel):
    bins: list[AnalysisCalibrationBin]
    ece: float
    brier_score: float
    nll: float


class AnalysisModelMetrics(BaseModel):
    accuracy: float
    loss: float
    macro_f1: float
    per_class_f1: list[AnalysisPerClassMetric]
    confusion_matrix: list[list[int]]
    calibration: AnalysisCalibration


class AnalysisOverlap(BaseModel):
    total: int
    correct_both: int
    left_only_correct: int
    right_only_correct: int
    error_both: int
    disagreements: int
    both_error_same_prediction: int
    both_error_different_prediction: int
    upset: list[dict[str, Any]]


class AnalysisEmbeddingPoint(BaseModel):
    index: int
    x: float
    y: float
    label: int
    prediction: int
    correct: bool


class AnalysisEmbeddingProjection(BaseModel):
    left: list[AnalysisEmbeddingPoint]
    right: list[AnalysisEmbeddingPoint]


class AnalysisEmbeddings(BaseModel):
    pca: AnalysisEmbeddingProjection
    tsne: AnalysisEmbeddingProjection


class AnalysisLrpSample(BaseModel):
    index: int
    label: int
    label_name: str
    group: str
    left_prediction: int
    right_prediction: int
    left_confidence: float
    right_confidence: float
    image: list[list[list[float]]]
    left_relevance: list[list[float]]
    right_relevance: list[list[float]]
    difference_relevance: list[list[float]]


class AnalysisClassAverageRelevance(BaseModel):
    label: int
    name: str
    left_relevance: list[list[float]]
    right_relevance: list[list[float]]
    difference_relevance: list[list[float]]


class AnalysisLrpReport(BaseModel):
    samples: list[AnalysisLrpSample]
    class_averages: list[AnalysisClassAverageRelevance]


class AnalysisHistogram(BaseModel):
    bins: list[float]
    counts: list[int]


class AnalysisActivationLayerStats(BaseModel):
    name: str
    sparsity: float
    mean: float
    std: float
    q05: float
    q50: float
    q95: float
    histogram: AnalysisHistogram


class AnalysisActivationReport(BaseModel):
    left: list[AnalysisActivationLayerStats]
    right: list[AnalysisActivationLayerStats]


class AnalysisWeightLayerComparison(BaseModel):
    name: str
    left_norm: float
    right_norm: float
    distance: float
    relative_distance: float


class AnalysisRobustnessPoint(BaseModel):
    level: float
    left_accuracy: float
    right_accuracy: float
    left_loss: float
    right_loss: float


class AnalysisRobustnessCurve(BaseModel):
    perturbation: str
    points: list[AnalysisRobustnessPoint]


class AnalysisRuntimeReport(BaseModel):
    rows: list[AnalysisTableRow]


class AnalysisComparisonReport(BaseModel):
    analysis_version: str
    generated_at: str
    analysis_device: str
    left: CheckpointSummary
    right: CheckpointSummary
    params: AnalysisComparisonParams
    metadata: list[AnalysisTableRow]
    comparability: list[AnalysisTableRow]
    curves: dict[str, AnalysisCurves]
    metrics: dict[str, AnalysisModelMetrics]
    confusion_difference: list[list[int]]
    overlap: AnalysisOverlap
    embeddings: AnalysisEmbeddings
    lrp: AnalysisLrpReport
    activations: AnalysisActivationReport
    weights: list[AnalysisWeightLayerComparison]
    robustness: list[AnalysisRobustnessCurve]
    runtime: AnalysisRuntimeReport


AnalysisJobStatus = Literal["queued", "running", "completed", "failed"]
AnalysisCacheState = Literal["miss", "fresh", "stale", "recomputed"]


class AnalysisComparisonJobStatus(BaseModel):
    job_id: str
    status: AnalysisJobStatus
    progress: float
    stage: str
    message: str
    cache_state: AnalysisCacheState
    stale_sides: list[Literal["left", "right"]] = Field(default_factory=list)
    report: AnalysisComparisonReport | None = None
    error: str | None = None


class StartExperimentRequest(BaseModel):
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)
    opt_params: dict[str, dict[str, float | bool]] = Field(default_factory=dict)
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
    optimizer: OptimizerName
    dataset: DatasetName
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
    optimizer_params: dict[str, dict[str, float | bool]] = Field(default_factory=dict)


class AccuracyPoint(BaseModel):
    i: int
    value: float


class MutationStepPoint(BaseModel):
    i: int
    value: float


class TrainingHistory(BaseModel):
    loss: list[float] = Field(default_factory=list)
    acc: list[AccuracyPoint] = Field(default_factory=list)
    train_acc: list[AccuracyPoint] = Field(default_factory=list)
    val_loss: list[AccuracyPoint] = Field(default_factory=list)
    memory_mb: list[AccuracyPoint] = Field(default_factory=list)
    mutation_step: list[MutationStepPoint] = Field(default_factory=list)


class ExperimentStatus(BaseModel):
    is_running: bool = False
    is_paused: bool = False
    pause_requested: bool = False
    optimizer: OptimizerName | None = None
    run_id: str | None = None
    current_step: int = 0
    current_loss: float = 0.0
    current_mutation_step: float | None = None
    best_acc: float = 0.0
    total_elapsed_seconds: float = 0.0
    last_iteration_seconds: float = 0.0
    loss_mean_since_validation: float = 0.0
    loss_stdev_since_validation: float = 0.0
    mean_iteration_seconds_since_validation: float = 0.0
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
        default="mnist",
        options=[
            SelectOption(label="MNIST", value="mnist"),
            SelectOption(label="CIFAR-10", value="cifar10"),
        ],
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
        options=[
            SelectOption(label="LEEA", value="LEEA"),
            SelectOption(label="SGD", value="SGD"),
            SelectOption(label="CoSyNE", value="CoSyNE"),
        ],
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
    "CoSyNE": [
        OptimizerParamField(key="N", label="N", default=1000, step=1, desc="Population size"),
        OptimizerParamField(
            key=TOURNAMENT_SIZE,
            label="k",
            default=4,
            step=1,
            desc="Tournament size",
        ),
        OptimizerParamField(
            key=SIGMA_M,
            label=r"\sigma_{\mathrm{m}}",
            default=0.03,
            step=0.01,
            desc="Mutation standard deviation",
        ),
        OptimizerParamField(
            key=COSYNE_P_M,
            label=r"p_{\mathrm{m}}",
            default=1.0,
            step=0.01,
            desc="Mutation probability",
        ),
        OptimizerParamField(
            key=RHO_E,
            label=r"\rho_{\mathrm{e}}",
            default=0.01,
            step=0.01,
            desc="Elitism fraction",
        ),
        OptimizerParamField(
            key=ETA_SBX,
            label=r"\eta_{\mathrm{SBX}}",
            default=0,
            step=1,
            desc="SBX distribution index",
        ),
        OptimizerParamField(
            key=NUM_CHILDREN,
            label=r"\lambda_{\mathrm{c}}",
            default=0,
            step=1,
            desc="Children count",
        ),
        OptimizerParamField(
            key=PERMUTE_ALL,
            label=r"\pi_{\mathrm{all}}",
            type="boolean",
            default=False,
            desc="Permute all columns",
        ),
    ],
}


def schema_response() -> SchemaResponse:
    return SchemaResponse(config_schema=CONFIG_SCHEMA, optimizers_schema=OPTIMIZERS_SCHEMA)
