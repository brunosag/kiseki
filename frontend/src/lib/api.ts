export type SelectOption = {
  label: string
  value: string
}

export type ConfigField = {
  type: "select" | "number" | "boolean"
  label: string
  options?: SelectOption[] | null
  step?: number | null
  default?: string | number | boolean | null
}

export type OptimizerParamField = {
  key: string
  label: string
  type: "number" | "boolean"
  default: number | boolean
  step?: number | null
  desc: string
}

export type SchemaResponse = {
  config_schema: Record<string, ConfigField>
  optimizers_schema: Record<string, OptimizerParamField[]>
}

export type DatasetName = "mnist" | "fashion_mnist" | "cifar10"
export type OptimizerName = "LEEA" | "SGD" | "CoSyNE"

export type ExperimentConfig = {
  dataset: DatasetName
  device: "cpu" | "gpu"
  seed: number
  batch_size: number
  iterations: number
  target_acc: number
  deterministic: boolean
  checkpoint_interval: number
  optimizer: OptimizerName
}

export type CheckpointKind = "latest" | "best"
export type CheckpointListMode = "training" | "analysis"

export type CheckpointSelection = {
  run_id: string
  kind: CheckpointKind
}

export type CheckpointSummary = CheckpointSelection & {
  saved_at: string
  step: number
  optimizer: OptimizerName
  dataset: DatasetName
  seed: number
  requested_device?: string | null
  device: string
  device_name?: string | null
  deterministic: boolean
  accuracy?: number | null
  best_acc?: number | null
  current_loss?: number | null
  total_elapsed_seconds?: number | null
  reproducibility_mode: string
  reproducibility_status: string
  compatibility_warnings: string[]
  config: ExperimentConfig
  optimizer_params: OptimizerParams
}

export type AccuracyPoint = {
  i: number
  value: number
}

export type MutationStepPoint = {
  i: number
  value: number
}

export type TrainingHistory = {
  loss: number[]
  acc: AccuracyPoint[]
  train_acc: AccuracyPoint[]
  val_loss: AccuracyPoint[]
  memory_mb: AccuracyPoint[]
  mutation_step: MutationStepPoint[]
}

export type ExperimentStatus = {
  is_running: boolean
  is_paused: boolean
  pause_requested: boolean
  optimizer?: OptimizerName | null
  run_id?: string | null
  current_step: number
  current_loss: number
  current_mutation_step?: number | null
  best_acc: number
  total_elapsed_seconds: number
  last_iteration_seconds: number
  loss_mean_since_validation: number
  loss_stdev_since_validation: number
  mean_iteration_seconds_since_validation: number
  requested_device: string
  device: string
  device_name: string
  history: TrainingHistory
  error?: string | null
  last_checkpoint_step?: number | null
  last_checkpoint_acc?: number | null
  last_checkpoint_saved_at?: string | null
  checkpoint_path?: string | null
  best_checkpoint_acc?: number | null
  best_checkpoint_step?: number | null
  best_checkpoint_saved_at?: string | null
  best_checkpoint_path?: string | null
  reproducibility_mode: string
  checkpoint_warnings: string[]
}

export type OptimizerParamValue = number | boolean
export type OptimizerParams = Record<string, Record<string, OptimizerParamValue>>

export type TsneLearningRateMode = "auto" | "numeric"

export type AnalysisComparisonParams = {
  tsne_perplexity: number
  tsne_max_iter: number
  tsne_learning_rate_mode: TsneLearningRateMode
  tsne_learning_rate?: number | null
  tsne_angle: number
  tsne_pca_components: number
  tsne_seed?: number | null
  calibration_bins: number
  lrp_gallery_sample_count: number
  robustness_noise_levels: number[]
  robustness_brightness_levels: number[]
  robustness_cutout_levels: number[]
}

export type AnalysisTableRow = {
  label: string
  left: string
  right: string
  comparable?: boolean | null
}

export type AnalysisCurves = {
  training_loss: number[]
  validation_accuracy: AccuracyPoint[]
  training_accuracy: AccuracyPoint[]
  validation_loss: AccuracyPoint[]
}

export type AnalysisPerClassMetric = {
  label: number
  name: string
  precision: number
  recall: number
  f1: number
  support: number
}

export type AnalysisCalibrationBin = {
  lower: number
  upper: number
  count: number
  confidence: number
  accuracy: number
}

export type AnalysisCalibration = {
  bins: AnalysisCalibrationBin[]
  ece: number
  brier_score: number
  nll: number
}

export type AnalysisModelMetrics = {
  accuracy: number
  loss: number
  macro_f1: number
  per_class_f1: AnalysisPerClassMetric[]
  confusion_matrix: number[][]
  calibration: AnalysisCalibration
}

export type AnalysisOverlap = {
  total: number
  correct_both: number
  left_only_correct: number
  right_only_correct: number
  error_both: number
  disagreements: number
  both_error_same_prediction: number
  both_error_different_prediction: number
  upset: { set: string; count: number }[]
}

export type AnalysisEmbeddingPoint = {
  index: number
  x: number
  y: number
  label: number
  prediction: number
  correct: boolean
}

export type AnalysisEmbeddingProjection = {
  left: AnalysisEmbeddingPoint[]
  right: AnalysisEmbeddingPoint[]
  left_total: number
  right_total: number
  x_domain: [number, number]
  y_domain: [number, number]
}

export type AnalysisEmbeddings = {
  pca: AnalysisEmbeddingProjection
  tsne: AnalysisEmbeddingProjection
}

export type AnalysisLrpSample = {
  index: number
  label: number
  label_name: string
  group: string
  left_prediction: number
  right_prediction: number
  left_confidence: number
  right_confidence: number
  image: number[][][]
  left_relevance: number[][]
  right_relevance: number[][]
  difference_relevance: number[][]
}

export type AnalysisClassAverageRelevance = {
  label: number
  name: string
  left_relevance: number[][]
  right_relevance: number[][]
  difference_relevance: number[][]
}

export type AnalysisLrpReport = {
  samples: AnalysisLrpSample[]
  class_averages: AnalysisClassAverageRelevance[]
}

export type AnalysisHistogram = {
  bins: number[]
  counts: number[]
}

export type AnalysisActivationLayerStats = {
  name: string
  sparsity: number
  mean: number
  std: number
  q05: number
  q50: number
  q95: number
  histogram: AnalysisHistogram
}

export type AnalysisActivationReport = {
  left: AnalysisActivationLayerStats[]
  right: AnalysisActivationLayerStats[]
}

export type AnalysisWeightLayerComparison = {
  name: string
  left_norm: number
  right_norm: number
  distance: number
  relative_distance: number
}

export type AnalysisRobustnessPoint = {
  level: number
  left_accuracy: number
  right_accuracy: number
  left_loss: number
  right_loss: number
}

export type AnalysisRobustnessCurve = {
  perturbation: string
  points: AnalysisRobustnessPoint[]
}

export type AnalysisRuntimeReport = {
  rows: AnalysisTableRow[]
}

export type AnalysisComparisonReport = {
  analysis_version: string
  generated_at: string
  analysis_device: string
  left: CheckpointSummary
  right: CheckpointSummary
  params: AnalysisComparisonParams
  metadata: AnalysisTableRow[]
  comparability: AnalysisTableRow[]
  curves: Record<"left" | "right", AnalysisCurves>
  metrics: Record<"left" | "right", AnalysisModelMetrics>
  confusion_difference: number[][]
  overlap: AnalysisOverlap
  embeddings: AnalysisEmbeddings
  lrp: AnalysisLrpReport
  activations: AnalysisActivationReport
  weights: AnalysisWeightLayerComparison[]
  robustness: AnalysisRobustnessCurve[]
  runtime: AnalysisRuntimeReport
}

export type AnalysisJobStatus = "queued" | "running" | "completed" | "failed"
export type AnalysisCacheState = "miss" | "fresh" | "stale" | "recomputed"

export type AnalysisComparisonJobStatus = {
  job_id: string
  status: AnalysisJobStatus
  progress: number
  stage: string
  message: string
  cache_state: AnalysisCacheState
  stale_sides: ("left" | "right")[]
  report_available: boolean
  error?: string | null
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ""

export const fallbackSchema: SchemaResponse = {
  config_schema: {
    dataset: {
      type: "select",
      label: "Dataset",
      default: "mnist",
      options: [
        { label: "MNIST", value: "mnist" },
        { label: "Fashion MNIST", value: "fashion_mnist" },
        { label: "CIFAR-10", value: "cifar10" },
      ],
    },
    device: {
      type: "select",
      label: "Device",
      default: "gpu",
      options: [
        { label: "CPU", value: "cpu" },
        { label: "GPU", value: "gpu" },
      ],
    },
    seed: { type: "number", step: 1, default: 42, label: "Seed" },
    batch_size: {
      type: "number",
      step: 1,
      default: 512,
      label: "Batch size",
    },
    iterations: {
      type: "number",
      step: 1,
      default: 100000,
      label: "Iterations",
    },
    target_acc: {
      type: "number",
      step: 0.01,
      default: 100.0,
      label: "Target accuracy",
    },
    deterministic: {
      type: "boolean",
      default: false,
      label: "Deterministic",
    },
    checkpoint_interval: {
      type: "number",
      step: 1,
      default: 50,
      label: "Checkpoint interval",
    },
    optimizer: {
      type: "select",
      label: "Optimizer",
      options: [
        { label: "LEEA", value: "LEEA" },
        { label: "SGD", value: "SGD" },
        { label: "CoSyNE", value: "CoSyNE" },
      ],
    },
  },
  optimizers_schema: {
    SGD: [
      {
        key: "η",
        label: "\\eta",
        type: "number",
        default: 0.01,
        step: 0.01,
        desc: "Learning rate",
      },
    ],
    LEEA: [
      {
        key: "N",
        label: "N",
        type: "number",
        default: 1000,
        step: 1,
        desc: "Population size",
      },
      {
        key: "pₘ",
        label: "p_{\\mathrm{m}}",
        type: "number",
        default: 0.04,
        step: 0.01,
        desc: "Mutation probability",
      },
      {
        key: "η₀",
        label: "\\eta_0",
        type: "number",
        default: 0.03,
        step: 0.01,
        desc: "Initial mutation step size",
      },
      {
        key: "γ",
        label: "\\gamma",
        type: "number",
        default: 0.99,
        step: 0.01,
        desc: "Mutation decay factor",
      },
      {
        key: "ρ",
        label: "\\rho",
        type: "number",
        default: 0.4,
        step: 0.01,
        desc: "Retention fraction",
      },
      {
        key: "ρₓ",
        label: "\\rho_{\\mathrm{x}}",
        type: "number",
        default: 0.5,
        step: 0.01,
        desc: "Crossover fraction",
      },
      {
        key: "λ",
        label: "\\lambda",
        type: "number",
        default: 0.2,
        step: 0.01,
        desc: "Fitness decay coefficient",
      },
      {
        key: "τ_pat",
        label: "\\tau_{\\mathrm{pat}}",
        type: "number",
        default: 5,
        step: 1,
        desc: "Validation patience threshold",
      },
    ],
    CoSyNE: [
      {
        key: "N",
        label: "N",
        type: "number",
        default: 1000,
        step: 1,
        desc: "Population size",
      },
      {
        key: "k",
        label: "k",
        type: "number",
        default: 4,
        step: 1,
        desc: "Tournament size",
      },
      {
        key: "sigma_m",
        label: "\\sigma_{\\mathrm{m}}",
        type: "number",
        default: 0.03,
        step: 0.01,
        desc: "Mutation standard deviation",
      },
      {
        key: "p_m",
        label: "p_{\\mathrm{m}}",
        type: "number",
        default: 1.0,
        step: 0.01,
        desc: "Mutation probability",
      },
      {
        key: "rho_e",
        label: "\\rho_{\\mathrm{e}}",
        type: "number",
        default: 0.01,
        step: 0.01,
        desc: "Elitism fraction",
      },
      {
        key: "eta_sbx",
        label: "\\eta_{\\mathrm{SBX}}",
        type: "number",
        default: 0,
        step: 1,
        desc: "SBX distribution index",
      },
      {
        key: "num_children",
        label: "\\lambda_{\\mathrm{c}}",
        type: "number",
        default: 0,
        step: 1,
        desc: "Children count",
      },
      {
        key: "permute_all",
        label: "\\pi_{\\mathrm{all}}",
        type: "boolean",
        default: false,
        desc: "Permute all columns",
      },
    ],
  },
}

export const defaultStatus: ExperimentStatus = {
  is_running: false,
  is_paused: false,
  pause_requested: false,
  optimizer: null,
  run_id: null,
  current_step: 0,
  current_loss: 0.0,
  current_mutation_step: null,
  best_acc: 0.0,
  total_elapsed_seconds: 0.0,
  last_iteration_seconds: 0.0,
  loss_mean_since_validation: 0.0,
  loss_stdev_since_validation: 0.0,
  mean_iteration_seconds_since_validation: 0.0,
  requested_device: "cpu",
  device: "cpu",
  device_name: "cpu",
  history: {
    loss: [],
    acc: [],
    train_acc: [],
    val_loss: [],
    memory_mb: [],
    mutation_step: [],
  },
  error: null,
  last_checkpoint_step: null,
  last_checkpoint_acc: null,
  last_checkpoint_saved_at: null,
  checkpoint_path: null,
  best_checkpoint_acc: null,
  best_checkpoint_step: null,
  best_checkpoint_saved_at: null,
  best_checkpoint_path: null,
  reproducibility_mode: "best_effort",
  checkpoint_warnings: [],
}

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

export function configDefaults(schema: SchemaResponse): ExperimentConfig {
  return {
    dataset: selectDefault(schema.config_schema.dataset, "mnist") as DatasetName,
    device: selectDefault(schema.config_schema.device, "gpu") as "cpu" | "gpu",
    seed: numberDefault(schema.config_schema.seed, 42),
    batch_size: numberDefault(schema.config_schema.batch_size, 512),
    iterations: numberDefault(schema.config_schema.iterations, 100000),
    target_acc: numberDefault(schema.config_schema.target_acc, 100.0),
    deterministic: booleanDefault(schema.config_schema.deterministic, false),
    checkpoint_interval: numberDefault(
      schema.config_schema.checkpoint_interval,
      50
    ),
    optimizer: selectDefault(schema.config_schema.optimizer, "LEEA") as OptimizerName,
  }
}

export function optimizerParamDefaults(
  schema: SchemaResponse
): OptimizerParams {
  return Object.fromEntries(
    Object.entries(schema.optimizers_schema).map(([optimizer, fields]) => [
      optimizer,
      Object.fromEntries(fields.map((field) => [field.key, field.default])),
    ])
  )
}

function selectDefault(field: ConfigField, fallback: string): string {
  const options = field.options?.map((option) => option.value)

  if (typeof field.default === "string") {
    if (!options?.length || options.includes(field.default)) {
      return field.default
    }
  }

  return field.options?.[0]?.value ?? fallback
}

function numberDefault(field: ConfigField, fallback: number): number {
  return typeof field.default === "number" ? field.default : fallback
}

function booleanDefault(field: ConfigField, fallback: boolean): boolean {
  return typeof field.default === "boolean" ? field.default : fallback
}

export async function fetchSchema(): Promise<SchemaResponse> {
  const response = await fetch(apiUrl("/api/schema"))
  if (!response.ok) {
    throw new Error("Failed to load schema")
  }
  return response.json()
}

export async function fetchCheckpoints(
  mode: CheckpointListMode = "training"
): Promise<CheckpointSummary[]> {
  const search = new URLSearchParams({ mode })
  const response = await fetch(apiUrl(`/api/checkpoints?${search}`))
  if (!response.ok) {
    throw new Error("Failed to load checkpoints")
  }
  return response.json()
}

export async function createAnalysisComparisonJob(
  left: CheckpointSelection,
  right: CheckpointSelection,
  params: AnalysisComparisonParams,
  force_recompute = false
): Promise<AnalysisComparisonJobStatus> {
  const response = await fetch(apiUrl("/api/analysis/comparisons/jobs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ left, right, params, force_recompute }),
  })
  if (!response.ok) {
    throw new Error("Failed to start comparison")
  }
  return response.json()
}

export async function fetchAnalysisComparisonJob(
  jobId: string
): Promise<AnalysisComparisonJobStatus> {
  const response = await fetch(
    apiUrl(`/api/analysis/comparisons/jobs/${encodeURIComponent(jobId)}`)
  )
  if (!response.ok) {
    throw new Error("Failed to load comparison")
  }
  return response.json()
}

export async function fetchAnalysisComparisonReport(
  jobId: string
): Promise<AnalysisComparisonReport> {
  const response = await fetch(
    apiUrl(`/api/analysis/comparisons/jobs/${encodeURIComponent(jobId)}/report`)
  )
  if (!response.ok) {
    throw new Error("Failed to load comparison report")
  }
  return response.json()
}

export function analysisComparisonEventsUrl(jobId: string): string {
  return apiUrl(
    `/api/analysis/comparisons/jobs/${encodeURIComponent(jobId)}/events`
  )
}

export async function loadCheckpointStatus(
  selection: CheckpointSelection
): Promise<ExperimentStatus> {
  const response = await fetch(apiUrl("/api/checkpoints/load"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(selection),
  })
  if (!response.ok) {
    throw new Error("Failed to load checkpoint")
  }
  return response.json()
}

export async function deleteCheckpointRun(runId: string): Promise<void> {
  const response = await fetch(apiUrl(`/api/checkpoints/${encodeURIComponent(runId)}`), {
    method: "DELETE",
  })
  if (!response.ok) {
    throw new Error("Failed to delete checkpoint")
  }
}

export async function resetExperimentStatus(): Promise<ExperimentStatus> {
  const response = await fetch(apiUrl("/api/experiments/reset"), {
    method: "POST",
  })
  if (!response.ok) {
    throw new Error("Failed to start a new experiment")
  }
  return response.json()
}

export async function fetchStatus(): Promise<ExperimentStatus> {
  const response = await fetch(apiUrl("/api/experiments/status"))
  if (!response.ok) {
    throw new Error("Failed to load status")
  }
  return response.json()
}
