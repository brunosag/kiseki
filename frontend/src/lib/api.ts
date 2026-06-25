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
  type: "number"
  default: number
  step: number
  desc: string
}

export type SchemaResponse = {
  config_schema: Record<string, ConfigField>
  optimizers_schema: Record<string, OptimizerParamField[]>
}

export type DatasetName = "mnist" | "cifar10"

export type ExperimentConfig = {
  dataset: DatasetName
  device: "cpu" | "gpu"
  seed: number
  batch_size: number
  iterations: number
  target_acc: number
  deterministic: boolean
  checkpoint_interval: number
  optimizer: "LEEA" | "SGD"
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
  optimizer: "LEEA" | "SGD"
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
  mutation_step: MutationStepPoint[]
}

export type ExperimentStatus = {
  is_running: boolean
  is_paused: boolean
  pause_requested: boolean
  optimizer?: "LEEA" | "SGD" | null
  run_id?: string | null
  current_step: number
  current_loss: number
  current_mutation_step?: number | null
  best_acc: number
  total_elapsed_seconds: number
  last_iteration_seconds: number
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

export type OptimizerParams = Record<string, Record<string, number>>

export type TsneLearningRateMode = "auto" | "numeric"

export type TsneParams = {
  perplexity: number
  max_iter: number
  learning_rate_mode: TsneLearningRateMode
  learning_rate?: number | null
  angle: number
  pca_components: number
  seed?: number | null
  use_pca: boolean
}

export type TsnePoint = {
  x: number
  y: number
  label: number
  prediction: number
  correct: boolean
}

export type TsneAnalysisResponse = {
  checkpoint: CheckpointSummary
  params: TsneParams
  points: TsnePoint[]
}

export type LrpParams = {
  sample_count: number
  seed?: number | null
}

export type LrpSample = {
  index: number
  label: number
  prediction: number
  target: number
  correct: boolean
  score: number
  delta: number
  image: number[][][]
  relevance: number[][]
}

export type LrpAnalysisResponse = {
  checkpoint: CheckpointSummary
  params: LrpParams
  samples: LrpSample[]
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
  requested_device: "cpu",
  device: "cpu",
  device_name: "cpu",
  history: { loss: [], acc: [], mutation_step: [] },
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
    optimizer: selectDefault(schema.config_schema.optimizer, "LEEA") as
      | "LEEA"
      | "SGD",
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

export async function computeTsneAnalysis(
  checkpoint: CheckpointSelection,
  params: TsneParams
): Promise<TsneAnalysisResponse> {
  const response = await fetch(apiUrl("/api/analysis/tsne"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint, params }),
  })
  if (!response.ok) {
    throw new Error("Failed to compute t-SNE")
  }
  return response.json()
}

export async function computeLrpAnalysis(
  checkpoint: CheckpointSelection,
  params: LrpParams
): Promise<LrpAnalysisResponse> {
  const response = await fetch(apiUrl("/api/analysis/lrp"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint, params }),
  })
  if (!response.ok) {
    throw new Error("Failed to compute LRP")
  }
  return response.json()
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
