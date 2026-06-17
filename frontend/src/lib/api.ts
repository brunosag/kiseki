export type SelectOption = {
  label: string
  value: string
}

export type ConfigField = {
  type: "select" | "number"
  label: string
  options?: SelectOption[] | null
  step?: number | null
  default?: string | number | null
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

export type ExperimentConfig = {
  dataset: "mnist"
  device: "cpu" | "gpu"
  seed: number
  batch_size: number
  iterations: number
  target_acc: number
  optimizer: "LEEA" | "SGD"
}

export type AccuracyPoint = {
  i: number
  value: number
}

export type TrainingHistory = {
  loss: number[]
  acc: AccuracyPoint[]
}

export type ExperimentStatus = {
  is_running: boolean
  current_step: number
  current_loss: number
  best_acc: number
  requested_device: string
  device: string
  device_name: string
  history: TrainingHistory
  error?: string | null
}

export type OptimizerParams = Record<string, Record<string, number>>

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ""

export const fallbackSchema: SchemaResponse = {
  config_schema: {
    dataset: {
      type: "select",
      label: "Dataset",
      options: [{ label: "MNIST", value: "mnist" }],
    },
    device: {
      type: "select",
      label: "Device",
      options: [
        { label: "CPU", value: "cpu" },
        { label: "GPU", value: "gpu" },
      ],
    },
    seed: { type: "number", step: 1, default: 42, label: "Seed" },
    batch_size: {
      type: "number",
      step: 1,
      default: 1000,
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
        default: 200,
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
        default: 25,
        step: 1,
        desc: "Validation patience threshold",
      },
    ],
  },
}

export const defaultStatus: ExperimentStatus = {
  is_running: false,
  current_step: 0,
  current_loss: 0.0,
  best_acc: 0.0,
  requested_device: "cpu",
  device: "cpu",
  device_name: "cpu",
  history: { loss: [], acc: [] },
  error: null,
}

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

export function configDefaults(schema: SchemaResponse): ExperimentConfig {
  return {
    dataset: firstOption(schema.config_schema.dataset, "mnist") as "mnist",
    device: firstOption(schema.config_schema.device, "cpu") as "cpu" | "gpu",
    seed: numberDefault(schema.config_schema.seed, 42),
    batch_size: numberDefault(schema.config_schema.batch_size, 1000),
    iterations: numberDefault(schema.config_schema.iterations, 100000),
    target_acc: numberDefault(schema.config_schema.target_acc, 100.0),
    optimizer: firstOption(schema.config_schema.optimizer, "LEEA") as
      | "LEEA"
      | "SGD",
  }
}

export function optimizerParamDefaults(schema: SchemaResponse): OptimizerParams {
  return Object.fromEntries(
    Object.entries(schema.optimizers_schema).map(([optimizer, fields]) => [
      optimizer,
      Object.fromEntries(fields.map((field) => [field.key, field.default])),
    ])
  )
}

function firstOption(field: ConfigField, fallback: string): string {
  return field.options?.[0]?.value ?? fallback
}

function numberDefault(field: ConfigField, fallback: number): number {
  return typeof field.default === "number" ? field.default : fallback
}

export async function fetchSchema(): Promise<SchemaResponse> {
  const response = await fetch(apiUrl("/api/schema"))
  if (!response.ok) {
    throw new Error("Failed to load schema")
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
