export type Value =
  string | number | boolean | null | Value[] | { [key: string]: Value };
export interface Schema {
  type?: string;
  $ref?: string;
  $defs?: Record<string, Schema>;
  properties?: Record<string, Schema>;
  anyOf?: Schema[];
  enum?: Value[];
  items?: Schema;
  default?: Value;
  description?: string;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  multipleOf?: number;
}
export interface RunSpec {
  dataset: string;
  model: string;
  optimizer: string;
  params: Record<string, Value>;
  seed: number;
  iterations: number;
  label: string;
  [key: string]: Value;
}
export interface Status {
  state: string;
  step?: number;
  iterations?: number;
  best_accuracy?: number | null;
  best_step?: number | null;
  test_accuracy?: number | null;
  last_accuracy?: number | null;
  last_loss?: number | null;
  elapsed_seconds?: number;
  steps_per_second?: number | null;
  eta_seconds?: number | null;
  machine?: string;
  device?: string | null;
  error?: string | null;
  stage?: string;
  updated_at?: string;
  created_at?: string;
}
export interface Run {
  id: string;
  spec: RunSpec;
  status: Status;
  models?: string[];
  events?: Record<string, Value>[];
  summary?: Record<string, Value> | null;
  path?: string;
}
export interface RegistryItem {
  title?: string;
  summary: string;
  params: Schema;
}
export interface Options {
  datasets: Record<
    string,
    {
      title: string;
      default_model: string;
      image_shape: number[];
      class_names: string[];
    }
  >;
  cached: Record<string, boolean>;
  models: string[];
  optimizers: Record<string, RegistryItem>;
  analyses: Record<string, RegistryItem>;
  defaults: RunSpec;
  run: Schema;
  presets: Record<string, Record<string, Value>>;
  machines: MachineOption[];
}
export interface MachineOption {
  name: string;
  ssh: string | null;
  description: string;
  notice: string | null;
}
export type Columns = Record<string, number[]>;
export interface Rows {
  columns: Columns;
  offset: number;
}
export interface ReportSpec {
  models: string[];
  labels?: string[];
  split: "test" | "val";
  analyses: Record<string, Record<string, Value>>;
  seed: number;
}
export interface Section {
  name: string;
  title: string;
  summary: Record<string, Value>;
  data: Record<string, unknown>;
  markdown: string;
  figures: string[];
  arrays: string[];
}
export interface Subject {
  ref: string;
  label: string;
  color: string;
  optimizer: string;
}
export interface ReportDocument {
  dataset: string;
  split: string;
  class_names: string[];
  subjects: Subject[];
  sections: Section[];
}
export interface ReportRecord {
  id: string;
  spec: ReportSpec;
  status: Status;
  report?: ReportDocument | null;
}
export interface Machine {
  name: string;
  description: string;
  ssh: string | null;
  notice: string | null;
  gpu_awake?: boolean | null;
  runner?: boolean;
  runner_pid?: number | null;
  deployed?: boolean;
  error?: string;
  gpu_error?: string | null;
  gpus?: {
    name: string;
    memory_used_mib: number;
    memory_total_mib: number;
    utilization: number;
  }[];
}
