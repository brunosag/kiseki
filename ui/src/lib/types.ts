// The server's JSON (endpoints: the docstring of src/kiseki/server.py). The server is the
// contract: when data is wrong, fix it there.

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

export interface Schema {
  type?: string;
  $ref?: string;
  $defs?: Record<string, Schema>;
  properties?: Record<string, Schema>;
  anyOf?: Schema[];
  enum?: Json[];
  items?: Schema;
  default?: Json;
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
  params: Record<string, Json>;
  seed: number;
  iterations: number;
  label: string;
  [key: string]: Json;
}

export type RunState = "queued" | "running" | "paused" | "completed" | "failed" | "stopped";

export interface Status {
  state: RunState;
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
  seconds?: Record<string, number>;
  created_at?: string;
  started_at?: string;
  updated_at?: string;
  finished_at?: string;
}

export interface Run {
  id: string;
  spec: RunSpec;
  status: Status;
}

export interface RunEvent {
  time: string;
  kind: string;
  [key: string]: Json;
}

export interface RunDetail extends Run {
  path: string;
  models: string[];
  summary: Record<string, Json> | null;
  events: RunEvent[];
}

/** Column-wise rows past a byte offset: {columns: {step: [...], val_accuracy: [...]}}. */
export interface Rows {
  columns: Record<string, (number | null)[]>;
  offset: number;
}

export interface CatalogItem {
  title: string;
  summary: string;
  params: Schema;
}

export interface Dataset {
  title: string;
  default_model: string;
  image_shape: number[];
  class_names: string[];
  train_size?: number;
  validation_size?: number;
}

export interface MachineOption {
  name: string;
  ssh: string | null;
  description: string;
  notice: string | null;
}

export interface Options {
  datasets: Record<string, Dataset>;
  cached: Record<string, boolean>;
  models: string[];
  optimizers: Record<string, CatalogItem>;
  analyses: Record<string, CatalogItem>;
  run: Schema;
  defaults: RunSpec;
  presets: Record<string, Record<string, Json>>;
  machines: MachineOption[];
}

export interface Machine extends MachineOption {
  gpu_awake?: boolean | null;
  runner?: boolean;
  runner_pid?: number | null;
  deployed?: boolean;
  error?: string;
  gpu_error?: string | null;
  gpus?: { name: string; memory_used_mib: number; memory_total_mib: number; utilization: number }[];
}

export interface ReportSpec {
  models: string[];
  labels: string[];
  split: "test" | "val";
  analyses: Record<string, Record<string, Json>>;
  seed: number;
}

export interface Report {
  id: string;
  spec: ReportSpec;
  status: Status;
}

export interface Subject {
  ref: string;
  label: string;
  color: string;
  optimizer: string;
  dataset: string;
  model: string;
}

export interface Section {
  name: string;
  title: string;
  summary: Record<string, Json>;
  data: Record<string, Json>;
  markdown: string;
  figures: string[];
  arrays: string[];
}

export interface ReportDocument {
  dataset: string;
  split: string;
  class_names: string[];
  subjects: Subject[];
  sections: Section[];
}

export interface ReportDetail extends Report {
  report: ReportDocument | null;
}
