import { useEffect, useMemo, useState } from "react"
import ReactPlotly from "react-plotly.js"
import katex from "katex"
import { Moon, Pause, Play, RotateCcw, Square, Sun } from "lucide-react"
import type { ComponentType } from "react"
import type { PlotParams } from "react-plotly.js"
import type { Data, Layout } from "plotly.js"

import "katex/dist/katex.min.css"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useTheme } from "@/components/theme-provider"
import {
  apiUrl,
  configDefaults,
  defaultStatus,
  fallbackSchema,
  fetchSchema,
  fetchStatus,
  optimizerParamDefaults,
  type ConfigField,
  type ExperimentConfig,
  type ExperimentStatus,
  type OptimizerParams,
  type SchemaResponse,
} from "@/lib/api"
import { cn } from "@/lib/utils"

const Plot = (
  ReactPlotly as unknown as {
    default: ComponentType<PlotParams>
  }
).default

const configOrder: (keyof ExperimentConfig)[] = [
  "dataset",
  "device",
  "seed",
  "batch_size",
  "iterations",
  "checkpoint_interval",
  "target_acc",
  "optimizer",
  "deterministic",
]

const eventTypes = [
  "status",
  "started",
  "runtime",
  "step",
  "validation",
  "checkpoint",
  "pause_requested",
  "paused",
  "resumed",
  "completed",
  "stopped",
  "failed",
]

const PLOT_UI_REVISION = "training-telemetry"
const CONFIG_STORAGE_KEY = "kiseki.config.v1"
const OPT_PARAMS_STORAGE_KEY = "kiseki.optimizerParams.v1"
const MARKER_POINT_LIMIT = 20
const Y_AXIS_SEGMENTS = 10
const MUTATION_STEP_AXIS_GRANULARITY = 0.001
const MUTATION_STEP_AXIS_MIN_UPPER_BOUND = 0.001
const MUTATION_STEP_AXIS_PADDING = 1.05

type ResolvedTheme = "dark" | "light"

type PlotPalette = {
  accuracy: string
  axis: string
  grid: string
  hoverBackground: string
  hoverBorder: string
  hoverText: string
  loss: string
  mutationStep: string
  muted: string
  text: string
}

type MultiAxisLayout = Partial<Layout> & {
  yaxis3?: Record<string, unknown>
}

const plotPalettes: Record<ResolvedTheme, PlotPalette> = {
  light: {
    accuracy: "#2563eb",
    axis: "#d4d4d8",
    grid: "#e4e4e7",
    hoverBackground: "#ffffff",
    hoverBorder: "#e4e4e7",
    hoverText: "#18181b",
    loss: "#18181b",
    mutationStep: "#71717a",
    muted: "#71717a",
    text: "#18181b",
  },
  dark: {
    accuracy: "#38bdf8",
    axis: "#52525b",
    grid: "#3f3f46",
    hoverBackground: "#27272a",
    hoverBorder: "#3f3f46",
    hoverText: "#fafafa",
    loss: "#fafafa",
    mutationStep: "#a1a1aa",
    muted: "#a1a1aa",
    text: "#e4e4e7",
  },
}

export function App() {
  const { theme } = useTheme()
  const resolvedTheme = useResolvedTheme(theme)
  const plotPalette = plotPalettes[resolvedTheme]
  const [schema, setSchema] = useState<SchemaResponse>(fallbackSchema)
  const [config, setConfig] = useState<ExperimentConfig>(() =>
    readStoredConfig(fallbackSchema)
  )
  const [optParams, setOptParams] = useState<OptimizerParams>(() =>
    readStoredOptimizerParams(fallbackSchema)
  )
  const [status, setStatus] = useState<ExperimentStatus>(defaultStatus)

  useEffect(() => {
    let ignore = false

    async function loadInitialState() {
      const [nextSchema, nextStatus] = await Promise.all([
        fetchSchema(),
        fetchStatus(),
      ])

      if (ignore) {
        return
      }

      setSchema(nextSchema)
      setConfig(readStoredConfig(nextSchema))
      setOptParams(readStoredOptimizerParams(nextSchema))
      setStatus(nextStatus)
    }

    loadInitialState().catch(() => undefined)

    return () => {
      ignore = true
    }
  }, [])

  useEffect(() => {
    writeStorage(CONFIG_STORAGE_KEY, config)
  }, [config])

  useEffect(() => {
    writeStorage(OPT_PARAMS_STORAGE_KEY, optParams)
  }, [optParams])

  useEffect(() => {
    const source = new EventSource(apiUrl("/api/experiments/events"))
    const handleEvent = (event: MessageEvent<string>) => {
      const payload = JSON.parse(event.data) as
        | ExperimentStatus
        | { status: ExperimentStatus }
      setStatus("status" in payload ? payload.status : payload)
    }

    eventTypes.forEach((eventType) => {
      source.addEventListener(eventType, handleEvent)
    })

    return () => {
      eventTypes.forEach((eventType) => {
        source.removeEventListener(eventType, handleEvent)
      })
      source.close()
    }
  }, [])

  const isRunning = status.is_running
  const isPaused = status.is_paused
  const controlsDisabled = isRunning || isPaused
  const activeOptimizerSchema = schema.optimizers_schema[config.optimizer] ?? []
  const mutationStepHistory = useMemo(
    () => status.history.mutation_step ?? [],
    [status.history.mutation_step]
  )
  const hasMutationStepHistory = mutationStepHistory.length > 0
  const hasCurrentMutationStep =
    typeof status.current_mutation_step === "number"
  const stepAxisUpperBound = nextStepAxisUpperBound(status.current_step)
  const lossAxisUpperBound = lossAxisUpperBoundFor(
    status.history.loss,
    status.current_loss
  )
  const lossTickValues = useMemo(
    () => axisTickValues(lossAxisUpperBound, Y_AXIS_SEGMENTS),
    [lossAxisUpperBound]
  )
  const mutationStepAxisUpperBound = mutationStepAxisUpperBoundFor(
    mutationStepHistory.map((point) => point.value),
    status.current_mutation_step
  )
  const mutationStepTickValues = useMemo(
    () => axisTickValues(mutationStepAxisUpperBound, Y_AXIS_SEGMENTS),
    [mutationStepAxisUpperBound]
  )

  const plotData = useMemo<Data[]>(() => {
    const lossPointCount = status.history.loss.length
    const accuracyPointCount = status.history.acc.length

    const lossTrace: Data = {
      type: "scatter",
      mode: traceMode(lossPointCount),
      name: "Loss",
      uid: "loss",
      x: status.history.loss.map((_, index) => index + 1),
      y: status.history.loss,
      hoverlabel: {
        align: "left",
        bgcolor: plotPalette.hoverBackground,
        bordercolor: plotPalette.hoverBorder,
        font: { color: plotPalette.hoverText, size: 12 },
      },
      hovertemplate: "<b>Loss</b><br>Step %{x}<br>%{y:.4f}<extra></extra>",
      line: { color: plotPalette.loss, width: 1.5 },
      marker: { color: plotPalette.loss, size: 5 },
    }

    const accuracyTrace: Data = {
      type: "scatter",
      mode: accuracyTraceMode(accuracyPointCount),
      name: "Accuracy",
      uid: "accuracy",
      x: status.history.acc.map((point) => point.i),
      y: status.history.acc.map((point) => point.value),
      hoverlabel: {
        align: "left",
        bgcolor: plotPalette.hoverBackground,
        bordercolor: plotPalette.hoverBorder,
        font: { color: plotPalette.hoverText, size: 12 },
      },
      hovertemplate:
        "<b>Accuracy</b><br>Step %{x}<br>%{y:.2f}%<extra></extra>",
      yaxis: "y2",
      line: { color: plotPalette.accuracy, width: 1.5 },
      marker: { color: plotPalette.accuracy, size: 5 },
    }

    if (!mutationStepHistory.length) {
      return [lossTrace, accuracyTrace]
    }

    const mutationStepTrace: Data = {
      type: "scatter",
      mode: "lines",
      name: "Mutation step",
      uid: "mutation-step",
      x: mutationStepHistory.map((point) => point.i),
      y: mutationStepHistory.map((point) => point.value),
      hoverlabel: {
        align: "left",
        bgcolor: plotPalette.hoverBackground,
        bordercolor: plotPalette.hoverBorder,
        font: { color: plotPalette.hoverText, size: 12 },
      },
      hovertemplate:
        "<b>Mutation step</b><br>Step %{x}<br>%{y:.4f}<extra></extra>",
      yaxis: "y3",
      line: { color: plotPalette.mutationStep, width: 1.5, dash: "dot" },
    }

    return [mutationStepTrace, lossTrace, accuracyTrace]
  }, [
    mutationStepHistory,
    plotPalette.accuracy,
    plotPalette.hoverBackground,
    plotPalette.hoverBorder,
    plotPalette.hoverText,
    plotPalette.loss,
    plotPalette.mutationStep,
    status.history.acc,
    status.history.loss,
  ])

  const plotLayout = useMemo<MultiAxisLayout>(
    () => ({
      autosize: true,
      height: 420,
      uirevision: `${PLOT_UI_REVISION}-${stepAxisUpperBound}-${lossAxisUpperBound}-${mutationStepAxisUpperBound}`,
      margin: { l: 52, r: hasMutationStepHistory ? 116 : 58, t: 34, b: 48 },
      xaxis: {
        color: plotPalette.text,
        ...(hasMutationStepHistory ? { domain: [0, 0.86] } : {}),
        linecolor: plotPalette.axis,
        tickfont: { color: plotPalette.muted },
        title: { text: "Step", font: { color: plotPalette.muted } },
        range: [0, stepAxisUpperBound],
        tickmode: "auto",
        nticks: 12,
        fixedrange: false,
        gridcolor: plotPalette.grid,
        zerolinecolor: plotPalette.axis,
      },
      yaxis: {
        color: plotPalette.text,
        linecolor: plotPalette.axis,
        tickfont: { color: plotPalette.muted },
        title: { text: "Loss", font: { color: plotPalette.muted } },
        range: [0, lossAxisUpperBound],
        tickmode: "array",
        tickvals: lossTickValues,
        tickformat: ".2f",
        fixedrange: false,
        gridcolor: plotPalette.grid,
        showgrid: true,
        zeroline: false,
      },
      yaxis2: {
        color: plotPalette.text,
        linecolor: plotPalette.axis,
        tickfont: { color: plotPalette.muted },
        title: { text: "Accuracy (%)", font: { color: plotPalette.muted } },
        range: [0, 100],
        tick0: 0,
        dtick: 100 / Y_AXIS_SEGMENTS,
        fixedrange: false,
        overlaying: "y",
        side: "right",
        showgrid: false,
        zeroline: false,
      },
      ...(hasMutationStepHistory
        ? {
            yaxis3: {
              color: plotPalette.text,
              linecolor: plotPalette.axis,
              tickfont: { color: plotPalette.muted },
              title: {
                text: "Mutation step",
                font: { color: plotPalette.muted },
              },
              range: [0, mutationStepAxisUpperBound],
              tickmode: "array",
              tickvals: mutationStepTickValues,
              tickformat: ".3f",
              fixedrange: false,
              overlaying: "y",
              side: "right",
              anchor: "free",
              position: 0.98,
              showgrid: false,
              zeroline: false,
            },
          }
        : {}),
      legend: {
        orientation: "h",
        x: 0.5,
        y: 1,
        xanchor: "center",
        yanchor: "bottom",
        font: { color: plotPalette.muted },
      },
      hoverlabel: {
        align: "left",
        bgcolor: plotPalette.hoverBackground,
        bordercolor: plotPalette.hoverBorder,
        font: {
          color: plotPalette.hoverText,
          family: "Geist Variable, sans-serif",
          size: 12,
        },
      },
      font: { color: plotPalette.text, family: "Geist Variable, sans-serif" },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      dragmode: false,
    }),
    [
      hasMutationStepHistory,
      lossAxisUpperBound,
      lossTickValues,
      mutationStepAxisUpperBound,
      mutationStepTickValues,
      plotPalette,
      stepAxisUpperBound,
    ]
  )

  async function startExperiment() {
    const response = await fetch(apiUrl("/api/experiments/start"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, opt_params: optParams }),
    })
    if (response.ok) {
      setStatus(await response.json())
    }
  }

  async function stopExperiment() {
    const response = await fetch(apiUrl("/api/experiments/stop"), {
      method: "POST",
    })
    if (response.ok) {
      setStatus(await response.json())
    }
  }

  async function pauseExperiment() {
    const response = await fetch(apiUrl("/api/experiments/pause"), {
      method: "POST",
    })
    if (response.ok) {
      setStatus(await response.json())
    }
  }

  async function resumeExperiment() {
    const response = await fetch(apiUrl("/api/experiments/resume"), {
      method: "POST",
    })
    if (response.ok) {
      setStatus(await response.json())
    }
  }

  function updateConfig<K extends keyof ExperimentConfig>(
    key: K,
    value: ExperimentConfig[K]
  ) {
    setConfig((current) => ({ ...current, [key]: value }))
  }

  function updateOptimizerParam(key: string, value: number) {
    setOptParams((current) => ({
      ...current,
      [config.optimizer]: {
        ...current[config.optimizer],
        [key]: value,
      },
    }))
  }

  return (
    <div className="p-6">
      <div className="mb-4 flex justify-end">
        <ThemeToggle resolvedTheme={resolvedTheme} />
      </div>
      <div className="flex flex-col gap-6 md:flex-row">
        <Card className="w-full max-w-3xl md:max-w-md">
          <CardHeader>
            <CardTitle className="text-xl">Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <Separator />
            <div className="mt-4 flex flex-col gap-2">
              {configOrder.map((key) => {
                const field = schema.config_schema[key]
                return (
                  <ConfigControl
                    key={key}
                    fieldKey={key}
                    field={field}
                    value={config[key]}
                    disabled={controlsDisabled}
                    onChange={(value) => updateConfig(key, value)}
                  />
                )
              })}
            </div>

            <div className="mt-6 rounded-lg border p-4">
              <h4 className="mb-4 pb-1">
                {config.optimizer} parameters
              </h4>
              <Separator />
              <div className="mt-4 grid grid-cols-[max-content_auto_1fr] items-center gap-x-3 gap-y-2">
                {activeOptimizerSchema.map((param) => (
                  <div className="contents" key={param.key}>
                    <span className="text-lg">
                      <MathLabel math={param.label} />
                    </span>
                    <Input
                      className="h-8 w-24"
                      type={param.type}
                      step={param.step}
                      disabled={controlsDisabled}
                      value={optParams[config.optimizer]?.[param.key] ?? ""}
                      onChange={(event) =>
                        updateOptimizerParam(
                          param.key,
                          Number(event.currentTarget.value)
                        )
                      }
                    />
                    <Label className="text-sm font-normal text-muted-foreground">
                      {param.desc}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            <div
              className="mt-6 grid grid-cols-1 gap-2 data-[active=true]:grid-cols-2"
              data-active={isRunning || isPaused}
            >
              {!isRunning && !isPaused ? (
                <Button onClick={startExperiment}>
                  <Play className="size-4" />
                  Start
                </Button>
              ) : null}
              {isRunning ? (
                <Button
                  disabled={status.pause_requested}
                  variant="secondary"
                  onClick={pauseExperiment}
                >
                  <Pause className="size-4" />
                  Pause
                </Button>
              ) : null}
              {isPaused ? (
                <Button onClick={resumeExperiment}>
                  <RotateCcw className="size-4" />
                  Resume
                </Button>
              ) : null}
              {isRunning || isPaused ? (
                <Button variant="destructive" onClick={stopExperiment}>
                  <Square className="size-4" />
                  Stop
                </Button>
              ) : null}
            </div>
          </CardContent>
        </Card>

        <Card className="h-fit w-full">
          <CardHeader>
            <CardTitle className="text-xl">Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <Separator />
            <div
              className={cn(
                "mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2",
                hasCurrentMutationStep ? "xl:grid-cols-6" : "xl:grid-cols-5"
              )}
            >
              <Metric label="Step" value={status.current_step.toString()} />
              <Metric label="Loss" value={status.current_loss.toFixed(4)} />
              <Metric
                label="Best accuracy"
                value={`${status.best_acc.toFixed(2)}%`}
              />
              {hasCurrentMutationStep ? (
                <Metric
                  label="Mutation step"
                  value={formatMutationStep(status.current_mutation_step ?? 0)}
                />
              ) : null}
              <Metric
                label="Total elapsed"
                value={formatDuration(status.total_elapsed_seconds)}
              />
              <Metric
                label="Last iteration"
                value={formatDuration(status.last_iteration_seconds)}
              />
            </div>
            {status.error ? (
              <div className="mt-4 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                {status.error}
              </div>
            ) : null}
            {status.checkpoint_warnings.length ? (
              <div className="mt-4 rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-300">
                {status.checkpoint_warnings.join(" ")}
              </div>
            ) : null}

            <div className="mt-6 w-full">
              <Plot
                className="w-full"
                data={plotData}
                layout={plotLayout}
                config={{
                  displayModeBar: false,
                  displaylogo: false,
                  scrollZoom: false,
                  responsive: true,
                }}
                useResizeHandler
                style={{ width: "100%", height: "420px" }}
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function ThemeToggle({ resolvedTheme }: { resolvedTheme: ResolvedTheme }) {
  const { setTheme } = useTheme()
  const nextTheme = resolvedTheme === "dark" ? "light" : "dark"
  const Icon = resolvedTheme === "dark" ? Sun : Moon

  return (
    <Button
      aria-label={`Switch to ${nextTheme} mode`}
      size="icon"
      title={`Switch to ${nextTheme} mode`}
      variant="outline"
      onClick={() => setTheme(nextTheme)}
    >
      <Icon className="size-4" />
    </Button>
  )
}

function useResolvedTheme(theme: string): ResolvedTheme {
  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>(() =>
    readResolvedTheme()
  )

  useEffect(() => {
    const updateResolvedTheme = () => {
      setResolvedTheme(readResolvedTheme())
    }
    const observer = new MutationObserver(updateResolvedTheme)

    updateResolvedTheme()
    observer.observe(document.documentElement, {
      attributeFilter: ["class"],
      attributes: true,
    })

    return () => {
      observer.disconnect()
    }
  }, [theme])

  return resolvedTheme
}

function readResolvedTheme(): ResolvedTheme {
  if (typeof document === "undefined") {
    return "light"
  }

  return document.documentElement.classList.contains("dark") ? "dark" : "light"
}

function traceMode(pointCount: number): "lines" | "lines+markers" {
  return pointCount > 0 && pointCount <= MARKER_POINT_LIMIT
    ? "lines+markers"
    : "lines"
}

function accuracyTraceMode(
  pointCount: number
): "lines" | "lines+markers" | "markers" {
  if (pointCount === 1) {
    return "markers"
  }

  return traceMode(pointCount)
}

function nextStepAxisUpperBound(step: number): number {
  const safeStep = Number.isFinite(step) && step > 0 ? step : 0
  return (Math.floor(safeStep / 10) + 1) * 10
}

function lossAxisUpperBoundFor(losses: number[], currentLoss: number): number {
  return numericAxisUpperBoundFor(losses, currentLoss)
}

function mutationStepAxisUpperBoundFor(
  values: number[],
  currentValue: number | null | undefined
): number {
  const maxValue = maxFiniteAxisValue(values, currentValue)

  if (maxValue <= 0) {
    return MUTATION_STEP_AXIS_MIN_UPPER_BOUND
  }

  return roundUpToStep(
    maxValue * MUTATION_STEP_AXIS_PADDING,
    MUTATION_STEP_AXIS_GRANULARITY
  )
}

function numericAxisUpperBoundFor(
  values: number[],
  currentValue: number | null | undefined
): number {
  const maxValue = maxFiniteAxisValue(values, currentValue)

  if (maxValue <= 0) {
    return 1
  }

  return niceAxisUpperBound(maxValue)
}

function maxFiniteAxisValue(
  values: number[],
  currentValue: number | null | undefined
): number {
  const finiteValues = values.filter((value) => Number.isFinite(value))
  const safeCurrentValue =
    typeof currentValue === "number" && Number.isFinite(currentValue)
      ? currentValue
      : 0

  return Math.max(0, safeCurrentValue, ...finiteValues)
}

function niceAxisUpperBound(value: number): number {
  const magnitude = 10 ** Math.floor(Math.log10(value))
  const normalized = value / magnitude

  if (normalized <= 1) {
    return magnitude
  }

  if (normalized <= 2) {
    return 2 * magnitude
  }

  if (normalized <= 2.5) {
    return 2.5 * magnitude
  }

  if (normalized <= 5) {
    return 5 * magnitude
  }

  return 10 * magnitude
}

function axisTickValues(upperBound: number, segments: number): number[] {
  return Array.from({ length: segments + 1 }, (_, index) =>
    Number(((upperBound * index) / segments).toPrecision(12))
  )
}

function roundUpToStep(value: number, step: number): number {
  return Number((Math.ceil(value / step) * step).toPrecision(12))
}

function readStoredConfig(schema: SchemaResponse): ExperimentConfig {
  const defaults = configDefaults(schema)
  const stored = readStorage(CONFIG_STORAGE_KEY)

  if (!isRecord(stored)) {
    return defaults
  }

  return {
    dataset: coerceSelect(
      stored.dataset,
      schema.config_schema.dataset,
      defaults.dataset
    ) as ExperimentConfig["dataset"],
    device: coerceSelect(
      stored.device,
      schema.config_schema.device,
      defaults.device
    ) as ExperimentConfig["device"],
    seed: coerceNumber(stored.seed, defaults.seed),
    batch_size: coerceNumber(stored.batch_size, defaults.batch_size),
    iterations: coerceNumber(stored.iterations, defaults.iterations),
    target_acc: coerceNumber(stored.target_acc, defaults.target_acc),
    deterministic: coerceBoolean(stored.deterministic, defaults.deterministic),
    checkpoint_interval: coerceNumber(
      stored.checkpoint_interval,
      defaults.checkpoint_interval
    ),
    optimizer: coerceSelect(
      stored.optimizer,
      schema.config_schema.optimizer,
      defaults.optimizer
    ) as ExperimentConfig["optimizer"],
  }
}

function readStoredOptimizerParams(schema: SchemaResponse): OptimizerParams {
  const defaults = optimizerParamDefaults(schema)
  const stored = readStorage(OPT_PARAMS_STORAGE_KEY)

  if (!isRecord(stored)) {
    return defaults
  }

  return Object.fromEntries(
    Object.entries(defaults).map(([optimizer, params]) => {
      const storedParams = stored[optimizer]

      if (!isRecord(storedParams)) {
        return [optimizer, params]
      }

      return [
        optimizer,
        Object.fromEntries(
          Object.entries(params).map(([key, defaultValue]) => [
            key,
            coerceNumber(storedParams[key], defaultValue),
          ])
        ),
      ]
    })
  )
}

function coerceSelect(
  value: unknown,
  field: ConfigField,
  fallback: string
): string {
  if (typeof value !== "string") {
    return fallback
  }

  const allowedValues = field.options?.map((option) => option.value)
  if (allowedValues?.length && !allowedValues.includes(value)) {
    return fallback
  }

  return value
}

function coerceNumber(value: unknown, fallback: number): number {
  const numberValue =
    typeof value === "number"
      ? value
      : typeof value === "string"
        ? Number(value)
        : NaN

  return Number.isFinite(numberValue) ? numberValue : fallback
}

function coerceBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback
}

function readStorage(key: string): unknown {
  if (typeof window === "undefined") {
    return null
  }

  try {
    const rawValue = window.localStorage.getItem(key)
    return rawValue ? JSON.parse(rawValue) : null
  } catch {
    return null
  }
}

function writeStorage(key: string, value: unknown) {
  if (typeof window === "undefined") {
    return
  }

  try {
    window.localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Ignore storage quota and privacy-mode failures.
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

type ConfigControlProps<K extends keyof ExperimentConfig> = {
  fieldKey: K
  field: ConfigField
  value: ExperimentConfig[K]
  disabled: boolean
  onChange: (value: ExperimentConfig[K]) => void
}

function ConfigControl<K extends keyof ExperimentConfig>({
  fieldKey,
  field,
  value,
  disabled,
  onChange,
}: ConfigControlProps<K>) {
  if (field.type === "boolean") {
    return (
      <div className="flex h-8 items-center justify-between gap-4">
        <Label className="text-sm text-muted-foreground" htmlFor={fieldKey}>
          {field.label}
        </Label>
        <div className="flex w-40 justify-start">
          <Checkbox
            checked={Boolean(value)}
            disabled={disabled}
            id={fieldKey}
            name={fieldKey}
            onCheckedChange={(checked) =>
              onChange(Boolean(checked) as ExperimentConfig[K])
            }
          />
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-between gap-4">
      <Label className="text-sm text-muted-foreground">{field.label}</Label>
      {field.type === "select" ? (
        <Select
          value={String(value)}
          disabled={disabled}
          onValueChange={(nextValue) =>
            onChange(nextValue as ExperimentConfig[K])
          }
        >
          <SelectTrigger className="w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent
            className="data-[side=bottom]:translate-y-0"
            position="popper"
            sideOffset={0}
          >
            {field.options?.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <Input
          className="h-8 w-40"
          type="number"
          step={field.step ?? 1}
          disabled={disabled}
          value={Number(value)}
          onChange={(event) =>
            onChange(Number(event.currentTarget.value) as ExperimentConfig[K])
          }
          name={fieldKey}
        />
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-4">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="text-2xl">{value}</p>
    </div>
  )
}

function formatMutationStep(value: number): string {
  if (!Number.isFinite(value)) {
    return "0.0000"
  }

  return value.toFixed(4)
}

function formatDuration(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) && seconds > 0 ? seconds : 0

  if (safeSeconds < 1) {
    return `${Math.round(safeSeconds * 1000)}ms`
  }

  if (safeSeconds < 60) {
    return `${safeSeconds.toFixed(2)}s`
  }

  const totalSeconds = Math.round(safeSeconds)
  const minutes = Math.floor(totalSeconds / 60)
  const remainingSeconds = (totalSeconds % 60)
    .toString()
    .padStart(2, "0")
  return `${minutes}m ${remainingSeconds}s`
}

function MathLabel({ math }: { math: string }) {
  const html = useMemo(
    () => katex.renderToString(math, { throwOnError: false }),
    [math]
  )

  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export default App
