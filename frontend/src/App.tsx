import { useEffect, useMemo, useState } from "react"
import ReactPlotly from "react-plotly.js"
import katex from "katex"
import {
  FolderOpen,
  Moon,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Sun,
} from "lucide-react"
import type { ChangeEvent, ComponentType } from "react"
import type { PlotParams } from "react-plotly.js"
import type { Data, Layout } from "plotly.js"

import "katex/dist/katex.min.css"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group"
import { Label } from "@/components/ui/label"
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
  fetchCheckpoints,
  fetchSchema,
  fetchStatus,
  loadCheckpointStatus,
  type CheckpointSelection,
  type CheckpointSummary,
  optimizerParamDefaults,
  resetExperimentStatus,
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
  "optimizer",
  "device",
  "seed",
  "batch_size",
  "iterations",
  "checkpoint_interval",
  "target_acc",
  "deterministic",
]

const runtimeEditableConfigKeys = new Set<keyof ExperimentConfig>([
  "iterations",
  "target_acc",
  "checkpoint_interval",
])

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
const MUTATION_STEP_AXIS_MIN_UPPER_BOUND = 0.001
const MUTATION_STEP_AXIS_PADDING = 1.05
const MUTATION_STEP_AXIS_SEGMENTS = 8
const MUTATION_STEP_TICK_MAX_DECIMALS = 3
const MUTATION_STEP_PLOT_DOMAIN_END = 0.88

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
    grid: "rgba(24, 24, 27, 0.08)",
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
    grid: "rgba(250, 250, 250, 0.08)",
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
  const [loadedCheckpoint, setLoadedCheckpoint] =
    useState<CheckpointSummary | null>(null)

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
    if (loadedCheckpoint) {
      return
    }

    writeStorage(CONFIG_STORAGE_KEY, config)
  }, [config, loadedCheckpoint])

  useEffect(() => {
    if (loadedCheckpoint) {
      return
    }

    writeStorage(OPT_PARAMS_STORAGE_KEY, optParams)
  }, [loadedCheckpoint, optParams])

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
  const isRunActive = isRunning || isPaused
  const controlsDisabled = isRunActive
  const bestAccuracyStep = useMemo(
    () =>
      bestAccuracyStepFor(
        status.history.acc,
        status.best_acc,
        status.best_checkpoint_step
      ),
    [status.best_acc, status.best_checkpoint_step, status.history.acc]
  )
  const currentCheckpointSelection = useMemo(
    () =>
      selectionFromCheckpoint(loadedCheckpoint) ?? selectionFromStatus(status),
    [loadedCheckpoint, status]
  )
  const showNewExperiment =
    currentCheckpointSelection !== null ||
    Boolean(status.run_id) ||
    status.current_step > 0 ||
    isRunActive
  const activeOptimizerSchema = schema.optimizers_schema[config.optimizer] ?? []
  const mutationStepHistory = useMemo(
    () => status.history.mutation_step ?? [],
    [status.history.mutation_step]
  )
  const hasMutationStepHistory = mutationStepHistory.length > 0
  const showMutationStepAxis =
    config.optimizer === "LEEA" ||
    status.optimizer === "LEEA" ||
    hasMutationStepHistory
  const shouldUseSelectedMutationStep =
    config.optimizer === "LEEA" || status.optimizer === "LEEA"
  const selectedInitialMutationStep = initialMutationStepFor(schema, optParams)
  const latestMutationStep =
    mutationStepHistory.length > 0
      ? mutationStepHistory[mutationStepHistory.length - 1]?.value
      : undefined
  const mutationStepMetricValue =
    status.current_mutation_step ??
    latestMutationStep ??
    selectedInitialMutationStep ??
    0
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
    status.current_mutation_step,
    shouldUseSelectedMutationStep ? selectedInitialMutationStep : undefined
  )
  const mutationStepTickValues = useMemo(
    () =>
      niceTickValues(mutationStepAxisUpperBound, MUTATION_STEP_AXIS_SEGMENTS),
    [mutationStepAxisUpperBound]
  )
  const mutationStepTickText = useMemo(
    () => roundedTickText(mutationStepTickValues),
    [mutationStepTickValues]
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
      hovertemplate: "<b>Accuracy</b><br>Step %{x}<br>%{y:.2f}%<extra></extra>",
      yaxis: "y2",
      line: { color: plotPalette.accuracy, width: 1.5 },
      marker: { color: plotPalette.accuracy, size: 5 },
    }

    if (!mutationStepHistory.length && !showMutationStepAxis) {
      return [lossTrace, accuracyTrace]
    }

    const mutationStepTrace: Data = mutationStepHistory.length
      ? {
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
      : {
          type: "scatter",
          mode: "markers",
          name: "Mutation step",
          uid: "mutation-step-axis-anchor",
          x: [0],
          y: [0],
          hoverinfo: "skip",
          marker: { color: "rgba(0,0,0,0)", size: 1 },
          showlegend: false,
          yaxis: "y3",
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
    showMutationStepAxis,
    status.history.acc,
    status.history.loss,
  ])

  const plotLayout = useMemo<MultiAxisLayout>(
    () => ({
      autosize: true,
      height: 420,
      uirevision: `${PLOT_UI_REVISION}-${stepAxisUpperBound}-${lossAxisUpperBound}-${mutationStepAxisUpperBound}`,
      margin: { l: 52, r: showMutationStepAxis ? 116 : 58, t: 42, b: 48 },
      xaxis: {
        color: plotPalette.text,
        ...(showMutationStepAxis
          ? { domain: [0, MUTATION_STEP_PLOT_DOMAIN_END] }
          : {}),
        linecolor: plotPalette.axis,
        tickfont: { color: plotPalette.muted },
        title: { text: "Step", font: { color: plotPalette.muted } },
        range: [0, stepAxisUpperBound],
        tickmode: "auto",
        nticks: 12,
        fixedrange: true,
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
        fixedrange: true,
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
        fixedrange: true,
        overlaying: "y",
        side: "right",
        showgrid: false,
        zeroline: false,
      },
      ...(showMutationStepAxis
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
              ticktext: mutationStepTickText,
              fixedrange: true,
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
        y: 1.0,
        xanchor: "center",
        yanchor: "bottom",
        uirevision: PLOT_UI_REVISION,
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
      lossAxisUpperBound,
      lossTickValues,
      mutationStepAxisUpperBound,
      mutationStepTickText,
      mutationStepTickValues,
      plotPalette,
      showMutationStepAxis,
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

  async function pauseExperiment() {
    const response = await fetch(apiUrl("/api/experiments/pause"), {
      method: "POST",
    })
    if (response.ok) {
      setStatus(await response.json())
      setLoadedCheckpoint(null)
    }
  }

  async function resumeExperiment() {
    const response = await fetch(apiUrl("/api/experiments/resume"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        iterations: config.iterations,
        target_acc: config.target_acc,
        checkpoint_interval: config.checkpoint_interval,
      }),
    })
    if (response.ok) {
      setStatus(await response.json())
      setLoadedCheckpoint(null)
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

  async function loadCheckpoint(checkpoint: CheckpointSummary) {
    const nextStatus = await loadCheckpointStatus({
      run_id: checkpoint.run_id,
      kind: checkpoint.kind,
    })
    setLoadedCheckpoint(checkpoint)
    setConfig(checkpoint.config)
    setOptParams(mergeOptimizerParams(schema, checkpoint.optimizer_params))
    setStatus(nextStatus)
  }

  async function newExperiment() {
    const nextStatus = await resetExperimentStatus()
    setLoadedCheckpoint(null)
    setConfig(readStoredConfig(schema))
    setOptParams(readStoredOptimizerParams(schema))
    setStatus(nextStatus)
  }

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <CheckpointPicker
            currentSelection={currentCheckpointSelection}
            disabled={isRunning}
            schema={schema}
            onLoad={loadCheckpoint}
          />
          {showNewExperiment ? (
            <Button
              disabled={isRunning}
              variant="secondary"
              onClick={newExperiment}
            >
              <Plus className="size-4" />
              New experiment
            </Button>
          ) : null}
        </div>
        <ThemeToggle resolvedTheme={resolvedTheme} />
      </div>
      <div className="flex flex-col gap-6 md:flex-row">
        <div className="flex w-full max-w-3xl flex-col gap-3 md:max-w-md">
          <Card className="w-full">
            <CardHeader>
              <CardTitle className="text-xl">Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-2">
                {configOrder.map((key) => {
                  const field = schema.config_schema[key]
                  return (
                    <ConfigControl
                      key={key}
                      fieldKey={key}
                      field={field}
                      value={config[key]}
                      disabled={
                        runtimeEditableConfigKeys.has(key)
                          ? isRunning
                          : controlsDisabled
                      }
                      onChange={(value) => updateConfig(key, value)}
                    />
                  )
                })}
              </div>

              <div className="mt-6 flex flex-col gap-3 rounded-lg border p-4">
                <h4 className="font-medium">{config.optimizer} parameters</h4>
                <div className="mt-2 grid grid-cols-[max-content_auto_1fr] items-center gap-x-3 gap-y-2">
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

              <div className="mt-6 grid grid-cols-1 gap-2">
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
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="h-fit w-full">
          <CardHeader>
            <CardTitle className="text-xl">Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={cn(
                "grid gap-x-4 gap-y-5 rounded-lg sm:grid-cols-3",
                showMutationStepAxis ? "xl:grid-cols-6" : "xl:grid-cols-5"
              )}
            >
              <Metric label="Step" value={status.current_step.toString()} />
              <Metric
                label="Total elapsed"
                value={formatDuration(status.total_elapsed_seconds)}
              />
              <Metric
                label="Last iteration"
                value={formatSeconds(status.last_iteration_seconds)}
              />
              <Metric label="Loss" value={status.current_loss.toFixed(4)} />
              <Metric
                label="Best accuracy"
                value={`${status.best_acc.toFixed(2)}%`}
                detail={
                  bestAccuracyStep
                    ? `at ${formatInteger(bestAccuracyStep)}`
                    : undefined
                }
              />
              {showMutationStepAxis ? (
                <Metric
                  label="Mutation step"
                  value={formatMutationStep(mutationStepMetricValue)}
                />
              ) : null}
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

type CheckpointPickerProps = {
  currentSelection: CheckpointSelection | null
  disabled: boolean
  schema: SchemaResponse
  onLoad: (checkpoint: CheckpointSummary) => Promise<void>
}

function CheckpointPicker({
  currentSelection,
  disabled,
  schema,
  onLoad,
}: CheckpointPickerProps) {
  const [open, setOpen] = useState(false)
  const [checkpoints, setCheckpoints] = useState<CheckpointSummary[]>([])
  const [pendingSelection, setPendingSelection] =
    useState<CheckpointSelection | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    let ignore = false

    fetchCheckpoints()
      .then((nextCheckpoints) => {
        if (ignore) {
          return
        }

        setCheckpoints(nextCheckpoints)
      })
      .catch(() => {
        if (ignore) {
          return
        }

        setError("Failed to load checkpoints")
        setCheckpoints([])
      })
      .finally(() => {
        if (!ignore) {
          setIsLoading(false)
        }
      })

    return () => {
      ignore = true
    }
  }, [open])

  const pendingCheckpoint = pendingSelection
    ? (checkpoints.find((checkpoint) =>
        sameCheckpoint(checkpoint, pendingSelection)
      ) ?? null)
    : null
  const canLoad =
    pendingCheckpoint !== null &&
    !sameCheckpoint(pendingCheckpoint, currentSelection)
  const buttonLabel = currentSelection?.run_id ?? "Load checkpoint"

  function togglePendingSelection(checkpoint: CheckpointSummary) {
    setPendingSelection((current) =>
      sameCheckpoint(current, checkpoint)
        ? null
        : selectionFromCheckpoint(checkpoint)
    )
  }

  async function loadPendingCheckpoint() {
    if (!pendingCheckpoint) {
      return
    }

    setError(null)
    try {
      await onLoad(pendingCheckpoint)
      setOpen(false)
    } catch {
      setError("Failed to load checkpoint")
    }
  }

  function handleOpenChange(nextOpen: boolean) {
    if (nextOpen) {
      setPendingSelection(currentSelection)
      setIsLoading(true)
      setError(null)
    }

    setOpen(nextOpen)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button
          className="max-w-[18rem] justify-start"
          disabled={disabled}
          variant="outline"
        >
          <FolderOpen className="size-4" />
          <span className="truncate">{buttonLabel}</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Checkpoints</DialogTitle>
          <DialogDescription className="sr-only">
            Select a checkpoint to load into the paused experiment view.
          </DialogDescription>
        </DialogHeader>

        <div className="max-h-[min(60vh,32rem)] overflow-y-auto pr-1">
          {isLoading ? (
            <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
              Loading checkpoints
            </div>
          ) : null}
          {error ? (
            <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
              {error}
            </div>
          ) : null}
          {!isLoading && !error && checkpoints.length === 0 ? (
            <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
              No checkpoints found
            </div>
          ) : null}
          {!isLoading && !error && checkpoints.length > 0 ? (
            <div className="grid gap-2">
              {checkpoints.map((checkpoint) => {
                const selected = sameCheckpoint(checkpoint, pendingSelection)
                const optimizerParams = optimizerParamEntries(
                  checkpoint,
                  schema
                )

                return (
                  <button
                    aria-label={`Select ${checkpoint.kind} checkpoint for ${checkpoint.optimizer} ${checkpoint.dataset}, seed ${checkpoint.seed}, step ${checkpoint.step}`}
                    aria-pressed={selected}
                    className={cn(
                      "rounded-lg border p-4 text-left transition-colors hover:bg-muted/60 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none",
                      selected
                        ? "border-primary bg-primary/5"
                        : "border-border bg-background"
                    )}
                    key={`${checkpoint.run_id}-${checkpoint.kind}`}
                    type="button"
                    onClick={() => togglePendingSelection(checkpoint)}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="truncate font-medium">
                            {checkpoint.optimizer} ·{" "}
                            {formatDatasetName(checkpoint.dataset)}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground/80">
                          Seed {checkpoint.seed} · Batch size{" "}
                          {formatInteger(checkpoint.config.batch_size)}
                        </p>
                      </div>
                      <div className="flex shrink-0 items-center gap-2">
                        <span className="text-sm text-muted-foreground/80">
                          {formatCheckpointDate(checkpoint.saved_at)}
                        </span>
                        <Badge
                          className="tracking-tight text-foreground/70 uppercase"
                          variant="outline"
                        >
                          {checkpoint.kind}
                        </Badge>
                      </div>
                    </div>

                    <div className="mt-4 flex flex-wrap items-start gap-x-16 gap-y-3">
                      <CheckpointMetric
                        label="Accuracy"
                        value={formatOptionalPercent(checkpoint.accuracy)}
                      />
                      <CheckpointMetric
                        label="Step"
                        value={formatInteger(checkpoint.step)}
                      />
                      <CheckpointMetric
                        label="Elapsed"
                        value={formatOptionalDuration(
                          checkpoint.total_elapsed_seconds
                        )}
                      />
                    </div>

                    {optimizerParams.length > 0 ? (
                      <div className="mt-5 flex flex-wrap gap-x-8 gap-y-2 text-sm">
                        {optimizerParams.map(([key, label, value]) => (
                          <span
                            className="inline-flex min-w-0 items-baseline gap-1.5"
                            key={key}
                          >
                            <span className="shrink-0 text-muted-foreground">
                              <MathLabel math={label} />
                            </span>
                            <span className="truncate text-foreground/90">
                              {formatParamValue(value)}
                            </span>
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </button>
                )
              })}
            </div>
          ) : null}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button disabled={!canLoad} onClick={loadPendingCheckpoint}>
            Load
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function CheckpointMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-lg leading-tight font-medium">{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
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

function selectionFromCheckpoint(
  checkpoint: CheckpointSummary | null
): CheckpointSelection | null {
  if (!checkpoint) {
    return null
  }

  return { run_id: checkpoint.run_id, kind: checkpoint.kind }
}

function selectionFromStatus(
  status: ExperimentStatus
): CheckpointSelection | null {
  if (!status.run_id) {
    return null
  }

  if (
    status.checkpoint_path ||
    status.last_checkpoint_saved_at ||
    status.last_checkpoint_step != null
  ) {
    return { run_id: status.run_id, kind: "latest" }
  }

  if (
    status.best_checkpoint_path ||
    status.best_checkpoint_saved_at ||
    status.best_checkpoint_step != null
  ) {
    return { run_id: status.run_id, kind: "best" }
  }

  return null
}

function sameCheckpoint(
  left: CheckpointSelection | null,
  right: CheckpointSelection | null
): boolean {
  if (!left || !right) {
    return left === right
  }

  return left.run_id === right.run_id && left.kind === right.kind
}

function mergeOptimizerParams(
  schema: SchemaResponse,
  checkpointParams: OptimizerParams
): OptimizerParams {
  const defaults = optimizerParamDefaults(schema)

  return Object.fromEntries(
    Object.entries(defaults).map(([optimizer, params]) => [
      optimizer,
      {
        ...params,
        ...(checkpointParams[optimizer] ?? {}),
      },
    ])
  )
}

function formatOptionalPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a"
  }

  return `${value.toFixed(2)}%`
}

function formatOptionalDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) {
    return "n/a"
  }

  return formatDuration(seconds)
}

function bestAccuracyStepFor(
  points: { i: number; value: number }[],
  bestAccuracy: number,
  fallbackStep: number | null | undefined
): number | null {
  if (points.length > 0 && Number.isFinite(bestAccuracy)) {
    const bestPoint = points.find((point) => point.value === bestAccuracy)
    if (bestPoint) {
      return bestPoint.i
    }
  }

  return fallbackStep ?? null
}

function formatDatasetName(dataset: string): string {
  return dataset.toUpperCase()
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 0,
  }).format(value)
}

function optimizerParamEntries(
  checkpoint: CheckpointSummary,
  schema: SchemaResponse
): [string, string, number][] {
  const params = checkpoint.optimizer_params[checkpoint.optimizer] ?? {}
  const fields = schema.optimizers_schema[checkpoint.optimizer] ?? []
  const fieldKeys = new Set(fields.map((field) => field.key))
  const schemaEntries: [string, string, number][] = fields.flatMap((field) => {
    const value = params[field.key]
    return Number.isFinite(value) ? [[field.key, field.label, value]] : []
  })
  const extraEntries: [string, string, number][] = Object.entries(params)
    .filter(([key, value]) => !fieldKeys.has(key) && Number.isFinite(value))
    .map(([key, value]) => [key, key, value])

  return [...schemaEntries, ...extraEntries]
}

function formatParamValue(value: number): string {
  if (Number.isInteger(value)) {
    return formatInteger(value)
  }

  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 6,
  }).format(value)
}

function formatCheckpointDate(savedAt: string): string {
  const date = new Date(savedAt)
  if (Number.isNaN(date.getTime())) {
    return savedAt
  }
  const options: Intl.DateTimeFormatOptions = {
    day: "numeric",
    hour: "2-digit",
    hour12: false,
    minute: "2-digit",
    month: "short",
  }

  if (date.getFullYear() !== new Date().getFullYear()) {
    options.year = "numeric"
  }

  return date.toLocaleString(undefined, options)
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
  currentValue: number | null | undefined,
  selectedInitialValue: number | null | undefined
): number {
  const selectedValue =
    typeof selectedInitialValue === "number" &&
    Number.isFinite(selectedInitialValue)
      ? selectedInitialValue
      : undefined
  const maxObservedValue = maxFiniteAxisValue(values, currentValue)
  const baselineValue = selectedValue ?? maxObservedValue

  if (baselineValue <= 0) {
    return MUTATION_STEP_AXIS_MIN_UPPER_BOUND
  }

  if (maxObservedValue <= baselineValue) {
    return Math.max(MUTATION_STEP_AXIS_MIN_UPPER_BOUND, baselineValue)
  }

  return Math.max(
    MUTATION_STEP_AXIS_MIN_UPPER_BOUND,
    niceAxisUpperBound(maxObservedValue * MUTATION_STEP_AXIS_PADDING)
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
  ...currentValues: (number | null | undefined)[]
): number {
  const finiteValues = values.filter((value) => Number.isFinite(value))
  const finiteCurrentValues = currentValues.filter(
    (value): value is number =>
      typeof value === "number" && Number.isFinite(value)
  )

  return Math.max(0, ...finiteCurrentValues, ...finiteValues)
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

function niceTickValues(
  upperBound: number,
  preferredSegments: number
): number[] {
  const safeUpperBound =
    Number.isFinite(upperBound) && upperBound > 0 ? upperBound : 1
  const rawStep = safeUpperBound / preferredSegments
  const tickStep = niceStepSize(rawStep)
  const tickCount = Math.ceil(safeUpperBound / tickStep)
  const tickValues = Array.from({ length: tickCount + 1 }, (_, index) =>
    Number((tickStep * index).toPrecision(12))
  ).filter((value) => value <= safeUpperBound + Number.EPSILON)
  const roundedUpperBound = Number(safeUpperBound.toPrecision(12))
  const lastTick = tickValues[tickValues.length - 1] ?? 0

  if (Math.abs(lastTick - roundedUpperBound) > roundedUpperBound * 1e-9) {
    tickValues.push(roundedUpperBound)
  }

  return tickValues
}

function niceStepSize(value: number): number {
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

function roundedTickText(values: number[]): string[] {
  const decimals = Math.min(
    MUTATION_STEP_TICK_MAX_DECIMALS,
    maxDecimalPlacesForValues(values)
  )

  return values.map((value) => formatRoundedTick(value, decimals))
}

function maxDecimalPlacesForValues(values: number[]): number {
  return Math.max(0, ...values.map(decimalPlacesForRoundedValue))
}

function decimalPlacesForRoundedValue(value: number): number {
  if (!Number.isFinite(value)) {
    return 0
  }

  const text = Math.abs(value)
    .toFixed(MUTATION_STEP_TICK_MAX_DECIMALS)
    .replace(/0+$/, "")
    .replace(/\.$/, "")
  const decimalPointIndex = text.indexOf(".")

  return decimalPointIndex === -1 ? 0 : text.length - decimalPointIndex - 1
}

function formatRoundedTick(value: number, decimals: number): string {
  const normalizedValue = Object.is(value, -0) ? 0 : value
  return normalizedValue.toFixed(decimals)
}

function initialMutationStepFor(
  schema: SchemaResponse,
  optParams: OptimizerParams
): number | undefined {
  const mutationStepField = schema.optimizers_schema.LEEA?.find(
    (field) => field.label === "\\eta_0"
  )

  if (!mutationStepField) {
    return undefined
  }

  const value = optParams.LEEA?.[mutationStepField.key]
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : mutationStepField.default
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
        <NumberConfigInput
          fieldKey={fieldKey}
          step={field.step ?? 1}
          disabled={disabled}
          value={Number(value)}
          onChange={(nextValue) => onChange(nextValue as ExperimentConfig[K])}
        />
      )}
    </div>
  )
}

type NumberConfigInputProps<K extends keyof ExperimentConfig> = {
  fieldKey: K
  step: number
  disabled: boolean
  value: number
  onChange: (value: number) => void
}

function NumberConfigInput<K extends keyof ExperimentConfig>({
  fieldKey,
  step,
  disabled,
  value,
  onChange,
}: NumberConfigInputProps<K>) {
  const inputProps = {
    type: "number",
    step,
    disabled,
    value,
    name: fieldKey,
    onChange: (event: ChangeEvent<HTMLInputElement>) =>
      onChange(Number(event.currentTarget.value)),
  }

  if (fieldKey === "target_acc") {
    return (
      <InputGroup className="w-40">
        <InputGroupInput {...inputProps} />
        <InputGroupAddon align="inline-end">%</InputGroupAddon>
      </InputGroup>
    )
  }

  return <Input className="h-8 w-40" {...inputProps} />
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string
  value: string
  detail?: string
}) {
  return (
    <div className="min-w-0 rounded-xl border bg-input px-4 py-3">
      <p className="text-sm text-muted-foreground/80">{label}</p>
      <p className="flex min-w-0 items-baseline gap-2">
        <span className="truncate text-2xl tabular-nums">{value}</span>
        {detail ? (
          <span className="truncate text-xs text-muted-foreground/60">
            {detail}
          </span>
        ) : null}
      </p>
    </div>
  )
}

function formatMutationStep(value: number): string {
  if (!Number.isFinite(value)) {
    return "0.0000"
  }

  return value.toFixed(4)
}

function formatSeconds(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) && seconds > 0 ? seconds : 0

  return `${safeSeconds.toFixed(3)}s`
}

function formatDuration(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) && seconds > 0 ? seconds : 0
  const totalSeconds = Math.round(safeSeconds)

  if (totalSeconds < 60) {
    return `${totalSeconds}s`
  }

  const hours = Math.floor(totalSeconds / 3600)

  if (hours > 0) {
    const minutes = Math.floor((totalSeconds % 3600) / 60)
    const remainingSeconds = totalSeconds % 60
    const parts = [`${hours}h`]

    if (minutes > 0) {
      parts.push(`${minutes}m`)
    }

    if (remainingSeconds > 0) {
      parts.push(`${remainingSeconds}s`)
    }

    return parts.join(" ")
  }

  const minutes = Math.floor(totalSeconds / 60)
  const remainingSeconds = (totalSeconds % 60).toString().padStart(2, "0")
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
