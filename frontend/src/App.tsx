import { useEffect, useMemo, useRef, useState } from "react"
import ReactPlotly from "react-plotly.js"
import katex from "katex"
import {
  AlertTriangle,
  ArrowDown,
  ArrowUpDown,
  CircleHelp,
  Funnel,
  FolderOpen,
  LoaderCircle,
  Moon,
  Pause,
  Play,
  Plus,
  Sun,
  Trash2,
} from "lucide-react"
import type {
  ChangeEvent,
  ComponentType,
  KeyboardEvent,
  ReactElement,
  RefObject,
} from "react"
import type { PlotParams } from "react-plotly.js"
import type { Data, Layout } from "plotly.js"

import "katex/dist/katex.min.css"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import {
  Alert,
  AlertAction,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert"
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
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card"
import { Input } from "@/components/ui/input"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { useTheme } from "@/components/theme-provider"
import {
  analysisComparisonEventsUrl,
  apiUrl,
  configDefaults,
  createAnalysisComparisonJob,
  deleteCheckpointRun,
  defaultStatus,
  fallbackSchema,
  fetchAnalysisComparisonJob,
  fetchCheckpoints,
  fetchSchema,
  fetchStatus,
  loadCheckpointStatus,
  type CheckpointSelection,
  type CheckpointListMode,
  type CheckpointSummary,
  optimizerParamDefaults,
  resetExperimentStatus,
  type AnalysisComparisonJobStatus,
  type AnalysisComparisonParams,
  type AnalysisComparisonReport,
  type AnalysisEmbeddingProjection,
  type AnalysisLrpSample,
  type AnalysisTableRow,
  type ConfigField,
  type ExperimentConfig,
  type ExperimentStatus,
  type OptimizerParamValue,
  type OptimizerParams,
  type SchemaResponse,
  type SelectOption,
} from "@/lib/api"
import { cn } from "@/lib/utils"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"

const Plot = (
  ReactPlotly as unknown as {
    default: ComponentType<PlotParams>
  }
).default

const OPTIMIZER_PARAM_HELP: Record<string, string> = {
  eta_sbx: "0 uses one-point crossover; positive values use SBX.",
  num_children: "0 uses the automatic CoSyNE child count.",
  permute_all:
    "If enabled, permutes every parameter column rather than rank-selective columns.",
}

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
const MUTATION_STEP_TICK_DECIMALS = 2
const MUTATION_STEP_PLOT_DOMAIN_END = 0.925
const TRAINING_STATUS_UPDATE_INTERVAL_MS = 200
const MISSING_VALUE_LABEL = "—"

type ResolvedTheme = "dark" | "light"
type CheckpointSortKey = "saved_at" | "accuracy" | "step" | "elapsed"
type SortDirection = "asc" | "desc"
type CheckpointOptimizerFilter = ExperimentConfig["optimizer"] | "all"
type CheckpointDatasetFilter = ExperimentConfig["dataset"] | "all"
type AppTab = "training" | "analysis"
type AnalysisSide = "left" | "right"
type AnalysisSideLabels = Record<AnalysisSide, string>

type PlotPalette = {
  accuracy: string
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

type PlotLegendItem = {
  color: string
  label: string
}

const fallbackPlotPalettes: Record<ResolvedTheme, PlotPalette> = {
  light: {
    accuracy: "#2563eb",
    grid: "rgba(23, 23, 23, 0.045)",
    hoverBackground: "#ffffff",
    hoverBorder: "#e5e5e5",
    hoverText: "#171717",
    loss: "#171717",
    mutationStep: "#737373",
    muted: "rgba(82, 82, 82, 0.62)",
    text: "#171717",
  },
  dark: {
    accuracy: "#38bdf8",
    grid: "rgba(250, 250, 250, 0.05)",
    hoverBackground: "#262626",
    hoverBorder: "#404040",
    hoverText: "#fafafa",
    loss: "#fafafa",
    mutationStep: "#a3a3a3",
    muted: "rgba(229, 229, 229, 0.52)",
    text: "#e5e5e5",
  },
}

const LONG_CHECKPOINT_DELETE_SECONDS = 600
const ANALYSIS_SIDES: AnalysisSide[] = ["left", "right"]
const EMBEDDING_CLASS_COLORS = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#be123c",
  "#65a30d",
  "#c026d3",
  "#4f46e5",
]
const DEFAULT_ANALYSIS_PARAMS: AnalysisComparisonParams = {
  tsne_perplexity: 30,
  tsne_max_iter: 1000,
  tsne_learning_rate_mode: "auto",
  tsne_learning_rate: null,
  tsne_angle: 0.5,
  tsne_pca_components: 50,
  tsne_seed: null,
  calibration_bins: 15,
  lrp_gallery_sample_count: 24,
  robustness_noise_levels: [0.0, 0.05, 0.1, 0.2],
  robustness_brightness_levels: [0.0, 0.1, 0.2],
  robustness_cutout_levels: [0.0, 0.125, 0.25],
}

const CIFAR10_CLASS_LABELS = [
  "airplane",
  "automobile",
  "bird",
  "cat",
  "deer",
  "dog",
  "frog",
  "horse",
  "ship",
  "truck",
]

export function App() {
  const { theme } = useTheme()
  const resolvedTheme = useResolvedTheme(theme)
  const plotPalette = usePlotPalette(resolvedTheme)
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
  const [activeTab, setActiveTab] = useState<AppTab>("training")
  const [analysisCheckpoints, setAnalysisCheckpoints] = useState<
    Record<AnalysisSide, CheckpointSummary | null>
  >({ left: null, right: null })
  const [analysisJob, setAnalysisJob] =
    useState<AnalysisComparisonJobStatus | null>(null)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [analysisStarting, setAnalysisStarting] = useState(false)

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
    let pendingStepStatus: ExperimentStatus | null = null
    let pendingStepTimer: number | null = null
    let lastStatusUpdateAt = 0

    const statusFromEvent = (event: MessageEvent<string>) => {
      const payload = JSON.parse(event.data) as
        | ExperimentStatus
        | { status: ExperimentStatus }
      return "status" in payload ? payload.status : payload
    }

    const clearPendingStep = () => {
      if (pendingStepTimer !== null) {
        window.clearTimeout(pendingStepTimer)
      }
      pendingStepTimer = null
      pendingStepStatus = null
    }

    const commitStatus = (nextStatus: ExperimentStatus) => {
      lastStatusUpdateAt = performance.now()
      setStatus(nextStatus)
    }

    const commitPendingStep = () => {
      if (pendingStepStatus !== null) {
        commitStatus(pendingStepStatus)
      }
      pendingStepTimer = null
      pendingStepStatus = null
    }

    const handleStepEvent = (event: MessageEvent<string>) => {
      const nextStatus = statusFromEvent(event)
      const elapsed = performance.now() - lastStatusUpdateAt
      if (elapsed >= TRAINING_STATUS_UPDATE_INTERVAL_MS) {
        clearPendingStep()
        commitStatus(nextStatus)
        return
      }

      pendingStepStatus = nextStatus
      if (pendingStepTimer === null) {
        pendingStepTimer = window.setTimeout(
          commitPendingStep,
          TRAINING_STATUS_UPDATE_INTERVAL_MS - elapsed
        )
      }
    }

    const handleEvent = (event: MessageEvent<string>) => {
      clearPendingStep()
      commitStatus(statusFromEvent(event))
    }

    eventTypes.forEach((eventType) => {
      source.addEventListener(
        eventType,
        eventType === "step" ? handleStepEvent : handleEvent
      )
    })

    return () => {
      eventTypes.forEach((eventType) => {
        source.removeEventListener(
          eventType,
          eventType === "step" ? handleStepEvent : handleEvent
        )
      })
      clearPendingStep()
      source.close()
    }
  }, [])

  const analysisJobId = analysisJob?.job_id ?? null
  const analysisJobStatus = analysisJob?.status ?? null

  useEffect(() => {
    if (
      analysisJobId === null ||
      analysisJobStatus === "completed" ||
      analysisJobStatus === "failed"
    ) {
      return
    }

    const source = new EventSource(analysisComparisonEventsUrl(analysisJobId))
    const handleEvent = (event: MessageEvent<string>) => {
      const payload = JSON.parse(event.data) as AnalysisComparisonJobStatus
      setAnalysisJob(payload)
    }

    for (const eventType of ["status", "queued", "running", "completed", "failed"]) {
      source.addEventListener(eventType, handleEvent)
    }

    return () => {
      for (const eventType of [
        "status",
        "queued",
        "running",
        "completed",
        "failed",
      ]) {
        source.removeEventListener(eventType, handleEvent)
      }
      source.close()
    }
  }, [analysisJobId, analysisJobStatus])

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
      fixedPrecisionTickValues(
        mutationStepAxisUpperBound,
        MUTATION_STEP_AXIS_SEGMENTS,
        MUTATION_STEP_TICK_DECIMALS
      ),
    [mutationStepAxisUpperBound]
  )
  const mutationStepTickText = useMemo(
    () => fixedTickText(mutationStepTickValues, MUTATION_STEP_TICK_DECIMALS),
    [mutationStepTickValues]
  )
  const comparisonError = useMemo(
    () => comparisonSelectionError(analysisCheckpoints),
    [analysisCheckpoints]
  )
  const currentAnalysisReport =
    analysisJob?.status === "completed" ? analysisJob.report ?? null : null
  const analysisBusy =
    analysisStarting ||
    analysisJob?.status === "queued" ||
    analysisJob?.status === "running"

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
      uirevision: `${PLOT_UI_REVISION}-${stepAxisUpperBound}-${lossAxisUpperBound}-${mutationStepAxisUpperBound}`,
      margin: { l: 52, r: showMutationStepAxis ? 76 : 46, t: 42, b: 48 },
      xaxis: {
        color: plotPalette.muted,
        ...(showMutationStepAxis
          ? { domain: [0, MUTATION_STEP_PLOT_DOMAIN_END] }
          : {}),
        automargin: true,
        tickfont: { color: plotPalette.muted },
        ticks: "",
        title: { text: "Step", font: { color: plotPalette.muted } },
        range: [0, stepAxisUpperBound],
        tickmode: "auto",
        nticks: 12,
        fixedrange: true,
        showgrid: false,
        showline: false,
        zeroline: false,
      },
      yaxis: {
        color: plotPalette.muted,
        automargin: true,
        tickfont: { color: plotPalette.muted },
        ticks: "",
        title: {
          text: "Loss",
          font: { color: plotPalette.muted },
          standoff: 18,
        },
        range: [0, lossAxisUpperBound],
        tickmode: "array",
        tickvals: lossTickValues,
        tickformat: ".1f",
        fixedrange: true,
        gridcolor: plotPalette.grid,
        gridwidth: 1,
        showgrid: true,
        showline: false,
        mirror: false,
        zeroline: false,
      },
      yaxis2: {
        color: plotPalette.muted,
        automargin: true,
        tickfont: { color: plotPalette.muted },
        ticks: "",
        title: {
          text: "Accuracy (%)",
          font: { color: plotPalette.muted },
          standoff: 14,
        },
        range: [0, 100],
        tick0: 0,
        dtick: 100 / Y_AXIS_SEGMENTS,
        fixedrange: true,
        overlaying: "y",
        side: "right",
        showgrid: false,
        showline: false,
        mirror: false,
        zeroline: false,
      },
      ...(showMutationStepAxis
        ? {
            yaxis3: {
              color: plotPalette.muted,
              automargin: true,
              tickfont: { color: plotPalette.muted },
              ticks: "",
              title: {
                text: "Mutation step",
                font: { color: plotPalette.muted },
                standoff: 14,
              },
              range: [0, mutationStepAxisUpperBound],
              tickmode: "array",
              tickvals: mutationStepTickValues,
              ticktext: mutationStepTickText,
              fixedrange: true,
              overlaying: "y",
              side: "right",
              anchor: "free",
              position: 1,
              showgrid: false,
              showline: false,
              mirror: false,
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

  function updateOptimizerParam(key: string, value: OptimizerParamValue) {
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

  async function loadAnalysisCheckpoint(
    side: AnalysisSide,
    checkpoint: CheckpointSummary
  ) {
    setAnalysisCheckpoints((current) => ({ ...current, [side]: checkpoint }))
    setAnalysisJob(null)
    setAnalysisError(null)
  }

  async function runAnalysisComparison(forceRecompute = false) {
    const left = selectionFromCheckpoint(analysisCheckpoints.left)
    const right = selectionFromCheckpoint(analysisCheckpoints.right)
    const validationError =
      validateAnalysisParams(DEFAULT_ANALYSIS_PARAMS) ??
      comparisonError ??
      "Select two checkpoints"
    if (!left || !right || comparisonError) {
      setAnalysisError(validationError)
      return
    }

    setAnalysisStarting(true)
    setAnalysisError(null)
    try {
      const job = await createAnalysisComparisonJob(
        left,
        right,
        analysisRequestParams(DEFAULT_ANALYSIS_PARAMS),
        forceRecompute
      )
      setAnalysisJob(job)
      if (job.status !== "completed" && job.status !== "failed") {
        fetchAnalysisComparisonJob(job.job_id)
          .then(setAnalysisJob)
          .catch(() => undefined)
      }
    } catch {
      setAnalysisError("Failed to start comparison")
    } finally {
      setAnalysisStarting(false)
    }
  }

  async function newExperiment() {
    const nextStatus = await resetExperimentStatus()
    setLoadedCheckpoint(null)
    setConfig(readStoredConfig(schema))
    setOptParams(readStoredOptimizerParams(schema))
    setStatus(nextStatus)
  }

  function newAnalysis() {
    setAnalysisCheckpoints({ left: null, right: null })
    setAnalysisJob(null)
    setAnalysisError(null)
    setAnalysisStarting(false)
  }

  return (
    <Tabs
      className="h-dvh min-h-0 gap-0 overflow-hidden p-4"
      value={activeTab}
      onValueChange={(value) => setActiveTab(value as AppTab)}
    >
      <div className="mb-4 grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2">
        <div className="flex min-w-0 items-center gap-2">
          {activeTab === "training" ? (
            <>
              <CheckpointPicker
                currentSelection={currentCheckpointSelection}
                disabled={isRunning}
                mode="training"
                pausedRunId={isPaused ? status.run_id : null}
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
            </>
          ) : null}
          {activeTab === "analysis" && currentAnalysisReport ? (
            <Button variant="outline" onClick={newAnalysis}>
              <Plus className="size-4" />
              New analysis
            </Button>
          ) : null}
        </div>
        <TabsList>
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>
        <div className="flex justify-end">
          <ThemeToggle resolvedTheme={resolvedTheme} />
        </div>
      </div>
      <TabsContent
        className="mt-0 min-h-0 overflow-y-auto md:flex md:overflow-hidden"
        value="training"
      >
        <div className="flex w-full flex-col gap-6 md:min-h-0 md:flex-1 md:flex-row">
          <div className="flex w-full max-w-3xl flex-col gap-3 md:max-h-full md:max-w-md">
            <Card className="w-full md:h-fit md:max-h-full">
              <CardHeader>
                <CardTitle className="text-xl">Configuration</CardTitle>
              </CardHeader>
              <CardContent className="min-h-0 md:flex md:flex-col md:overflow-hidden">
                <ScrollArea className="h-auto md:min-h-0 md:max-h-full md:pr-3">
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
                    <h4 className="font-medium">
                      {config.optimizer} parameters
                    </h4>
                    <div className="mt-2 grid grid-cols-[max-content_auto_1fr] items-center gap-x-3 gap-y-2">
                      {activeOptimizerSchema.map((param) => {
                        const value =
                          optParams[config.optimizer]?.[param.key] ??
                          param.default
                        const inputId = `optimizer-${config.optimizer}-${param.key}`
                        const helpText = OPTIMIZER_PARAM_HELP[param.key]

                        return (
                          <div className="contents" key={param.key}>
                            <span className="text-lg">
                              <MathLabel math={param.label} />
                            </span>
                            {param.type === "boolean" ? (
                              <div className="flex h-8 w-24 items-center">
                                <Checkbox
                                  checked={Boolean(value)}
                                  disabled={controlsDisabled}
                                  id={inputId}
                                  onCheckedChange={(checked) =>
                                    updateOptimizerParam(param.key, Boolean(checked))
                                  }
                                />
                              </div>
                            ) : (
                              <Input
                                className="h-8 w-24"
                                type="number"
                                step={param.step ?? 1}
                                disabled={controlsDisabled}
                                id={inputId}
                                value={
                                  typeof value === "number" ? value : ""
                                }
                                onChange={(event) =>
                                  updateOptimizerParam(
                                    param.key,
                                    Number(event.currentTarget.value)
                                  )
                                }
                              />
                            )}
                            <div className="flex min-w-0 items-center gap-1.5">
                              <Label
                                className="min-w-0 text-sm font-normal text-muted-foreground"
                                htmlFor={inputId}
                              >
                                {param.desc}
                              </Label>
                              {helpText ? (
                                <HoverCard openDelay={150} closeDelay={100}>
                                  <HoverCardTrigger asChild>
                                    <Button
                                      aria-label={`About ${param.desc}`}
                                      className="size-5 text-muted-foreground/60 hover:text-muted-foreground"
                                      size="icon-xs"
                                      type="button"
                                      variant="ghost"
                                    >
                                      <CircleHelp className="size-3.5" />
                                    </Button>
                                  </HoverCardTrigger>
                                  <HoverCardContent
                                    align="start"
                                    className="w-72 text-center leading-relaxed text-muted-foreground"
                                  >
                                    {helpText}
                                  </HoverCardContent>
                                </HoverCard>
                              ) : null}
                            </div>
                          </div>
                        )
                      })}
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
                        <Play className="size-4" />
                        Resume
                      </Button>
                    ) : null}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>

          <Card className="min-h-[520px] w-full md:min-h-0 md:flex-1">
            <CardHeader>
              <CardTitle className="text-xl">Metrics</CardTitle>
            </CardHeader>
            <CardContent className="flex min-h-0 flex-1 flex-col">
              <div
                className={cn(
                  "grid gap-x-4 gap-y-5 rounded-lg sm:grid-cols-3",
                  showMutationStepAxis ? "xl:grid-cols-6" : "xl:grid-cols-5"
                )}
              >
                <Metric
                  label="Step"
                  value={formatInteger(status.current_step)}
                />
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

              <div className="mt-6 min-h-[320px] w-full flex-1">
                <Plot
                  className="h-full w-full"
                  data={plotData}
                  layout={plotLayout}
                  config={{
                    displayModeBar: false,
                    displaylogo: false,
                    scrollZoom: false,
                    responsive: true,
                  }}
                  useResizeHandler
                  style={{ width: "100%", height: "100%" }}
                />
              </div>
            </CardContent>
          </Card>
        </div>
      </TabsContent>
      <TabsContent
        className="mt-0 flex min-h-0 flex-1 overflow-hidden"
        value="analysis"
      >
        <AnalysisTab
          busy={analysisBusy}
          checkpoints={analysisCheckpoints}
          comparisonError={comparisonError}
          error={analysisError}
          job={analysisJob}
          pausedRunId={isPaused ? status.run_id : null}
          plotPalette={plotPalette}
          report={currentAnalysisReport}
          schema={schema}
          onLoadCheckpoint={loadAnalysisCheckpoint}
          onRun={runAnalysisComparison}
        />
      </TabsContent>
    </Tabs>
  )
}

type AnalysisTabProps = {
  busy: boolean
  checkpoints: Record<AnalysisSide, CheckpointSummary | null>
  comparisonError: string | null
  error: string | null
  job: AnalysisComparisonJobStatus | null
  pausedRunId: string | null | undefined
  plotPalette: PlotPalette
  report: AnalysisComparisonReport | null
  schema: SchemaResponse
  onLoadCheckpoint: (
    side: AnalysisSide,
    checkpoint: CheckpointSummary
  ) => Promise<void>
  onRun: (forceRecompute?: boolean) => void
}

function AnalysisTab({
  busy,
  checkpoints,
  comparisonError,
  error,
  job,
  pausedRunId,
  plotPalette,
  report,
  schema,
  onLoadCheckpoint,
  onRun,
}: AnalysisTabProps) {
  const scrollRootRef = useRef<HTMLDivElement | null>(null)
  const canRun =
    checkpoints.left !== null &&
    checkpoints.right !== null &&
    comparisonError === null &&
    !busy
  const showSetup = report === null
  const reportSideLabels = report ? analysisSideLabels(report) : undefined

  return (
    <div
      className={cn(
        "flex min-h-0 flex-1 flex-col gap-4",
        report ? "overflow-hidden" : "overflow-y-auto pr-1"
      )}
    >
      {showSetup ? (
        <div className="grid gap-4 md:grid-cols-2">
          {ANALYSIS_SIDES.map((side) => (
            <AnalysisModelSelector
              checkpoint={checkpoints[side]}
              key={side}
              pausedRunId={pausedRunId}
              schema={schema}
              side={side}
              onLoadCheckpoint={onLoadCheckpoint}
            />
          ))}
          <div className="flex flex-col justify-between gap-3 rounded-lg border p-4 md:col-span-2 xl:flex-row xl:items-center">
            <div className="text-sm text-muted-foreground">
              Reports use the full test set with cached defaults for metrics,
              embeddings, LRP, activations, weights, and robustness.
            </div>
            <Button
              className="w-full xl:w-auto"
              disabled={!canRun}
              onClick={() => onRun(false)}
            >
              {busy ? <LoaderCircle className="size-4 animate-spin" /> : null}
              Generate Report
            </Button>
          </div>
        </div>
      ) : null}

      {error ?? comparisonError ? (
        <Alert variant="destructive">
          <AlertTriangle className="size-4" />
          <AlertTitle>Comparison unavailable</AlertTitle>
          <AlertDescription>{error ?? comparisonError}</AlertDescription>
        </Alert>
      ) : null}

      {job?.cache_state === "stale" && report && job.stale_sides.length > 0 ? (
        <Alert>
          <AlertTriangle className="size-4" />
          <AlertTitle>Cached report is stale</AlertTitle>
          <AlertDescription>
            {`Changed checkpoint side: ${job.stale_sides
              .map((side) => sideLabel(side, reportSideLabels))
              .join(", ")}.`}
          </AlertDescription>
          <AlertAction>
            <Button
              className="h-7"
              disabled={busy}
              size="sm"
              onClick={() => onRun(true)}
            >
              Recompute
            </Button>
          </AlertAction>
        </Alert>
      ) : null}

      {busy ? <AnalysisProgress job={job} /> : null}
      {job?.status === "failed" ? (
        <Alert variant="destructive">
          <AlertTriangle className="size-4" />
          <AlertTitle>Comparison failed</AlertTitle>
          <AlertDescription>{job.error ?? job.message}</AlertDescription>
        </Alert>
      ) : null}

      {report ? (
        <ScrollArea
          className="min-h-0 flex-1 pr-1"
          viewportRef={scrollRootRef}
        >
          <AnalysisReport
            plotPalette={plotPalette}
            report={report}
            scrollRootRef={scrollRootRef}
          />
        </ScrollArea>
      ) : !busy ? (
        <Empty className="min-h-96 border">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <FolderOpen className="size-6" />
            </EmptyMedia>
            <EmptyTitle>Select two checkpoints</EmptyTitle>
            <EmptyDescription>
              Reports are generated for two checkpoints from the same dataset.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : null}
    </div>
  )
}

function AnalysisModelSelector({
  checkpoint,
  pausedRunId,
  schema,
  side,
  onLoadCheckpoint,
}: {
  checkpoint: CheckpointSummary | null
  pausedRunId: string | null | undefined
  schema: SchemaResponse
  side: AnalysisSide
  onLoadCheckpoint: (
    side: AnalysisSide,
    checkpoint: CheckpointSummary
  ) => Promise<void>
}) {
  return (
    <Card>
      <CardHeader className="gap-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="text-base">{sideLabel(side)}</CardTitle>
            <div className="mt-1 text-xs text-muted-foreground">
              {checkpoint
                ? `${checkpoint.optimizer} · ${formatDatasetName(checkpoint.dataset)}`
                : "Required"}
            </div>
          </div>
          <CheckpointPicker
            closeOnLoadStart
            currentSelection={selectionFromCheckpoint(checkpoint)}
            disabled={false}
            mode="analysis"
            pausedRunId={pausedRunId}
            schema={schema}
            onLoad={(nextCheckpoint) => onLoadCheckpoint(side, nextCheckpoint)}
          />
        </div>
      </CardHeader>
      <CardContent>
        {checkpoint ? (
          <div className="grid grid-cols-3 gap-3">
            <CheckpointMetric
              label="Accuracy"
              value={formatOptionalPercent(checkpoint.accuracy)}
            />
            <CheckpointMetric label="Step" value={formatInteger(checkpoint.step)} />
            <CheckpointMetric
              label="Elapsed"
              value={formatOptionalDuration(checkpoint.total_elapsed_seconds)}
            />
          </div>
        ) : (
          <div className="grid h-24 place-items-center rounded-lg border border-dashed text-sm text-muted-foreground">
            No checkpoint selected
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function AnalysisProgress({ job }: { job: AnalysisComparisonJobStatus | null }) {
  const progress = Math.round((job?.progress ?? 0) * 100)
  return (
    <Card>
      <CardContent>
        <div className="flex items-center justify-between gap-4 text-sm">
          <div className="min-w-0">
            <div className="font-medium">{job?.stage ?? "load/cache"}</div>
            <div className="truncate text-muted-foreground">
              {job?.message ?? "Starting comparison."}
            </div>
          </div>
          <div className="shrink-0 tabular-nums text-muted-foreground">
            {progress}%
          </div>
        </div>
        <Progress className="mt-3" value={progress} />
        <div className="mt-4 grid gap-2 md:grid-cols-3">
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
        </div>
      </CardContent>
    </Card>
  )
}

const ANALYSIS_REPORT_SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "metrics", label: "Metrics" },
  { id: "embeddings", label: "Embeddings" },
  { id: "lrp", label: "LRP" },
  { id: "robustness", label: "Robustness" },
] as const

type AnalysisReportSectionId = (typeof ANALYSIS_REPORT_SECTIONS)[number]["id"]

function AnalysisReport({
  plotPalette,
  report,
  scrollRootRef,
}: {
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  scrollRootRef: RefObject<HTMLDivElement | null>
}) {
  const [activeSection, setActiveSection] =
    useState<AnalysisReportSectionId>("overview")
  const sideLabels = analysisSideLabels(report)
  const sectionRefs = useRef<Record<AnalysisReportSectionId, HTMLElement | null>>({
    overview: null,
    metrics: null,
    embeddings: null,
    lrp: null,
    robustness: null,
  })

  useEffect(() => {
    let frame = 0
    const scrollRoot = scrollRootRef.current
    const updateActiveSection = () => {
      frame = 0
      const activationLine = scrollRoot
        ? scrollRoot.getBoundingClientRect().top +
          Math.min(scrollRoot.clientHeight * 0.35, 240)
        : Math.min(window.innerHeight * 0.35, 240)
      let nextActive: AnalysisReportSectionId = "overview"

      for (const section of ANALYSIS_REPORT_SECTIONS) {
        const element = sectionRefs.current[section.id]
        if (!element) {
          continue
        }
        if (element.getBoundingClientRect().top <= activationLine) {
          nextActive = section.id
        }
      }

      setActiveSection((current) =>
        current === nextActive ? current : nextActive
      )
    }
    const requestUpdate = () => {
      if (frame !== 0) {
        return
      }
      frame = window.requestAnimationFrame(updateActiveSection)
    }

    updateActiveSection()
    const scrollTarget: Window | HTMLDivElement = scrollRoot ?? window
    scrollTarget.addEventListener("scroll", requestUpdate, { passive: true })
    window.addEventListener("resize", requestUpdate)

    return () => {
      if (frame !== 0) {
        window.cancelAnimationFrame(frame)
      }
      scrollTarget.removeEventListener("scroll", requestUpdate)
      window.removeEventListener("resize", requestUpdate)
    }
  }, [report, scrollRootRef])

  function selectSection(sectionId: AnalysisReportSectionId) {
    setActiveSection(sectionId)
    sectionRefs.current[sectionId]?.scrollIntoView({
      block: "start",
      behavior: "smooth",
    })
  }

  return (
    <div className="grid min-h-0 flex-1 gap-4 pb-6 lg:grid-cols-[10rem_minmax(0,1fr)] xl:grid-cols-[12rem_minmax(0,1fr)]">
      <AnalysisReportToc
        activeSection={activeSection}
        onSelect={selectSection}
      />
      <div className="grid min-w-0 gap-8">
        <AnalysisReportSection
          id="overview"
          sectionRef={(element) => {
            sectionRefs.current.overview = element
          }}
          title="Overview"
        >
          <AnalysisOverview
            plotPalette={plotPalette}
            report={report}
            sideLabels={sideLabels}
          />
        </AnalysisReportSection>
        <AnalysisReportSection
          id="metrics"
          sectionRef={(element) => {
            sectionRefs.current.metrics = element
          }}
          title="Metrics"
        >
          <AnalysisMetrics
            plotPalette={plotPalette}
            report={report}
            sideLabels={sideLabels}
          />
        </AnalysisReportSection>
        <AnalysisReportSection
          id="embeddings"
          sectionRef={(element) => {
            sectionRefs.current.embeddings = element
          }}
          title="Embeddings"
        >
          <AnalysisEmbeddingsView
            plotPalette={plotPalette}
            report={report}
            sideLabels={sideLabels}
          />
        </AnalysisReportSection>
        <AnalysisReportSection
          id="lrp"
          sectionRef={(element) => {
            sectionRefs.current.lrp = element
          }}
          title="LRP"
        >
          <AnalysisLrpView report={report} sideLabels={sideLabels} />
        </AnalysisReportSection>
        <AnalysisReportSection
          id="robustness"
          sectionRef={(element) => {
            sectionRefs.current.robustness = element
          }}
          title="Robustness"
        >
          <AnalysisRobustnessView
            plotPalette={plotPalette}
            report={report}
            sideLabels={sideLabels}
          />
        </AnalysisReportSection>
      </div>
    </div>
  )
}

function AnalysisReportToc({
  activeSection,
  onSelect,
}: {
  activeSection: AnalysisReportSectionId
  onSelect: (section: AnalysisReportSectionId) => void
}) {
  return (
    <aside className="lg:sticky lg:top-4 lg:self-start">
      <nav aria-label="Analysis report sections" className="py-1">
        <div className="relative grid gap-1">
          <div
            aria-hidden="true"
            className="absolute top-3 bottom-3 left-1.5 border-l border-dashed border-muted-foreground/40"
          />
          {ANALYSIS_REPORT_SECTIONS.map((section) => {
            const isActive = activeSection === section.id
            return (
              <a
                aria-current={isActive ? "location" : undefined}
                className={cn(
                  "relative grid grid-cols-[0.75rem_minmax(0,1fr)] items-center gap-3 py-1 text-sm",
                  isActive
                    ? "font-medium text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                href={`#analysis-${section.id}`}
                key={section.id}
                onClick={(event) => {
                  event.preventDefault()
                  onSelect(section.id)
                }}
              >
                <span
                  aria-hidden="true"
                  className={cn(
                    "z-10 size-3 rounded-full border bg-background",
                    isActive
                      ? "border-foreground bg-foreground"
                      : "border-muted-foreground"
                  )}
                />
                {section.label}
              </a>
            )
          })}
        </div>
      </nav>
    </aside>
  )
}

function AnalysisReportSection({
  children,
  id,
  sectionRef,
  title,
}: {
  children: ReactElement
  id: AnalysisReportSectionId
  sectionRef: (element: HTMLElement | null) => void
  title: string
}) {
  return (
    <section
      className="grid scroll-mt-4 gap-3"
      id={`analysis-${id}`}
      ref={sectionRef}
    >
      <h2 className="text-lg font-semibold tracking-normal">{title}</h2>
      {children}
    </section>
  )
}

function AnalysisOverview({
  plotPalette,
  report,
  sideLabels,
}: {
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  return (
    <div className="grid gap-4">
      <div className="grid gap-3 md:grid-cols-4">
        <ReportMetric
          label={`${sideLabel("left", sideLabels)} accuracy`}
          value={formatOptionalPercent(report.metrics.left.accuracy)}
        />
        <ReportMetric
          label={`${sideLabel("right", sideLabels)} accuracy`}
          value={formatOptionalPercent(report.metrics.right.accuracy)}
        />
        <ReportMetric
          label="Disagreements"
          value={formatInteger(report.overlap.disagreements)}
        />
        <ReportMetric
          label="Device"
          value={report.analysis_device.toUpperCase()}
        />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <AnalysisRowsTable
          rows={report.metadata}
          sideLabels={sideLabels}
          title="Checkpoints"
        />
        <AnalysisRowsTable
          rows={report.runtime.rows.filter((row) => row.label !== "Steps")}
          sideLabels={sideLabels}
          title="Runtime"
        />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <ReportPlot
          data={trainingCurveData(report, sideLabels)}
          layout={plotLayout("Training History", plotPalette)}
        />
        <ReportPlot
          data={overlapData(report, sideLabels)}
          layout={plotLayout("Outcome Overlap", plotPalette)}
        />
      </div>
    </div>
  )
}

function AnalysisMetrics({
  plotPalette,
  report,
  sideLabels,
}: {
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  const confusionDeltaLabel = `${sideShortLabel(
    "left",
    sideLabels
  )} - ${sideShortLabel("right", sideLabels)}`

  return (
    <div className="grid gap-4">
      <div className="grid gap-4 xl:grid-cols-3">
        <ReportPlot
          data={confusionData(
            report.metrics.left.confusion_matrix,
            sideLabel("left", sideLabels)
          )}
          layout={heatmapLayout(
            `${sideLabel("left", sideLabels)} Confusion`,
            plotPalette
          )}
        />
        <ReportPlot
          data={confusionData(
            report.metrics.right.confusion_matrix,
            sideLabel("right", sideLabels)
          )}
          layout={heatmapLayout(
            `${sideLabel("right", sideLabels)} Confusion`,
            plotPalette
          )}
        />
        <ReportPlot
          data={confusionData(report.confusion_difference, confusionDeltaLabel)}
          layout={heatmapLayout(
            `Confusion Delta (${confusionDeltaLabel})`,
            plotPalette
          )}
        />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <ReportPlot
          data={calibrationData(report, sideLabels)}
          layout={plotLayout("Calibration", plotPalette)}
        />
        <PerClassMetricTable report={report} sideLabels={sideLabels} />
      </div>
    </div>
  )
}

function AnalysisEmbeddingsView({
  plotPalette,
  report,
  sideLabels,
}: {
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  const tsneLeftData = embeddingSideData(
    report.embeddings.tsne.left,
    "left",
    report.left.dataset,
    sideLabels
  )
  const tsneRightData = embeddingSideData(
    report.embeddings.tsne.right,
    "right",
    report.right.dataset,
    sideLabels
  )
  const pcaData = embeddingData(
    report.embeddings.pca,
    report.left.dataset,
    sideLabels
  )

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <ReportPlot
        className="xl:aspect-[5/3] xl:h-auto"
        data={tsneLeftData}
        layout={embeddingLayout(
          `t-SNE ${sideLabel("left", sideLabels)}`,
          plotPalette
        )}
        legendItems={embeddingLegendItems(
          report.embeddings.tsne.left,
          report.left.dataset
        )}
      />
      <ReportPlot
        className="xl:aspect-[5/3] xl:h-auto"
        data={tsneRightData}
        layout={embeddingLayout(
          `t-SNE ${sideLabel("right", sideLabels)}`,
          plotPalette
        )}
        legendItems={embeddingLegendItems(
          report.embeddings.tsne.right,
          report.right.dataset
        )}
      />
      <div className="xl:col-span-2">
        <ReportPlot
          data={pcaData}
          layout={embeddingLayout("Joint PCA", plotPalette)}
          legendItems={embeddingLegendItems(
            [...report.embeddings.pca.left, ...report.embeddings.pca.right],
            report.left.dataset
          )}
        />
      </div>
    </div>
  )
}

function AnalysisLrpView({
  report,
  sideLabels,
}: {
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  if (report.lrp.samples.length === 0) {
    return (
      <Empty className="min-h-72 border">
        <EmptyHeader>
          <EmptyTitle>No LRP samples</EmptyTitle>
          <EmptyDescription>The backend returned no gallery items.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-[repeat(auto-fill,minmax(17rem,1fr))] gap-3">
        {report.lrp.samples.map((sample) => (
          <LrpSampleCard
            dataset={report.left.dataset}
            key={sample.index}
            sample={sample}
            sideLabels={sideLabels}
          />
        ))}
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(10rem,1fr))] gap-3">
        {report.lrp.class_averages.map((average) => (
          <div className="rounded-lg border p-2" key={average.label}>
            <div className="mb-2 flex items-center justify-between gap-2 text-sm">
              <span className="font-medium">{average.name}</span>
              <Badge variant="outline">avg</Badge>
            </div>
            <div className="aspect-square overflow-hidden rounded-md border bg-muted">
              <RelevanceCanvas relevance={average.difference_relevance} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function AnalysisRobustnessView({
  plotPalette,
  report,
  sideLabels,
}: {
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  return (
    <div className="grid gap-4">
      <ReportPlot
        data={robustnessData(report, sideLabels)}
        layout={plotLayout("Robustness", plotPalette)}
      />
      <div className="grid gap-4 xl:grid-cols-2">
        <ReportPlot
          data={activationData(report, sideLabels)}
          layout={plotLayout("Activation Sparsity", plotPalette)}
        />
        <ReportPlot
          data={weightData(report)}
          layout={plotLayout("Relative Weight Distance", plotPalette)}
        />
      </div>
    </div>
  )
}

function ReportMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-4">
      <div className="text-lg leading-tight font-medium">{value}</div>
      <div className="mt-1 text-xs text-muted-foreground">{label}</div>
    </div>
  )
}

function AnalysisRowsTable({
  rows,
  sideLabels,
  title,
}: {
  rows: AnalysisTableRow[]
  sideLabels: AnalysisSideLabels
  title: string
}) {
  return (
    <div className="rounded-lg border p-3">
      <div className="mb-2 text-sm font-medium">{title}</div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Field</TableHead>
            <TableHead>{sideLabel("left", sideLabels)}</TableHead>
            <TableHead>{sideLabel("right", sideLabels)}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.label}>
              <TableCell className="font-medium">{row.label}</TableCell>
              <TableCell>{formatAnalysisTableValue(row.label, row.left)}</TableCell>
              <TableCell>{formatAnalysisTableValue(row.label, row.right)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function PerClassMetricTable({
  report,
  sideLabels,
}: {
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  return (
    <div className="rounded-lg border p-3">
      <div className="mb-2 text-sm font-medium">Per-class F1</div>
      <ScrollArea className="h-[24rem]">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Class</TableHead>
              <TableHead>{sideLabel("left", sideLabels)}</TableHead>
              <TableHead>{sideLabel("right", sideLabels)}</TableHead>
              <TableHead>Support</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {report.metrics.left.per_class_f1.map((leftMetric, index) => {
              const rightMetric = report.metrics.right.per_class_f1[index]
              return (
                <TableRow key={leftMetric.label}>
                  <TableCell className="font-medium">{leftMetric.name}</TableCell>
                  <TableCell>{formatCompactNumber(leftMetric.f1)}</TableCell>
                  <TableCell>{formatCompactNumber(rightMetric?.f1 ?? 0)}</TableCell>
                  <TableCell>{formatInteger(leftMetric.support)}</TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </ScrollArea>
    </div>
  )
}

function ReportPlot({
  className,
  data,
  layout,
  legendItems,
}: {
  className?: string
  data: Data[]
  layout: Partial<Layout>
  legendItems?: PlotLegendItem[]
}) {
  const plotLayout = legendItems ? { ...layout, showlegend: false } : layout

  return (
    <div className={cn("flex h-[24rem] flex-col rounded-lg border p-2", className)}>
      <div className="min-h-0 flex-1">
        <Plot
          className="h-full w-full"
          config={{
            displayModeBar: false,
            displaylogo: false,
            responsive: true,
            scrollZoom: false,
          }}
          data={data}
          layout={plotLayout}
          style={{ height: "100%", width: "100%" }}
          useResizeHandler
        />
      </div>
      {legendItems ? (
        <div className="flex shrink-0 flex-wrap items-center justify-center gap-x-5 gap-y-1 pb-1 pt-2 text-xs text-muted-foreground">
          {legendItems.map((item) => (
            <span className="inline-flex items-center gap-2" key={item.label}>
              <span
                className="size-1 rounded-full"
                style={{ backgroundColor: item.color }}
              />
              {item.label}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function LrpSampleCard({
  dataset,
  sample,
  sideLabels,
}: {
  dataset: CheckpointSummary["dataset"]
  sample: AnalysisLrpSample
  sideLabels: AnalysisSideLabels
}) {
  return (
    <div className="rounded-lg border p-3">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium">
            {classLabelFor(dataset, sample.label)}
          </div>
          <div className="truncate text-xs text-muted-foreground">
            #{sample.index} · {overlapSetLabel(sample.group, sideLabels)}
          </div>
        </div>
        <Badge variant="outline">
          {sideShortLabel("left", sideLabels)}{" "}
          {classLabelFor(dataset, sample.left_prediction)} ·{" "}
          {sideShortLabel("right", sideLabels)}{" "}
          {classLabelFor(dataset, sample.right_prediction)}
        </Badge>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <LrpCanvasPanel
          image={sample.image}
          label={sideLabel("left", sideLabels)}
          relevance={sample.left_relevance}
        />
        <LrpCanvasPanel
          image={sample.image}
          label={sideLabel("right", sideLabels)}
          relevance={sample.right_relevance}
        />
        <LrpCanvasPanel
          image={sample.image}
          label="Delta"
          relevance={sample.difference_relevance}
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <CheckpointMetric
          label={`${sideShortLabel("left", sideLabels)} conf.`}
          value={formatOptionalPercent(sample.left_confidence * 100)}
        />
        <CheckpointMetric
          label={`${sideShortLabel("right", sideLabels)} conf.`}
          value={formatOptionalPercent(sample.right_confidence * 100)}
        />
      </div>
    </div>
  )
}

function LrpCanvasPanel({
  image,
  label,
  relevance,
}: {
  image: number[][][]
  label: string
  relevance: number[][]
}) {
  return (
    <div className="min-w-0">
      <div className="aspect-square overflow-hidden rounded-md border bg-muted">
        <RelevanceCanvas image={image} relevance={relevance} />
      </div>
      <div className="mt-1 truncate text-center text-xs text-muted-foreground">
        {label}
      </div>
    </div>
  )
}

function RelevanceCanvas({
  image,
  relevance,
}: {
  image?: number[][][]
  relevance: number[][]
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    drawRelevance(canvas, relevance, image)
  }, [image, relevance])

  return (
    <canvas
      className="block h-full w-full"
      ref={canvasRef}
      style={{ imageRendering: "pixelated" }}
    />
  )
}

function drawRelevance(
  canvas: HTMLCanvasElement,
  relevance: number[][],
  image?: number[][][]
) {
  const height = relevance.length || image?.length || 0
  const width = relevance[0]?.length || image?.[0]?.length || 0
  if (height === 0 || width === 0) {
    return
  }

  canvas.width = width
  canvas.height = height
  const context = canvas.getContext("2d")
  if (!context) {
    return
  }
  context.imageSmoothingEnabled = false
  const imageData = context.createImageData(width, height)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4
      const base = image?.[y]?.[x] ?? [0.5, 0.5, 0.5]
      const value = clamp(relevance[y]?.[x] ?? 0, -1, 1)
      const overlay = value >= 0 ? [220, 38, 38] : [37, 99, 235]
      const alpha = Math.abs(value) * 0.7
      imageData.data[offset] = blendChannel(base[0] ?? 0, overlay[0], alpha)
      imageData.data[offset + 1] = blendChannel(base[1] ?? 0, overlay[1], alpha)
      imageData.data[offset + 2] = blendChannel(base[2] ?? 0, overlay[2], alpha)
      imageData.data[offset + 3] = 255
    }
  }
  context.putImageData(imageData, 0, 0)
}

function blendChannel(base: number, overlay: number, alpha: number): number {
  return Math.round(clamp(base, 0, 1) * 255 * (1 - alpha) + overlay * alpha)
}

type CheckpointPickerProps = {
  closeOnLoadStart?: boolean
  currentSelection: CheckpointSelection | null
  disabled: boolean
  mode?: CheckpointListMode
  pausedRunId: string | null | undefined
  schema: SchemaResponse
  trigger?: ReactElement
  onLoad: (checkpoint: CheckpointSummary) => Promise<void>
}

function CheckpointPicker({
  closeOnLoadStart = false,
  currentSelection,
  disabled,
  mode = "training",
  pausedRunId,
  schema,
  trigger,
  onLoad,
}: CheckpointPickerProps) {
  const [open, setOpen] = useState(false)
  const [checkpoints, setCheckpoints] = useState<CheckpointSummary[]>([])
  const [pendingSelection, setPendingSelection] =
    useState<CheckpointSelection | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sortKey, setSortKey] = useState<CheckpointSortKey>("saved_at")
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc")
  const [optimizerFilter, setOptimizerFilter] =
    useState<CheckpointOptimizerFilter>("all")
  const [datasetFilter, setDatasetFilter] =
    useState<CheckpointDatasetFilter>("all")
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null)
  const [confirmingCheckpoint, setConfirmingCheckpoint] =
    useState<CheckpointSummary | null>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    let ignore = false

    fetchCheckpoints(mode)
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
  }, [mode, open])

  const optimizerOptions = useMemo(
    () =>
      checkpointOptimizerOptions(
        checkpoints,
        schema.config_schema.optimizer.options
      ),
    [checkpoints, schema.config_schema.optimizer.options]
  )
  const datasetOptions = useMemo(
    () =>
      checkpointDatasetOptions(
        checkpoints,
        schema.config_schema.dataset.options
      ),
    [checkpoints, schema.config_schema.dataset.options]
  )
  const visibleCheckpoints = useMemo(
    () =>
      sortCheckpoints(
        checkpoints.filter(
          (checkpoint) =>
            (optimizerFilter === "all" ||
              checkpoint.optimizer === optimizerFilter) &&
            (datasetFilter === "all" || checkpoint.dataset === datasetFilter)
        ),
        sortKey,
        sortDirection
      ),
    [checkpoints, datasetFilter, optimizerFilter, sortDirection, sortKey]
  )
  const pendingCheckpoint = pendingSelection
    ? (visibleCheckpoints.find((checkpoint) =>
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
      if (closeOnLoadStart) {
        setOpen(false)
      }
      await onLoad(pendingCheckpoint)
      if (!closeOnLoadStart) {
        setOpen(false)
      }
    } catch {
      setError("Failed to load checkpoint")
      if (closeOnLoadStart) {
        setOpen(true)
      }
    }
  }

  async function deleteCheckpoint(checkpoint: CheckpointSummary) {
    if (checkpoint.run_id === pausedRunId) {
      setError("Stop or resume the paused experiment before deleting it")
      return
    }

    setDeletingRunId(checkpoint.run_id)
    setError(null)
    try {
      await deleteCheckpointRun(checkpoint.run_id)
      setCheckpoints((current) =>
        current.filter((item) => item.run_id !== checkpoint.run_id)
      )
      setPendingSelection((current) =>
        current?.run_id === checkpoint.run_id ? null : current
      )
      setConfirmingCheckpoint(null)
    } catch {
      setError("Failed to delete checkpoint")
    } finally {
      setDeletingRunId(null)
    }
  }

  function requestDeleteCheckpoint(checkpoint: CheckpointSummary) {
    if (checkpoint.run_id === pausedRunId) {
      setError("Stop or resume the paused experiment before deleting it")
      return
    }

    if (
      (checkpoint.total_elapsed_seconds ?? 0) > LONG_CHECKPOINT_DELETE_SECONDS
    ) {
      setConfirmingCheckpoint(checkpoint)
      return
    }

    void deleteCheckpoint(checkpoint)
  }

  function toggleSortDirection() {
    setSortDirection((current) => (current === "desc" ? "asc" : "desc"))
  }

  function handleCheckpointRowKeyDown(
    event: KeyboardEvent<HTMLDivElement>,
    checkpoint: CheckpointSummary
  ) {
    if (event.target !== event.currentTarget) {
      return
    }

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault()
      togglePendingSelection(checkpoint)
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

  const hasCheckpoints = checkpoints.length > 0
  const hasVisibleCheckpoints = visibleCheckpoints.length > 0
  const deleteConfirmationName = confirmingCheckpoint
    ? checkpointDisplayName(confirmingCheckpoint)
    : "this checkpoint"

  return (
    <>
      <Dialog open={open} onOpenChange={handleOpenChange}>
        <DialogTrigger asChild>
          {trigger ?? (
            <Button
              className="max-w-[18rem] justify-start"
              disabled={disabled}
              variant="outline"
            >
              <FolderOpen className="size-4" />
              <span className="truncate">{buttonLabel}</span>
            </Button>
          )}
        </DialogTrigger>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Checkpoints</DialogTitle>
            <DialogDescription className="sr-only">
              Select a checkpoint to load into the paused experiment view.
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <div className="flex min-w-0 items-center gap-1.5">
              <ArrowUpDown
                aria-hidden="true"
                className="size-4 text-muted-foreground"
              />
              <Select
                value={sortKey}
                onValueChange={(value) =>
                  setSortKey(value as CheckpointSortKey)
                }
              >
                <SelectTrigger
                  aria-label="Sort checkpoints by"
                  className="h-8 w-32"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent
                  align="start"
                  className="w-(--radix-select-trigger-width) min-w-(--radix-select-trigger-width)"
                  position="popper"
                >
                  <SelectItem value="saved_at">Saved date</SelectItem>
                  <SelectItem value="accuracy">Accuracy</SelectItem>
                  <SelectItem value="step">Step</SelectItem>
                  <SelectItem value="elapsed">Elapsed time</SelectItem>
                </SelectContent>
              </Select>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    aria-label={`Sort ${sortDirection === "desc" ? "descending" : "ascending"}`}
                    className="size-8 bg-background"
                    size="icon"
                    type="button"
                    variant="outline"
                    onClick={toggleSortDirection}
                  >
                    <ArrowDown
                      className={cn(
                        "size-4 transition-transform",
                        sortDirection === "asc" ? "rotate-180" : "rotate-0"
                      )}
                    />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {sortDirection === "desc" ? "Descending" : "Ascending"}
                </TooltipContent>
              </Tooltip>
            </div>
            <div className="flex min-w-0 items-center gap-1.5">
              <Funnel
                aria-hidden="true"
                className="size-4 text-muted-foreground"
              />
              <Select
                value={optimizerFilter}
                onValueChange={(value) =>
                  setOptimizerFilter(value as CheckpointOptimizerFilter)
                }
              >
                <SelectTrigger
                  aria-label="Filter checkpoints by optimizer"
                  className="h-8 w-32"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent
                  align="start"
                  className="w-(--radix-select-trigger-width) min-w-(--radix-select-trigger-width)"
                  position="popper"
                >
                  <SelectItem value="all">All optimizers</SelectItem>
                  {optimizerOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select
                value={datasetFilter}
                onValueChange={(value) =>
                  setDatasetFilter(value as CheckpointDatasetFilter)
                }
              >
                <SelectTrigger
                  aria-label="Filter checkpoints by dataset"
                  className="h-8 w-32"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent
                  align="start"
                  className="w-(--radix-select-trigger-width) min-w-(--radix-select-trigger-width)"
                  position="popper"
                >
                  <SelectItem value="all">All datasets</SelectItem>
                  {datasetOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {error ? (
            <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
              {error}
            </div>
          ) : null}

          <ScrollArea className="h-[min(60vh,32rem)] pr-3">
            {isLoading ? <CheckpointListSkeleton /> : null}
            {!isLoading && !error && !hasCheckpoints ? (
              <CheckpointEmpty
                title="No checkpoints"
                description="Completed latest checkpoints will appear here."
              />
            ) : null}
            {!isLoading &&
            !error &&
            hasCheckpoints &&
            !hasVisibleCheckpoints ? (
              <CheckpointEmpty
                title="No matching checkpoints"
                description="Change the optimizer or dataset filter."
              />
            ) : null}
            {!isLoading && !error && hasVisibleCheckpoints ? (
              <div className="grid gap-2">
                {visibleCheckpoints.map((checkpoint) => {
                  const selected = sameCheckpoint(checkpoint, pendingSelection)
                  const optimizerParams = optimizerParamEntries(
                    checkpoint,
                    schema
                  )
                  const deleteBlocked = checkpoint.run_id === pausedRunId
                  const isDeleting = deletingRunId === checkpoint.run_id

                  return (
                    <div
                      aria-label={`Select checkpoint for ${checkpoint.optimizer} ${checkpoint.dataset}, seed ${checkpoint.seed}, step ${checkpoint.step}`}
                      aria-pressed={selected}
                      className={cn(
                        "cursor-pointer rounded-lg border p-4 text-left transition-colors hover:bg-muted/60 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none",
                        selected
                          ? "border-primary bg-primary/5"
                          : "border-border bg-background"
                      )}
                      key={`${checkpoint.run_id}-${checkpoint.kind}`}
                      role="button"
                      tabIndex={0}
                      onClick={() => togglePendingSelection(checkpoint)}
                      onKeyDown={(event) =>
                        handleCheckpointRowKeyDown(event, checkpoint)
                      }
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
                        <div className="flex shrink-0 items-center gap-1">
                          <span className="text-sm text-muted-foreground/80">
                            {formatCheckpointDate(checkpoint.saved_at)}
                          </span>
                          <CheckpointDeleteButton
                            blocked={deleteBlocked}
                            deleting={isDeleting}
                            onDelete={() => requestDeleteCheckpoint(checkpoint)}
                          />
                        </div>
                      </div>

                      <div className="mt-4 flex flex-wrap items-start gap-x-12 gap-y-3">
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
                        <div className="mt-5 flex flex-wrap gap-x-7 gap-y-2 text-sm">
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
                    </div>
                  )
                })}
              </div>
            ) : null}
          </ScrollArea>

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

      <AlertDialog
        open={confirmingCheckpoint !== null}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) {
            setConfirmingCheckpoint(null)
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete checkpoint?</AlertDialogTitle>
            <AlertDialogDescription className="text-wrap">
              Delete {deleteConfirmationName}, including any hidden best
              checkpoint for this run.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={
                confirmingCheckpoint !== null &&
                deletingRunId === confirmingCheckpoint.run_id
              }
              variant="destructive"
              onClick={() => {
                if (confirmingCheckpoint) {
                  void deleteCheckpoint(confirmingCheckpoint)
                }
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}

function CheckpointListSkeleton() {
  return (
    <div className="grid gap-2">
      {Array.from({ length: 4 }).map((_, index) => (
        <div className="rounded-lg border p-4" key={index}>
          <div className="flex items-start justify-between gap-4">
            <div className="grid flex-1 gap-2">
              <Skeleton className="h-5 w-44" />
              <Skeleton className="h-4 w-36" />
            </div>
            <Skeleton className="h-4 w-20" />
          </div>
          <div className="mt-4 flex gap-12">
            <Skeleton className="h-10 w-24" />
            <Skeleton className="h-10 w-20" />
            <Skeleton className="h-10 w-24" />
          </div>
        </div>
      ))}
    </div>
  )
}

function CheckpointEmpty({
  title,
  description,
}: {
  title: string
  description: string
}) {
  return (
    <Empty className="min-h-64 border">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <FolderOpen className="size-4" />
        </EmptyMedia>
        <EmptyTitle>{title}</EmptyTitle>
        <EmptyDescription>{description}</EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}

function CheckpointDeleteButton({
  blocked,
  deleting,
  onDelete,
}: {
  blocked: boolean
  deleting: boolean
  onDelete: () => void
}) {
  const label = blocked
    ? "Current paused checkpoint cannot be deleted"
    : "Delete checkpoint"

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className="inline-flex"
          onClick={(event) => event.stopPropagation()}
        >
          <Button
            aria-label={label}
            disabled={blocked || deleting}
            size="icon-sm"
            type="button"
            variant="ghost"
            onClick={(event) => {
              event.stopPropagation()
              onDelete()
            }}
          >
            <Trash2 className="size-4" />
          </Button>
        </span>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
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

function usePlotPalette(resolvedTheme: ResolvedTheme): PlotPalette {
  return useMemo(() => readPlotPalette(resolvedTheme), [resolvedTheme])
}

function readPlotPalette(resolvedTheme: ResolvedTheme): PlotPalette {
  const fallback = fallbackPlotPalettes[resolvedTheme]

  if (typeof document === "undefined") {
    return fallback
  }

  const styles = getComputedStyle(document.documentElement)

  return {
    accuracy: readCssColor(styles, "--plot-accuracy", fallback.accuracy),
    grid: readCssColor(styles, "--plot-grid", fallback.grid),
    hoverBackground: readCssColor(
      styles,
      "--plot-hover-background",
      fallback.hoverBackground
    ),
    hoverBorder: readCssColor(
      styles,
      "--plot-hover-border",
      fallback.hoverBorder
    ),
    hoverText: readCssColor(styles, "--plot-hover-text", fallback.hoverText),
    loss: readCssColor(styles, "--plot-loss", fallback.loss),
    mutationStep: readCssColor(
      styles,
      "--plot-mutation-step",
      fallback.mutationStep
    ),
    muted: readCssColor(styles, "--plot-muted", fallback.muted),
    text: readCssColor(styles, "--plot-text", fallback.text),
  }
}

function readCssColor(
  styles: CSSStyleDeclaration,
  propertyName: string,
  fallback: string
): string {
  return styles.getPropertyValue(propertyName).trim() || fallback
}

function validateAnalysisParams(params: AnalysisComparisonParams): string | null {
  if (!isFiniteInRange(params.tsne_perplexity, 5, 50)) {
    return "Perplexity must be between 5 and 50"
  }
  if (!Number.isFinite(params.tsne_max_iter) || params.tsne_max_iter < 250) {
    return "t-SNE iterations must be at least 250"
  }
  if (!isFiniteInRange(params.tsne_angle, 0.2, 0.8)) {
    return "t-SNE angle must be between 0.2 and 0.8"
  }
  if (
    !Number.isFinite(params.tsne_pca_components) ||
    params.tsne_pca_components < 2 ||
    params.tsne_pca_components > 120
  ) {
    return "PCA dimensions must be between 2 and 120"
  }
  if (
    params.tsne_learning_rate_mode === "numeric" &&
    (!Number.isFinite(params.tsne_learning_rate) ||
      Number(params.tsne_learning_rate) <= 0)
  ) {
    return "Learning rate must be positive"
  }
  if (
    !Number.isInteger(params.calibration_bins) ||
    params.calibration_bins < 5 ||
    params.calibration_bins > 50
  ) {
    return "Calibration bins must be an integer between 5 and 50"
  }
  if (
    !Number.isInteger(params.lrp_gallery_sample_count) ||
    params.lrp_gallery_sample_count < 1 ||
    params.lrp_gallery_sample_count > 60
  ) {
    return "LRP samples must be an integer between 1 and 60"
  }
  return null
}

function comparisonSelectionError(
  checkpoints: Record<AnalysisSide, CheckpointSummary | null>
): string | null {
  if (!checkpoints.left || !checkpoints.right) {
    return null
  }
  if (checkpoints.left.dataset !== checkpoints.right.dataset) {
    const sideLabels = checkpointSideLabels(checkpoints.left, checkpoints.right)
    return `${sideLabel("left", sideLabels)} and ${sideLabel(
      "right",
      sideLabels
    )} must use the same dataset`
  }
  return null
}

function analysisRequestParams(
  params: AnalysisComparisonParams
): AnalysisComparisonParams {
  return {
    ...params,
    tsne_learning_rate:
      params.tsne_learning_rate_mode === "numeric"
        ? params.tsne_learning_rate
        : null,
    tsne_seed: params.tsne_seed ?? null,
  }
}

function isFiniteInRange(value: number, min: number, max: number): boolean {
  return Number.isFinite(value) && value >= min && value <= max
}

function analysisSideLabels(report: AnalysisComparisonReport): AnalysisSideLabels {
  return checkpointSideLabels(report.left, report.right)
}

function checkpointSideLabels(
  left: CheckpointSummary,
  right: CheckpointSummary
): AnalysisSideLabels {
  if (left.optimizer !== right.optimizer) {
    return { left: left.optimizer, right: right.optimizer }
  }
  return { left: "Model A", right: "Model B" }
}

function sideLabel(
  side: AnalysisSide,
  sideLabels?: AnalysisSideLabels
): string {
  return sideLabels?.[side] ?? (side === "left" ? "Model A" : "Model B")
}

function sideShortLabel(
  side: AnalysisSide,
  sideLabels?: AnalysisSideLabels
): string {
  const label = sideLabel(side, sideLabels)
  if (label === "Model A") {
    return "A"
  }
  if (label === "Model B") {
    return "B"
  }
  return label
}

function overlapSetLabel(set: string, sideLabels: AnalysisSideLabels): string {
  switch (set) {
    case "correct_both":
      return "Both correct"
    case "left_only_correct":
      return `${sideLabel("left", sideLabels)} only correct`
    case "right_only_correct":
      return `${sideLabel("right", sideLabels)} only correct`
    case "error_both_same_prediction":
      return "Both incorrect, same prediction"
    case "error_both_different_prediction":
      return "Both incorrect, different predictions"
    case "disagreement":
      return "Disagreement"
    case "disagreements":
      return "Disagreements"
    default:
      return set.replaceAll("_", " ")
  }
}

function trainingCurveData(
  report: AnalysisComparisonReport,
  sideLabels: AnalysisSideLabels
): Data[] {
  return [
    {
      type: "scatter",
      mode: traceMode(report.curves.left.training_loss.length),
      name: `${sideShortLabel("left", sideLabels)} train loss`,
      x: report.curves.left.training_loss.map((_, index) => index + 1),
      y: report.curves.left.training_loss,
      line: { color: "#2563eb", width: 1.5 },
    },
    {
      type: "scatter",
      mode: traceMode(report.curves.right.training_loss.length),
      name: `${sideShortLabel("right", sideLabels)} train loss`,
      x: report.curves.right.training_loss.map((_, index) => index + 1),
      y: report.curves.right.training_loss,
      line: { color: "#dc2626", width: 1.5 },
    },
    {
      type: "scatter",
      mode: accuracyTraceMode(report.curves.left.validation_accuracy.length),
      name: `${sideShortLabel("left", sideLabels)} val acc.`,
      x: report.curves.left.validation_accuracy.map((point) => point.i),
      y: report.curves.left.validation_accuracy.map((point) => point.value),
      yaxis: "y2",
      line: { color: "#0891b2", width: 1.5, dash: "dot" },
    },
    {
      type: "scatter",
      mode: accuracyTraceMode(report.curves.right.validation_accuracy.length),
      name: `${sideShortLabel("right", sideLabels)} val acc.`,
      x: report.curves.right.validation_accuracy.map((point) => point.i),
      y: report.curves.right.validation_accuracy.map((point) => point.value),
      yaxis: "y2",
      line: { color: "#ea580c", width: 1.5, dash: "dot" },
    },
  ]
}

function overlapData(
  report: AnalysisComparisonReport,
  sideLabels: AnalysisSideLabels
): Data[] {
  return [
    {
      type: "bar",
      orientation: "h",
      x: report.overlap.upset.map((row) => row.count),
      y: report.overlap.upset.map((row) =>
        overlapSetLabel(row.set, sideLabels)
      ),
      marker: { color: "#2563eb" },
    },
  ]
}

function confusionData(matrix: number[][], name: string): Data[] {
  return [
    {
      type: "heatmap",
      name,
      z: matrix,
      colorscale: "RdBu",
      reversescale: true,
      showscale: false,
    },
  ]
}

function calibrationData(
  report: AnalysisComparisonReport,
  sideLabels: AnalysisSideLabels
): Data[] {
  return [
    {
      type: "scatter",
      mode: "lines+markers",
      name: sideLabel("left", sideLabels),
      x: report.metrics.left.calibration.bins.map((bin) => bin.confidence),
      y: report.metrics.left.calibration.bins.map((bin) => bin.accuracy),
      line: { color: "#2563eb", width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: sideLabel("right", sideLabels),
      x: report.metrics.right.calibration.bins.map((bin) => bin.confidence),
      y: report.metrics.right.calibration.bins.map((bin) => bin.accuracy),
      line: { color: "#dc2626", width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Ideal",
      x: [0, 1],
      y: [0, 1],
      line: { color: "#737373", width: 1, dash: "dot" },
    },
  ]
}

type AnalysisEmbeddingPlotPoint =
  AnalysisEmbeddingProjection["left"][number] & {
    side: AnalysisSide
  }

function embeddingSideData(
  points: AnalysisEmbeddingProjection["left"],
  side: AnalysisSide,
  dataset: CheckpointSummary["dataset"],
  sideLabels: AnalysisSideLabels
): Data[] {
  return embeddingClassData(
    points.map((point) => ({ ...point, side })),
    dataset,
    sideLabels
  )
}

function embeddingData(
  projection: AnalysisEmbeddingProjection,
  dataset: CheckpointSummary["dataset"],
  sideLabels: AnalysisSideLabels
): Data[] {
  const points: AnalysisEmbeddingPlotPoint[] = [
    ...projection.left.map((point) => ({ ...point, side: "left" as const })),
    ...projection.right.map((point) => ({ ...point, side: "right" as const })),
  ]

  return embeddingClassData(points, dataset, sideLabels)
}

function embeddingClassData(
  points: AnalysisEmbeddingPlotPoint[],
  dataset: CheckpointSummary["dataset"],
  sideLabels: AnalysisSideLabels
): Data[] {
  const grouped = new Map<number, AnalysisEmbeddingPlotPoint[]>()
  for (const point of points) {
    const group = grouped.get(point.label)
    if (group) {
      group.push(point)
    } else {
      grouped.set(point.label, [point])
    }
  }

  return Array.from(grouped.entries())
    .sort(([leftLabel], [rightLabel]) => leftLabel - rightLabel)
    .map(([label, classPoints]) =>
      embeddingTrace(
        classPoints,
        classLabelFor(dataset, label),
        embeddingClassColor(label),
        dataset,
        sideLabels
      )
    )
}

function embeddingLegendItems(
  points: AnalysisEmbeddingProjection["left"],
  dataset: CheckpointSummary["dataset"]
): PlotLegendItem[] {
  return Array.from(new Set(points.map((point) => point.label)))
    .sort((leftLabel, rightLabel) => leftLabel - rightLabel)
    .map((label) => ({
      color: embeddingClassColor(label),
      label: classLabelFor(dataset, label),
    }))
}

function embeddingTrace(
  points: AnalysisEmbeddingPlotPoint[],
  name: string,
  color: string,
  dataset: CheckpointSummary["dataset"],
  sideLabels: AnalysisSideLabels
): Data {
  return {
    type: "scattergl",
    mode: "markers",
    name,
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    customdata: points.map((point) => [
      sideLabel(point.side, sideLabels),
      classLabelFor(dataset, point.label),
      classLabelFor(dataset, point.prediction),
      point.correct ? "correct" : "incorrect",
    ]),
    hovertemplate:
      "<b>%{customdata[0]}</b><br>Class %{customdata[1]}<br>Prediction %{customdata[2]}<br>%{customdata[3]}<extra></extra>",
    marker: {
      color,
      opacity: 0.58,
      size: 2.5,
      symbol: points.map((point) => (point.side === "left" ? "circle" : "x")),
    },
  }
}

function embeddingClassColor(label: number): string {
  return EMBEDDING_CLASS_COLORS[
    Math.abs(label) % EMBEDDING_CLASS_COLORS.length
  ]
}

function robustnessData(
  report: AnalysisComparisonReport,
  sideLabels: AnalysisSideLabels
): Data[] {
  return report.robustness.flatMap((curve) => [
    {
      type: "scatter",
      mode: "lines+markers",
      name: `${sideShortLabel("left", sideLabels)} ${curve.perturbation}`,
      x: curve.points.map((point) => point.level),
      y: curve.points.map((point) => point.left_accuracy),
      line: { width: 1.5 },
    } satisfies Data,
    {
      type: "scatter",
      mode: "lines+markers",
      name: `${sideShortLabel("right", sideLabels)} ${curve.perturbation}`,
      x: curve.points.map((point) => point.level),
      y: curve.points.map((point) => point.right_accuracy),
      line: { width: 1.5, dash: "dot" },
    } satisfies Data,
  ])
}

function activationData(
  report: AnalysisComparisonReport,
  sideLabels: AnalysisSideLabels
): Data[] {
  const names = [
    ...new Set([
      ...report.activations.left.map((layer) => layer.name),
      ...report.activations.right.map((layer) => layer.name),
    ]),
  ]
  return [
    {
      type: "bar",
      name: sideLabel("left", sideLabels),
      x: names,
      y: names.map(
        (name) =>
          report.activations.left.find((layer) => layer.name === name)
            ?.sparsity ?? 0
      ),
      marker: { color: "#2563eb" },
    },
    {
      type: "bar",
      name: sideLabel("right", sideLabels),
      x: names,
      y: names.map(
        (name) =>
          report.activations.right.find((layer) => layer.name === name)
            ?.sparsity ?? 0
      ),
      marker: { color: "#dc2626" },
    },
  ]
}

function weightData(report: AnalysisComparisonReport): Data[] {
  return [
    {
      type: "bar",
      orientation: "h",
      name: "Relative distance",
      x: report.weights.map((weight) => weight.relative_distance),
      y: report.weights.map((weight) => weight.name),
      marker: { color: "#0891b2" },
    },
  ]
}

function plotLayout(title: string, plotPalette: PlotPalette): Partial<Layout> {
  return {
    autosize: true,
    font: { color: plotPalette.text, family: "Geist Variable, sans-serif" },
    height: 360,
    hoverlabel: {
      bgcolor: plotPalette.hoverBackground,
      bordercolor: plotPalette.hoverBorder,
      font: { color: plotPalette.hoverText },
    },
    legend: { orientation: "h", font: { color: plotPalette.muted } },
    margin: { b: 46, l: 48, r: 36, t: 42 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    title: { text: title, font: { size: 14 } },
    xaxis: { gridcolor: plotPalette.grid, zeroline: false },
    yaxis: { gridcolor: plotPalette.grid, zeroline: false },
    yaxis2: {
      overlaying: "y",
      side: "right",
      range: [0, 100],
      showgrid: false,
      zeroline: false,
    },
  }
}

function heatmapLayout(title: string, plotPalette: PlotPalette): Partial<Layout> {
  return {
    ...plotLayout(title, plotPalette),
    xaxis: { title: { text: "Predicted" } },
    yaxis: { title: { text: "True" }, autorange: "reversed" },
  }
}

function embeddingLayout(title: string, plotPalette: PlotPalette): Partial<Layout> {
  return {
    ...plotLayout(title, plotPalette),
    dragmode: "pan",
    height: undefined,
    margin: { b: 24, l: 48, r: 36, t: 42 },
    xaxis: { visible: false },
    yaxis: { visible: false },
  }
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
    status.best_checkpoint_path ||
    status.best_checkpoint_saved_at ||
    status.best_checkpoint_step != null
  ) {
    return { run_id: status.run_id, kind: "best" }
  }

  if (
    status.checkpoint_path ||
    status.last_checkpoint_saved_at ||
    status.last_checkpoint_step != null
  ) {
    return { run_id: status.run_id, kind: "latest" }
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

function checkpointOptimizerOptions(
  checkpoints: CheckpointSummary[],
  schemaOptions: SelectOption[] | null | undefined
): SelectOption[] {
  return checkpointFilterOptions(
    checkpoints.map((checkpoint) => checkpoint.optimizer),
    schemaOptions
  )
}

function checkpointDatasetOptions(
  checkpoints: CheckpointSummary[],
  schemaOptions: SelectOption[] | null | undefined
): SelectOption[] {
  return checkpointFilterOptions(
    checkpoints.map((checkpoint) => checkpoint.dataset),
    schemaOptions
  )
}

function checkpointFilterOptions(
  values: string[],
  schemaOptions: SelectOption[] | null | undefined
): SelectOption[] {
  const availableValues = new Set(values)
  const options = (schemaOptions ?? []).filter((option) =>
    availableValues.has(option.value)
  )
  const optionValues = new Set(options.map((option) => option.value))
  const extraOptions = [...availableValues]
    .filter((value) => !optionValues.has(value))
    .sort((left, right) => left.localeCompare(right))
    .map((value) => ({ value, label: value }))

  return [...options, ...extraOptions]
}

function sortCheckpoints(
  checkpoints: CheckpointSummary[],
  sortKey: CheckpointSortKey,
  direction: SortDirection
): CheckpointSummary[] {
  return [...checkpoints].sort((left, right) =>
    compareCheckpoints(left, right, sortKey, direction)
  )
}

function compareCheckpoints(
  left: CheckpointSummary,
  right: CheckpointSummary,
  sortKey: CheckpointSortKey,
  direction: SortDirection
): number {
  const leftValue = checkpointSortValue(left, sortKey)
  const rightValue = checkpointSortValue(right, sortKey)
  const leftMissing = leftValue === null
  const rightMissing = rightValue === null

  if (leftMissing || rightMissing) {
    if (leftMissing && rightMissing) {
      return left.run_id.localeCompare(right.run_id)
    }
    return leftMissing ? 1 : -1
  }

  const comparison =
    direction === "asc" ? leftValue - rightValue : rightValue - leftValue
  return comparison || left.run_id.localeCompare(right.run_id)
}

function checkpointSortValue(
  checkpoint: CheckpointSummary,
  sortKey: CheckpointSortKey
): number | null {
  if (sortKey === "saved_at") {
    const timestamp = new Date(checkpoint.saved_at).getTime()
    return Number.isFinite(timestamp) ? timestamp : null
  }

  if (sortKey === "accuracy") {
    return finiteNumberOrNull(checkpoint.accuracy)
  }

  if (sortKey === "elapsed") {
    return finiteNumberOrNull(checkpoint.total_elapsed_seconds)
  }

  return finiteNumberOrNull(checkpoint.step)
}

function finiteNumberOrNull(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function checkpointDisplayName(checkpoint: CheckpointSummary): string {
  const accuracy = formatOptionalPercent(checkpoint.accuracy)
  const elapsed = formatOptionalDuration(checkpoint.total_elapsed_seconds)
  return `${checkpoint.optimizer} ${formatDatasetName(checkpoint.dataset)} checkpoint (${accuracy}, ${elapsed})`
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

function formatAnalysisTableValue(label: string, value: string): string {
  if (isMissingText(value)) {
    return MISSING_VALUE_LABEL
  }

  if (label === "Elapsed") {
    const seconds = secondsFromText(value)
    return seconds === null ? value : formatDuration(seconds)
  }

  if (label === "Saved") {
    return formatReadableDateTime(value)
  }

  return value
}

function isMissingText(value: string): boolean {
  const normalized = value.trim().toLowerCase()
  return normalized === "" || normalized === "n/a" || normalized === "na"
}

function secondsFromText(value: string): number | null {
  const trimmed = value.trim()
  const numeric = trimmed.endsWith("s") ? trimmed.slice(0, -1) : trimmed
  const seconds = Number(numeric)
  return Number.isFinite(seconds) ? seconds : null
}

function formatOptionalPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return MISSING_VALUE_LABEL
  }

  return `${value.toFixed(2)}%`
}

function formatOptionalDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) {
    return MISSING_VALUE_LABEL
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
  if (dataset === "cifar10") {
    return "CIFAR-10"
  }

  return dataset.toUpperCase()
}

function classLabelFor(dataset: CheckpointSummary["dataset"], label: number): string {
  if (dataset === "cifar10") {
    return CIFAR10_CLASS_LABELS[label] ?? String(label)
  }

  return String(label)
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 0,
  }).format(value)
}

function optimizerParamEntries(
  checkpoint: CheckpointSummary,
  schema: SchemaResponse
): [string, string, OptimizerParamValue][] {
  const params = checkpoint.optimizer_params[checkpoint.optimizer] ?? {}
  const fields = schema.optimizers_schema[checkpoint.optimizer] ?? []
  const fieldKeys = new Set(fields.map((field) => field.key))
  const schemaEntries: [string, string, OptimizerParamValue][] = fields.flatMap((field) => {
    const value = params[field.key]
    if (field.type === "boolean") {
      return typeof value === "boolean" ? [[field.key, field.label, value]] : []
    }
    return typeof value === "number" && Number.isFinite(value)
      ? [[field.key, field.label, value]]
      : []
  })
  const extraEntries: [string, string, OptimizerParamValue][] = Object.entries(params)
    .filter(
      ([key, value]) =>
        !fieldKeys.has(key) &&
        ((typeof value === "number" && Number.isFinite(value)) ||
          typeof value === "boolean")
    )
    .map(([key, value]) => [key, key, value])

  return [...schemaEntries, ...extraEntries]
}

function formatParamValue(value: OptimizerParamValue): string {
  if (typeof value === "boolean") {
    return value ? "On" : "Off"
  }

  if (Number.isInteger(value)) {
    return formatInteger(value)
  }

  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 6,
  }).format(value)
}

function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return MISSING_VALUE_LABEL
  }

  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(2)
  }

  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 4,
  }).format(value)
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min
  }

  return Math.min(max, Math.max(min, value))
}

function formatCheckpointDate(savedAt: string): string {
  return formatReadableDateTime(savedAt, { compact: true })
}

function formatReadableDateTime(
  savedAt: string,
  options: { compact?: boolean } = {}
): string {
  const date = new Date(savedAt)
  if (Number.isNaN(date.getTime())) {
    return savedAt
  }
  const dateOptions: Intl.DateTimeFormatOptions = {
    day: "numeric",
    hour: "2-digit",
    hour12: false,
    minute: "2-digit",
    month: "short",
  }

  if (!options.compact || date.getFullYear() !== new Date().getFullYear()) {
    dateOptions.year = "numeric"
  }

  return date.toLocaleString(undefined, dateOptions)
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

function fixedPrecisionTickValues(
  upperBound: number,
  preferredSegments: number,
  decimals: number
): number[] {
  const safeUpperBound =
    Number.isFinite(upperBound) && upperBound > 0 ? upperBound : 1
  const rawStep = safeUpperBound / preferredSegments
  const minimumDistinctStep = 10 ** -decimals
  const tickStep = Math.max(niceStepSize(rawStep), minimumDistinctStep)
  const tickCount = Math.floor(safeUpperBound / tickStep)
  const tickValues = Array.from({ length: tickCount + 1 }, (_, index) =>
    Number((tickStep * index).toPrecision(12))
  ).filter((value) => value <= safeUpperBound + Number.EPSILON)
  const roundedUpperBound = Number(safeUpperBound.toPrecision(12))
  const lastTick = tickValues[tickValues.length - 1] ?? 0

  if (
    formatFixedTick(lastTick, decimals) !==
    formatFixedTick(roundedUpperBound, decimals)
  ) {
    tickValues.push(roundedUpperBound)
  }

  return uniqueFixedTickValues(tickValues, decimals)
}

function uniqueFixedTickValues(values: number[], decimals: number): number[] {
  const labels = new Set<string>()

  return values.filter((value) => {
    const label = formatFixedTick(value, decimals)

    if (labels.has(label)) {
      return false
    }

    labels.add(label)
    return true
  })
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

function fixedTickText(values: number[], decimals: number): string[] {
  return values.map((value) => formatFixedTick(value, decimals))
}

function formatFixedTick(value: number, decimals: number): string {
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
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }

  return typeof mutationStepField.default === "number"
    ? mutationStepField.default
    : undefined
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
      const fields = schema.optimizers_schema[optimizer] ?? []

      if (!isRecord(storedParams)) {
        return [optimizer, params]
      }

      return [
        optimizer,
        Object.fromEntries(
          fields.map((field) => {
            const defaultValue = params[field.key] ?? field.default
            const value =
              field.type === "boolean"
                ? coerceBoolean(storedParams[field.key], Boolean(defaultValue))
                : coerceNumber(
                    storedParams[field.key],
                    typeof defaultValue === "number" ? defaultValue : 0
                  )

            return [field.key, value]
          })
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
    <div className="min-w-0 rounded-xl border bg-input/30 px-4 py-3">
      <p className="text-sm text-muted-foreground/80">{label}</p>
      <p className="flex min-w-0 items-baseline gap-1.5">
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

  if (safeSeconds < 1) {
    return `${Math.round(safeSeconds * 1000)}ms`
  }

  return `${formatSignificantNumber(safeSeconds, 3)}s`
}

function formatSignificantNumber(value: number, digits: number): string {
  return Number(value.toPrecision(digits)).toString()
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
