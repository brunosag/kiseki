import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import katex from "katex"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Scatter,
  ScatterChart,
  type ScatterShapeProps,
  type TooltipPayloadEntry,
  type TooltipValueType,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts"
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
  KeyboardEvent,
  PointerEvent,
  ReactElement,
  ReactNode,
  RefObject,
} from "react"

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
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart"
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
  fetchAnalysisComparisonReport,
  type AnalysisComparisonJobStatus,
  type AnalysisComparisonParams,
  type AnalysisComparisonReport,
  type AnalysisEmbeddingProjection,
  type AnalysisLrpSample,
  type AnalysisTableRow,
  type ConfigField,
  type ExperimentConfig,
  type ExperimentStatus,
  type ExperimentStatusCompactEvent,
  type OptimizerParamValue,
  type OptimizerParams,
  type SchemaResponse,
  type SelectOption,
  type TrainingHistory,
  type TrainingHistoryDelta,
} from "@/lib/api"
import { cn } from "@/lib/utils"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"

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

const CONFIG_STORAGE_KEY = "kiseki.config.v1"
const OPT_PARAMS_STORAGE_KEY = "kiseki.optimizerParams.v1"
const MARKER_POINT_LIMIT = 20
const MUTATION_STEP_AXIS_MIN_UPPER_BOUND = 0.001
const MUTATION_STEP_AXIS_PADDING = 1.05
const TRAINING_STATUS_UPDATE_INTERVAL_MS = 200
const MISSING_VALUE_LABEL = "—"
const LINE_RENDER_POINT_LIMIT = 2500
const TSNE_SIDE_SAMPLE_LIMIT = 2000
const PCA_TOTAL_SAMPLE_LIMIT = 2000
const EMBEDDING_DOMAIN_PADDING_RATIO = 0.04
const EMBEDDING_POINT_OPACITY = 0.88
const EMBEDDING_POINT_RADIUS = 2.8
const EMBEDDING_CROSS_HALF_SIZE = 3.1
const TSNE_PLOT_ASPECT_RATIO = 5 / 3
const EMBEDDING_TOOLTIP_OFFSET = 12
const EMBEDDING_TOOLTIP_HIT_RADIUS = 12
const EMBEDDING_CHART_MARGIN = { bottom: 20, left: 0, right: 0, top: 28 }

type ResolvedTheme = "dark" | "light"
type CheckpointSortKey = "saved_at" | "accuracy" | "step" | "elapsed"
type SortDirection = "asc" | "desc"
type CheckpointOptimizerFilter = ExperimentConfig["optimizer"] | "all"
type CheckpointDatasetFilter = ExperimentConfig["dataset"] | "all"
type AppTab = "training" | "analysis"
type AnalysisSide = "left" | "right"
type AnalysisSideLabels = Record<AnalysisSide, string>
type ExperimentStatusUpdater = (current: ExperimentStatus) => ExperimentStatus

type PlotPalette = {
  accuracy: string
  loss: string
  mutationStep: string
}

type ChartLegendItem = {
  color: string
  label: string
}

type NumericSeriesPoint = {
  x: number
  y: number
}

type NumericChartDatum = {
  x: number
} & Record<string, number | undefined>

type TrainingTelemetryChartState = {
  data: NumericChartDatum[]
  lossAxisUpperBound: number
  mutationStepAxisUpperBound: number
}

type TrainingTelemetryCache = {
  currentStep: number
  runId: string | null | undefined
  series: Record<TrainingTelemetrySeriesKey, IncrementalSeriesCache>
}

type TrainingTelemetrySeriesKey = "loss" | "accuracy" | "mutationStep"

type IncrementalSeriesCache = {
  bucketSize: number
  buckets: Map<number, NumericSeriesBucket>
  maxY: number
  processedLength: number
}

type NumericSeriesBucket = {
  max: NumericSeriesPoint | null
  min: NumericSeriesPoint | null
  values: Map<number, number>
}

type CategoryChartDatum = {
  label: string
} & Record<string, number | string | undefined>

type ChartSeries = {
  color: string
  dataKey: string
  dashed?: boolean
  label: string
}

type ChartValueFormatter = (value: number) => string

type ChartTooltipConfig = Record<
  string,
  {
    color: string
    formatter?: ChartValueFormatter
    label: string
  }
>

type EmbeddingChartPoint = AnalysisEmbeddingProjection["left"][number] & {
  className: string
  correctText: string
  predictionName: string
  side: AnalysisSide
  sideName: string
}

type EmbeddingPlotLayout = "domain" | "rotated-equal-scale"
type ChartPointerPosition = { x: number; y: number }
type EmbeddingChartLayout = {
  points: EmbeddingChartPoint[]
  xDomain: [number, number]
  yDomain: [number, number]
}
type EmbeddingPointGroup = {
  color: string
  label: string
  points: EmbeddingChartPoint[]
}

type ConfusionHeatmapDatum = {
  count: number
  fill: string
  predictedClass: string
  trueClass: string
  x: number
  y: number
}

type ConfusionHeatmapTooltipState = {
  cell: ConfusionHeatmapDatum
  x: number
  y: number
}

const fallbackPlotPalettes: Record<ResolvedTheme, PlotPalette> = {
  light: {
    accuracy: "#2563eb",
    loss: "#171717",
    mutationStep: "#737373",
  },
  dark: {
    accuracy: "#38bdf8",
    loss: "#fafafa",
    mutationStep: "#a3a3a3",
  },
}

const LONG_CHECKPOINT_DELETE_SECONDS = 600
const ANALYSIS_SIDES: AnalysisSide[] = ["left", "right"]
const CLASS_COUNT = 10
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
const REPORT_LEFT_COLOR = "#2563eb"
const REPORT_RIGHT_COLOR = "#dc2626"
const REPORT_LEFT_ACCENT_COLOR = "#0891b2"
const REPORT_RIGHT_ACCENT_COLOR = "#ea580c"

const FASHION_MNIST_CLASS_LABELS = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
]

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
  const [analysisReport, setAnalysisReport] = useState<{
    jobId: string
    report: AnalysisComparisonReport
  } | null>(null)
  const [analysisReportFailedJobId, setAnalysisReportFailedJobId] =
    useState<string | null>(null)
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
    const source = new EventSource(apiUrl("/api/experiments/events?compact=true"))
    let pendingStepUpdate: ExperimentStatusUpdater | null = null
    let pendingStepTimer: number | null = null
    let lastStatusUpdateAt = 0

    const statusUpdateFromEvent = (
      event: MessageEvent<string>
    ): ExperimentStatusUpdater => {
      const payload = JSON.parse(event.data) as
        | ExperimentStatus
        | ExperimentStatusCompactEvent
        | { status: ExperimentStatus }
      if ("status" in payload) {
        return () => payload.status
      }
      if (isExperimentStatusCompactEvent(payload)) {
        return (current) => applyExperimentStatusCompactEvent(current, payload)
      }
      return () => payload
    }

    const clearPendingStep = () => {
      if (pendingStepTimer !== null) {
        window.clearTimeout(pendingStepTimer)
      }
      pendingStepTimer = null
      pendingStepUpdate = null
    }

    const commitStatusUpdate = (updateStatus: ExperimentStatusUpdater) => {
      lastStatusUpdateAt = performance.now()
      setStatus((current) => updateStatus(current))
    }

    const commitPendingStep = () => {
      if (pendingStepUpdate !== null) {
        commitStatusUpdate(pendingStepUpdate)
      }
      pendingStepTimer = null
      pendingStepUpdate = null
    }

    const handleStepEvent = (event: MessageEvent<string>) => {
      const updateStatus = statusUpdateFromEvent(event)
      const elapsed = performance.now() - lastStatusUpdateAt
      if (elapsed >= TRAINING_STATUS_UPDATE_INTERVAL_MS) {
        clearPendingStep()
        commitStatusUpdate(updateStatus)
        return
      }

      pendingStepUpdate =
        pendingStepUpdate === null
          ? updateStatus
          : composeStatusUpdates(pendingStepUpdate, updateStatus)
      if (pendingStepTimer === null) {
        pendingStepTimer = window.setTimeout(
          commitPendingStep,
          TRAINING_STATUS_UPDATE_INTERVAL_MS - elapsed
        )
      }
    }

    const handleEvent = (event: MessageEvent<string>) => {
      clearPendingStep()
      commitStatusUpdate(statusUpdateFromEvent(event))
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

  useEffect(() => {
    if (
      analysisJobId === null ||
      analysisJobStatus !== "completed" ||
      !analysisJob?.report_available ||
      analysisReport?.jobId === analysisJobId ||
      analysisReportFailedJobId === analysisJobId
    ) {
      return
    }

    let ignore = false
    fetchAnalysisComparisonReport(analysisJobId)
      .then((report) => {
        if (!ignore) {
          setAnalysisReport({ jobId: analysisJobId, report })
          setAnalysisReportFailedJobId(null)
        }
      })
      .catch(() => {
        if (!ignore) {
          setAnalysisReportFailedJobId(analysisJobId)
          setAnalysisError("Failed to load comparison report")
        }
      })

    return () => {
      ignore = true
    }
  }, [
    analysisJob?.report_available,
    analysisJobId,
    analysisJobStatus,
    analysisReportFailedJobId,
    analysisReport?.jobId,
  ])

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
  const trainingTelemetryChart = useTrainingTelemetryChartState({
    accuracyPoints: status.history.acc,
    currentLoss: status.current_loss,
    currentMutationStep: status.current_mutation_step,
    currentStep: status.current_step,
    losses: status.history.loss,
    mutationStepPoints: mutationStepHistory,
    runId: status.run_id,
    selectedInitialMutationStep: shouldUseSelectedMutationStep
      ? selectedInitialMutationStep
      : undefined,
    showMutationStepAxis,
  })
  const comparisonError = useMemo(
    () => comparisonSelectionError(analysisCheckpoints),
    [analysisCheckpoints]
  )
  const currentAnalysisReport =
    analysisReport?.jobId === analysisJobId ? analysisReport.report : null
  const analysisReportPending =
    analysisJobId !== null &&
    analysisJob?.status === "completed" &&
    analysisJob.report_available &&
    currentAnalysisReport === null &&
    analysisReportFailedJobId !== analysisJobId
  const analysisBusy =
    analysisStarting ||
    analysisReportPending ||
    analysisJob?.status === "queued" ||
    analysisJob?.status === "running"

  const trainingTelemetryConfig = useMemo<ChartConfig>(
    () => ({
      loss: { label: "Loss", color: plotPalette.loss },
      accuracy: { label: "Accuracy", color: plotPalette.accuracy },
      mutationStep: {
        label: "Mutation step",
        color: plotPalette.mutationStep,
      },
    }),
    [plotPalette.accuracy, plotPalette.loss, plotPalette.mutationStep]
  )
  const trainingTelemetryTooltip = useMemo<ChartTooltipConfig>(
    () => ({
      loss: {
        label: "Loss",
        color: plotPalette.loss,
        formatter: (value) => value.toFixed(4),
      },
      accuracy: {
        label: "Accuracy",
        color: plotPalette.accuracy,
        formatter: (value) => `${value.toFixed(2)}%`,
      },
      mutationStep: {
        label: "Mutation step",
        color: plotPalette.mutationStep,
        formatter: formatMutationStep,
      },
    }),
    [plotPalette.accuracy, plotPalette.loss, plotPalette.mutationStep]
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
    setAnalysisReport(null)
    setAnalysisReportFailedJobId(null)
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
    setAnalysisReport(null)
    setAnalysisReportFailedJobId(null)
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
    setAnalysisReport(null)
    setAnalysisReportFailedJobId(null)
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
            <Card
              className="w-full rounded-lg border bg-transparent ring-0 md:h-fit md:max-h-full"
            >
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

          <Card
            className="min-h-[520px] w-full rounded-lg border bg-transparent ring-0 md:min-h-0 md:flex-1"
          >
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
                <TrainingTelemetryChart
                  accuracyPointCount={status.history.acc.length}
                  config={trainingTelemetryConfig}
                  data={trainingTelemetryChart.data}
                  lossAxisUpperBound={trainingTelemetryChart.lossAxisUpperBound}
                  lossPointCount={status.history.loss.length}
                  mutationStepAxisUpperBound={
                    trainingTelemetryChart.mutationStepAxisUpperBound
                  }
                  mutationStepPointCount={mutationStepHistory.length}
                  showMutationStepAxis={showMutationStepAxis}
                  stepAxisUpperBound={stepAxisUpperBound}
                  tooltipConfig={trainingTelemetryTooltip}
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
        <div className="mx-auto mt-8 grid w-full max-w-4xl gap-3">
          <div className="rounded-lg border bg-card/40 p-3">
            <div className="grid gap-3 md:grid-cols-2">
              {ANALYSIS_SIDES.map((side) => (
                <AnalysisCheckpointSlot
                  checkpoint={checkpoints[side]}
                  key={side}
                  pausedRunId={pausedRunId}
                  schema={schema}
                  side={side}
                  onLoadCheckpoint={onLoadCheckpoint}
                />
              ))}
            </div>
          </div>
          <div className="flex justify-center">
            <Button disabled={!canRun} onClick={() => onRun(false)}>
              {busy ? <LoaderCircle className="size-4 animate-spin" /> : null}
              Generate report
            </Button>
          </div>
        </div>
      ) : null}

      {error ?? comparisonError ? (
        <Alert className="mx-auto w-full max-w-4xl" variant="destructive">
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
        <Alert className="mx-auto w-full max-w-4xl" variant="destructive">
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
            key={job?.job_id ?? report.generated_at}
            plotPalette={plotPalette}
            report={report}
            schema={schema}
            scrollRootRef={scrollRootRef}
          />
        </ScrollArea>
      ) : null}
    </div>
  )
}

function AnalysisCheckpointSlot({
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
  if (!checkpoint) {
    return (
      <CheckpointPicker
        closeOnLoadStart
        currentSelection={null}
        disabled={false}
        mode="analysis"
        pausedRunId={pausedRunId}
        schema={schema}
        trigger={
          <Button
            className="h-full min-h-[22rem] w-full flex-col gap-2 rounded-lg border border-dashed border-border bg-transparent px-4 text-muted-foreground hover:border-foreground/30 hover:bg-muted/40 hover:text-foreground"
            variant="ghost"
          >
            <Plus className="size-4" />
            <span>Add {sideLabel(side)}</span>
          </Button>
        }
        onLoad={(nextCheckpoint) => onLoadCheckpoint(side, nextCheckpoint)}
      />
    )
  }

  return (
    <div className="rounded-lg border bg-background p-2">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead>{sideLabel(side)}</TableHead>
            <TableHead className="text-right">
              <CheckpointPicker
                closeOnLoadStart
                currentSelection={selectionFromCheckpoint(checkpoint)}
                disabled={false}
                mode="analysis"
                pausedRunId={pausedRunId}
                schema={schema}
                trigger={
                  <Button size="sm" variant="outline">
                    <FolderOpen className="size-3.5" />
                    Change
                  </Button>
                }
                onLoad={(nextCheckpoint) =>
                  onLoadCheckpoint(side, nextCheckpoint)
                }
              />
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {analysisCheckpointSummaryRows(checkpoint).map(([label, value]) => (
            <TableRow className="hover:bg-transparent" key={label}>
              <TableCell className="text-muted-foreground">{label}</TableCell>
              <TableCell className="max-w-0 truncate text-right font-medium">
                {value}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function AnalysisProgress({ job }: { job: AnalysisComparisonJobStatus | null }) {
  const progress = Math.round((job?.progress ?? 0) * 100)
  return (
    <div className="mx-auto w-full max-w-4xl rounded-lg border bg-card/40 p-3">
      <div className="flex min-w-0 items-center gap-2 text-sm font-medium">
        <LoaderCircle className="size-4 shrink-0 animate-spin text-muted-foreground" />
        <span className="truncate">{analysisProgressLabel(job?.stage)}</span>
      </div>
      <Progress className="mt-3" value={progress} />
    </div>
  )
}

function analysisProgressLabel(stage: string | null | undefined): string {
  switch (stage) {
    case "load/cache":
      return "Loading checkpoints and cache"
    case "inference":
      return "Running model inference"
    case "metrics":
      return "Computing metrics"
    case "embeddings":
      return "Generating embedding projections"
    case "LRP":
      return "Computing LRP maps"
    case "activation/weights":
      return "Summarizing activations and weights"
    case "robustness":
      return "Evaluating robustness curves"
    case "persist":
      return "Writing report cache"
    case "failed":
      return "Handling analysis failure"
    default:
      return "Preparing report"
  }
}

function analysisCheckpointSummaryRows(
  checkpoint: CheckpointSummary
): [string, string][] {
  return [
    ["Optimizer", checkpoint.optimizer],
    ["Dataset", formatDatasetName(checkpoint.dataset)],
    ["Seed", formatInteger(checkpoint.seed)],
    ["Batch size", formatInteger(checkpoint.config.batch_size)],
    ["Steps", formatInteger(checkpoint.step)],
    ["Validation accuracy", formatOptionalPercent(checkpoint.accuracy)],
    ["Elapsed time", formatOptionalDuration(checkpoint.total_elapsed_seconds)],
    ["Saved at", formatCheckpointDate(checkpoint.saved_at)],
  ]
}

const ANALYSIS_REPORT_SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "metrics", label: "Metrics" },
  { id: "embeddings", label: "Embeddings" },
  { id: "lrp", label: "LRP" },
] as const

type AnalysisReportSectionId = (typeof ANALYSIS_REPORT_SECTIONS)[number]["id"]
type AnalysisMountedSections = Record<AnalysisReportSectionId, boolean>

const INITIAL_ANALYSIS_MOUNTED_SECTIONS: AnalysisMountedSections = {
  overview: true,
  metrics: false,
  embeddings: false,
  lrp: false,
}

function AnalysisReport({
  plotPalette,
  report,
  schema,
  scrollRootRef,
}: {
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  schema: SchemaResponse
  scrollRootRef: RefObject<HTMLDivElement | null>
}) {
  const [activeSection, setActiveSection] =
    useState<AnalysisReportSectionId>("overview")
  const [mountedSections, setMountedSections] =
    useState<AnalysisMountedSections>(INITIAL_ANALYSIS_MOUNTED_SECTIONS)
  const sectionRefs = useRef<Record<AnalysisReportSectionId, HTMLElement | null>>({
    overview: null,
    metrics: null,
    embeddings: null,
    lrp: null,
  })
  const setSectionRef = useCallback(
    (sectionId: AnalysisReportSectionId, element: HTMLElement | null) => {
      sectionRefs.current[sectionId] = element
    },
    []
  )

  useEffect(() => {
    let cancelled = false
    const sectionQueue: AnalysisReportSectionId[] = [
      "metrics",
      "embeddings",
      "lrp",
    ]
    const timers: number[] = []

    const mountNextSection = (index: number) => {
      if (cancelled || index >= sectionQueue.length) {
        return
      }
      const sectionId = sectionQueue[index]
      setMountedSections((current) =>
        current[sectionId] ? current : { ...current, [sectionId]: true }
      )
      timers.push(window.setTimeout(() => mountNextSection(index + 1), 120))
    }

    timers.push(window.setTimeout(() => mountNextSection(0), 0))

    return () => {
      cancelled = true
      for (const timer of timers) {
        window.clearTimeout(timer)
      }
    }
  }, [report])

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
  }, [scrollRootRef])

  const selectSection = useCallback((sectionId: AnalysisReportSectionId) => {
    setActiveSection(sectionId)
    setMountedSections((current) =>
      current[sectionId] ? current : { ...current, [sectionId]: true }
    )
    window.requestAnimationFrame(() => {
      sectionRefs.current[sectionId]?.scrollIntoView({
        block: "start",
        behavior: "smooth",
      })
    })
  }, [])

  return (
    <div className="grid min-h-0 flex-1 gap-4 pb-6 lg:grid-cols-[10rem_minmax(0,1fr)] xl:grid-cols-[12rem_minmax(0,1fr)]">
      <AnalysisReportToc
        activeSection={activeSection}
        onSelect={selectSection}
      />
      <AnalysisReportSections
        mountedSections={mountedSections}
        plotPalette={plotPalette}
        report={report}
        schema={schema}
        setSectionRef={setSectionRef}
      />
    </div>
  )
}

const AnalysisReportSections = memo(function AnalysisReportSections({
  mountedSections,
  plotPalette,
  report,
  schema,
  setSectionRef,
}: {
  mountedSections: AnalysisMountedSections
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  schema: SchemaResponse
  setSectionRef: (
    sectionId: AnalysisReportSectionId,
    element: HTMLElement | null
  ) => void
}) {
  const sideLabels = useMemo(() => analysisSideLabels(report), [report])

  return (
    <div className="grid min-w-0 gap-8">
      <AnalysisReportSection
        id="overview"
        sectionRef={(element) => {
          setSectionRef("overview", element)
        }}
        title="Overview"
      >
        <AnalysisOverview
          plotPalette={plotPalette}
          report={report}
          schema={schema}
          sideLabels={sideLabels}
        />
      </AnalysisReportSection>
      {mountedSections.metrics ? (
        <AnalysisReportSection
          id="metrics"
          sectionRef={(element) => {
            setSectionRef("metrics", element)
          }}
          title="Metrics"
        >
          <AnalysisMetrics
            plotPalette={plotPalette}
            report={report}
            sideLabels={sideLabels}
          />
        </AnalysisReportSection>
      ) : null}
      {mountedSections.embeddings ? (
        <AnalysisReportSection
          id="embeddings"
          sectionRef={(element) => {
            setSectionRef("embeddings", element)
          }}
          title="Embeddings"
        >
          <AnalysisEmbeddingsView report={report} sideLabels={sideLabels} />
        </AnalysisReportSection>
      ) : null}
      {mountedSections.lrp ? (
        <AnalysisReportSection
          id="lrp"
          sectionRef={(element) => {
            setSectionRef("lrp", element)
          }}
          title="LRP"
        >
          <AnalysisLrpView report={report} sideLabels={sideLabels} />
        </AnalysisReportSection>
      ) : null}
    </div>
  )
})

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
  schema,
  sideLabels,
}: {
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  schema: SchemaResponse
  sideLabels: AnalysisSideLabels
}) {
  const rowHeaderLabel =
    report.left.optimizer === report.right.optimizer ? "Model" : "Optimizer"
  const overviewRows = useMemo(
    () => analysisOverviewRows(report, rowHeaderLabel),
    [report, rowHeaderLabel]
  )

  return (
    <div className="grid gap-4">
      <div className="flex flex-col items-stretch gap-4 xl:flex-row">
        <AnalysisRowsTable
          className="min-w-0 flex-1"
          rows={overviewRows}
          sideLabels={sideLabels}
        />
        <AnalysisOptimizerParameterCards
          className="xl:shrink-0 xl:self-stretch"
          report={report}
          schema={schema}
          sideLabels={sideLabels}
        />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <TrainingHistoryChart
          report={report}
          sideLabels={sideLabels}
        />
        <OutcomeOverlapChart
          plotPalette={plotPalette}
          report={report}
          sideLabels={sideLabels}
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
        <ConfusionHeatmapChart
          dataset={report.left.dataset}
          matrix={report.metrics.left.confusion_matrix}
          title={`${sideLabel("left", sideLabels)} Confusion`}
        />
        <ConfusionHeatmapChart
          dataset={report.left.dataset}
          matrix={report.metrics.right.confusion_matrix}
          title={`${sideLabel("right", sideLabels)} Confusion`}
        />
        <ConfusionHeatmapChart
          deltaLabel={confusionDeltaLabel}
          dataset={report.left.dataset}
          matrix={report.confusion_difference}
          title={`Confusion Delta (${confusionDeltaLabel})`}
        />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <CalibrationChart
          plotPalette={plotPalette}
          report={report}
          sideLabels={sideLabels}
        />
        <PerClassMetricTable report={report} sideLabels={sideLabels} />
      </div>
    </div>
  )
}

function AnalysisEmbeddingsView({
  report,
  sideLabels,
}: {
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  const tsneLeftPoints = useMemo(
    () => embeddingSidePoints(report.embeddings.tsne.left, "left"),
    [report.embeddings.tsne.left]
  )
  const tsneRightPoints = useMemo(
    () => embeddingSidePoints(report.embeddings.tsne.right, "right"),
    [report.embeddings.tsne.right]
  )
  const pcaPoints = useMemo(
    () => embeddingProjectionPoints(report.embeddings.pca),
    [report.embeddings.pca]
  )

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <EmbeddingChartPanel
        bodyClassName="h-[20rem] xl:h-auto xl:flex-1"
        className="xl:aspect-[5/3]"
        dataset={report.left.dataset}
        layout="rotated-equal-scale"
        pointShape="circle"
        points={tsneLeftPoints}
        plotAspectRatio={TSNE_PLOT_ASPECT_RATIO}
        sampleLimit={TSNE_SIDE_SAMPLE_LIMIT}
        sideLabels={sideLabels}
        title={`t-SNE ${sideLabel("left", sideLabels)}`}
        totalPointCount={report.embeddings.tsne.left_total}
      />
      <EmbeddingChartPanel
        bodyClassName="h-[20rem] xl:h-auto xl:flex-1"
        className="xl:aspect-[5/3]"
        dataset={report.left.dataset}
        layout="rotated-equal-scale"
        pointShape="circle"
        points={tsneRightPoints}
        plotAspectRatio={TSNE_PLOT_ASPECT_RATIO}
        sampleLimit={TSNE_SIDE_SAMPLE_LIMIT}
        sideLabels={sideLabels}
        title={`t-SNE ${sideLabel("right", sideLabels)}`}
        totalPointCount={report.embeddings.tsne.right_total}
      />
      <div className="xl:col-span-2">
        <EmbeddingChartPanel
          dataset={report.left.dataset}
          pointShape="circle"
          points={pcaPoints}
          sampleLimit={PCA_TOTAL_SAMPLE_LIMIT}
          sideLabels={sideLabels}
          title="Joint PCA"
          totalPointCount={
            report.embeddings.pca.left_total + report.embeddings.pca.right_total
          }
          xDomain={report.embeddings.pca.x_domain}
          yDomain={report.embeddings.pca.y_domain}
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
    </div>
  )
}

function AnalysisRowsTable({
  className,
  rows,
  sideLabels,
  title,
}: {
  className?: string
  rows: AnalysisTableRow[]
  sideLabels: AnalysisSideLabels
  title?: string
}) {
  return (
    <div className={cn("rounded-lg border p-3", className)}>
      {title ? <div className="mb-2 text-sm font-medium">{title}</div> : null}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead />
            <TableHead>{sideLabel("left", sideLabels)}</TableHead>
            <TableHead>{sideLabel("right", sideLabels)}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => {
            const formattedValues = formatAnalysisTableRowValues(row)

            return (
              <TableRow key={row.label}>
                <TableCell className="text-muted-foreground">
                  {row.label}
                </TableCell>
                <TableCell>{formattedValues.left}</TableCell>
                <TableCell>{formattedValues.right}</TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}

function AnalysisOptimizerParameterCards({
  className,
  report,
  schema,
  sideLabels,
}: {
  className?: string
  report: AnalysisComparisonReport
  schema: SchemaResponse
  sideLabels: AnalysisSideLabels
}) {
  return (
    <div
      className={cn(
        "flex w-fit max-w-full flex-wrap items-stretch justify-start gap-4",
        className
      )}
    >
      {ANALYSIS_SIDES.map((side) => {
        const checkpoint = report[side]
        const entries = optimizerParamEntries(checkpoint, schema)

        return (
          <div
            className="flex w-fit max-w-full flex-col rounded-lg border py-4 pr-12 pl-4"
            key={side}
          >
            <div className="text-sm font-medium">
              {sideLabel(side, sideLabels)} parameters
            </div>
            {entries.length > 0 ? (
              <div className="mt-4 grid w-fit max-w-full grid-cols-[max-content_max-content_max-content] items-baseline gap-x-3 gap-y-2">
                {entries.map(([key, label, value]) => {
                  const field = schema.optimizers_schema[
                    checkpoint.optimizer
                  ]?.find((param) => param.key === key)

                  return (
                    <div className="contents" key={key}>
                      <span className="text-base text-muted-foreground/90">
                        <MathLabel math={label} />
                      </span>
                      <span className="text-sm tabular-nums">
                        {formatParamValue(value)}
                      </span>
                      <span className="min-w-0 text-sm text-muted-foreground/70">
                        {field?.desc ?? key}
                      </span>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="mt-4 text-sm text-muted-foreground">
                No optimizer parameters.
              </div>
            )}
          </div>
        )
      })}
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

function ChartPanel({
  bodyClassName = "h-[20rem]",
  children,
  className,
  description,
  footer,
  title,
}: {
  bodyClassName?: string
  children: ReactNode
  className?: string
  description?: string
  footer?: ReactNode
  title: string
}) {
  return (
    <Card
      className={cn(
        "min-h-0 rounded-lg border bg-transparent ring-0",
        className
      )}
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{title}</CardTitle>
        {description ? <CardDescription>{description}</CardDescription> : null}
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col">
        <div className={cn("min-h-0", bodyClassName)}>{children}</div>
        {footer ? (
          <div className="mt-3 flex shrink-0 flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
            {footer}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}

function FormattedChartTooltip({
  active,
  config,
  label,
  labelFormatter,
  payload,
}: {
  active?: boolean
  config: ChartTooltipConfig
  label?: unknown
  labelFormatter?: (label: unknown) => string
  payload?: readonly TooltipPayloadEntry<TooltipValueType, string | number>[]
}) {
  const rows = (payload ?? []).filter(
    (item) => item.type !== "none" && item.value !== undefined
  )

  if (!active || rows.length === 0) {
    return null
  }

  return (
    <div className="grid min-w-36 items-start gap-1.5 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl">
      {labelFormatter ? (
        <div className="font-medium">{labelFormatter(label)}</div>
      ) : null}
      <div className="grid gap-1.5">
        {rows.map((item, index) => {
          const dataKey = tooltipDataKey(item)
          const itemConfig = config[dataKey]
          const color =
            item.color ??
            item.stroke ??
            item.fill ??
            itemConfig?.color ??
            "currentColor"

          return (
            <div className="flex items-center gap-2" key={`${dataKey}-${index}`}>
              <span
                aria-hidden="true"
                className="size-2 shrink-0 rounded-[2px]"
                style={{ backgroundColor: color }}
              />
              <span className="min-w-0 flex-1 truncate text-muted-foreground">
                {itemConfig?.label ?? item.name ?? dataKey}
              </span>
              <span className="font-mono font-medium text-foreground tabular-nums">
                {formatTooltipValue(item.value, itemConfig?.formatter)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function TrainingTelemetryChart({
  accuracyPointCount,
  config,
  data,
  lossAxisUpperBound,
  lossPointCount,
  mutationStepAxisUpperBound,
  mutationStepPointCount,
  showMutationStepAxis,
  stepAxisUpperBound,
  tooltipConfig,
}: {
  accuracyPointCount: number
  config: ChartConfig
  data: NumericChartDatum[]
  lossAxisUpperBound: number
  lossPointCount: number
  mutationStepAxisUpperBound: number
  mutationStepPointCount: number
  showMutationStepAxis: boolean
  stepAxisUpperBound: number
  tooltipConfig: ChartTooltipConfig
}) {
  return (
    <ChartContainer className="h-full w-full aspect-auto" config={config}>
      <ComposedChart
        accessibilityLayer
        data={data}
        margin={{ bottom: 0, left: 8, right: 8, top: 12 }}
      >
        <CartesianGrid vertical={false} />
        <XAxis
          axisLine={false}
          dataKey="x"
          domain={[0, stepAxisUpperBound]}
          minTickGap={28}
          tickFormatter={formatAxisInteger}
          tickLine={false}
          tickMargin={8}
          type="number"
        />
        <YAxis domain={[0, lossAxisUpperBound]} hide yAxisId="loss" />
        <YAxis domain={[0, 100]} hide yAxisId="accuracy" />
        {showMutationStepAxis ? (
          <YAxis
            domain={[0, mutationStepAxisUpperBound]}
            hide
            yAxisId="mutationStep"
          />
        ) : null}
        <ChartTooltip
          content={
            <FormattedChartTooltip
              config={tooltipConfig}
              labelFormatter={(value) => `Step ${formatAxisInteger(value)}`}
            />
          }
        />
        <ChartLegend
          content={<ChartLegendContent className="flex-wrap gap-x-5 gap-y-1" />}
          verticalAlign="top"
        />
        {lossPointCount > 0 ? (
          <Line
            connectNulls
            dataKey="loss"
            dot={lossPointCount <= MARKER_POINT_LIMIT}
            isAnimationActive={false}
            name="Loss"
            stroke="var(--color-loss)"
            strokeWidth={1.6}
            type="monotone"
            yAxisId="loss"
          />
        ) : null}
        {accuracyPointCount > 0 ? (
          <Line
            connectNulls
            dataKey="accuracy"
            dot={accuracyPointCount <= MARKER_POINT_LIMIT}
            isAnimationActive={false}
            name="Accuracy"
            stroke="var(--color-accuracy)"
            strokeWidth={1.6}
            type="monotone"
            yAxisId="accuracy"
          />
        ) : null}
        {showMutationStepAxis && mutationStepPointCount > 0 ? (
          <Line
            connectNulls
            dataKey="mutationStep"
            dot={false}
            isAnimationActive={false}
            name="Mutation step"
            stroke="var(--color-mutationStep)"
            strokeDasharray="4 4"
            strokeWidth={1.6}
            type="monotone"
            yAxisId="mutationStep"
          />
        ) : null}
      </ComposedChart>
    </ChartContainer>
  )
}

function TrainingHistoryChart({
  report,
  sideLabels,
}: {
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  const series = useMemo<
    (ChartSeries & {
      points: NumericSeriesPoint[]
      yAxisId: "accuracy" | "loss"
    })[]
  >(
    () => [
      {
        color: REPORT_LEFT_COLOR,
        dataKey: "leftLoss",
        label: `${sideShortLabel("left", sideLabels)} train loss`,
        points: indexedNumberSeries(report.curves.left.training_loss),
        yAxisId: "loss",
      },
      {
        color: REPORT_RIGHT_COLOR,
        dataKey: "rightLoss",
        label: `${sideShortLabel("right", sideLabels)} train loss`,
        points: indexedNumberSeries(report.curves.right.training_loss),
        yAxisId: "loss",
      },
      {
        color: REPORT_LEFT_ACCENT_COLOR,
        dashed: true,
        dataKey: "leftAccuracy",
        label: `${sideShortLabel("left", sideLabels)} val acc.`,
        points: accuracyPointSeries(report.curves.left.validation_accuracy),
        yAxisId: "accuracy",
      },
      {
        color: REPORT_RIGHT_ACCENT_COLOR,
        dashed: true,
        dataKey: "rightAccuracy",
        label: `${sideShortLabel("right", sideLabels)} val acc.`,
        points: accuracyPointSeries(report.curves.right.validation_accuracy),
        yAxisId: "accuracy",
      },
    ],
    [
      report.curves.left.training_loss,
      report.curves.left.validation_accuracy,
      report.curves.right.training_loss,
      report.curves.right.validation_accuracy,
      sideLabels,
    ]
  )
  const data = useMemo(
    () =>
      mergeNumericSeries(
        series.map((item) => ({
          dataKey: item.dataKey,
          points: downsampleNumericSeries(item.points),
        }))
      ),
    [series]
  )
  const config = useMemo(() => chartConfigFromSeries(series), [series])
  const tooltipConfig = useMemo(
    () =>
      chartTooltipConfigFromSeries(series, {
        leftAccuracy: (value) => `${value.toFixed(2)}%`,
        leftLoss: (value) => value.toFixed(4),
        rightAccuracy: (value) => `${value.toFixed(2)}%`,
        rightLoss: (value) => value.toFixed(4),
      }),
    [series]
  )
  const maxStep = useMemo(
    () => maxSeriesX(series.flatMap((item) => item.points)),
    [series]
  )
  const lossUpperBound = useMemo(
    () =>
      numericAxisUpperBoundFor(
        [
          ...report.curves.left.training_loss,
          ...report.curves.right.training_loss,
        ],
        null
      ),
    [report.curves.left.training_loss, report.curves.right.training_loss]
  )

  return (
    <ChartPanel title="Training History">
      <ChartContainer className="h-full w-full aspect-auto" config={config}>
        <ComposedChart
          accessibilityLayer
          data={data}
          margin={{ bottom: 0, left: 8, right: 8, top: 8 }}
        >
          <CartesianGrid vertical={false} />
          <XAxis
            axisLine={false}
            dataKey="x"
            domain={[0, nextStepAxisUpperBound(maxStep)]}
            minTickGap={28}
            tickFormatter={formatAxisInteger}
            tickLine={false}
            tickMargin={8}
            type="number"
          />
          <YAxis domain={[0, lossUpperBound]} hide yAxisId="loss" />
          <YAxis domain={[0, 100]} hide yAxisId="accuracy" />
          <ChartTooltip
            content={
              <FormattedChartTooltip
                config={tooltipConfig}
                labelFormatter={(value) => `Step ${formatAxisInteger(value)}`}
              />
            }
          />
          <ChartLegend
            content={
              <ChartLegendContent className="flex-wrap gap-x-5 gap-y-1" />
            }
            verticalAlign="top"
          />
          {series.map((item) => (
            <Line
              connectNulls
              dataKey={item.dataKey}
              dot={item.points.length <= MARKER_POINT_LIMIT}
              isAnimationActive={false}
              key={item.dataKey}
              name={item.label}
              stroke={`var(--color-${item.dataKey})`}
              strokeDasharray={item.dashed ? "4 4" : undefined}
              strokeWidth={1.6}
              type="monotone"
              yAxisId={item.yAxisId}
            />
          ))}
        </ComposedChart>
      </ChartContainer>
    </ChartPanel>
  )
}

function OutcomeOverlapChart({
  plotPalette,
  report,
  sideLabels,
}: {
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  const data = useMemo(
    () =>
      report.overlap.upset.map((row) => ({
        count: row.count,
        label: overlapSetLabel(row.set, sideLabels),
      })),
    [report.overlap.upset, sideLabels]
  )
  const config = useMemo<ChartConfig>(
    () => ({
      count: { label: "Count", color: plotPalette.accuracy },
    }),
    [plotPalette.accuracy]
  )
  const tooltipConfig = useMemo<ChartTooltipConfig>(
    () => ({
      count: {
        color: plotPalette.accuracy,
        formatter: formatInteger,
        label: "Count",
      },
    }),
    [plotPalette.accuracy]
  )

  return (
    <CategoryBarChart
      config={config}
      data={data}
      dataKeys={["count"]}
      title="Outcome Overlap"
      tooltipConfig={tooltipConfig}
    />
  )
}

function CalibrationChart({
  plotPalette,
  report,
  sideLabels,
}: {
  plotPalette: PlotPalette
  report: AnalysisComparisonReport
  sideLabels: AnalysisSideLabels
}) {
  const series = useMemo<ChartSeries[]>(
    () => [
      {
        color: REPORT_LEFT_COLOR,
        dataKey: "left",
        label: sideLabel("left", sideLabels),
      },
      {
        color: REPORT_RIGHT_COLOR,
        dataKey: "right",
        label: sideLabel("right", sideLabels),
      },
      {
        color: plotPalette.mutationStep,
        dashed: true,
        dataKey: "ideal",
        label: "Ideal",
      },
    ],
    [plotPalette.mutationStep, sideLabels]
  )
  const data = useMemo(() => calibrationChartData(report), [report])
  const config = useMemo(() => chartConfigFromSeries(series), [series])
  const tooltipConfig = useMemo(
    () =>
      chartTooltipConfigFromSeries(series, {
        ideal: formatCompactNumber,
        left: formatCompactNumber,
        right: formatCompactNumber,
      }),
    [series]
  )

  return (
    <ChartPanel title="Calibration">
      <ChartContainer className="h-full w-full aspect-auto" config={config}>
        <ComposedChart
          accessibilityLayer
          data={data}
          margin={{ bottom: 0, left: 8, right: 8, top: 8 }}
        >
          <CartesianGrid vertical={false} />
          <XAxis
            axisLine={false}
            dataKey="x"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={formatCompactNumber}
            tickLine={false}
            tickMargin={8}
            type="number"
          />
          <YAxis domain={[0, 1]} hide />
          <ChartTooltip
            content={
              <FormattedChartTooltip
                config={tooltipConfig}
                labelFormatter={(value) =>
                  `Confidence ${formatAxisCompact(value)}`
                }
              />
            }
          />
          <ChartLegend
            content={
              <ChartLegendContent className="flex-wrap gap-x-5 gap-y-1" />
            }
            verticalAlign="top"
          />
          {series.map((item) => (
            <Line
              connectNulls
              dataKey={item.dataKey}
              dot={!item.dashed}
              isAnimationActive={false}
              key={item.dataKey}
              name={item.label}
              stroke={`var(--color-${item.dataKey})`}
              strokeDasharray={item.dashed ? "4 4" : undefined}
              strokeWidth={item.dashed ? 1.2 : 1.6}
              type="monotone"
            />
          ))}
        </ComposedChart>
      </ChartContainer>
    </ChartPanel>
  )
}

function CategoryBarChart({
  config,
  data,
  dataKeys,
  tickFormatter = compactCategoryTick,
  title,
  tooltipConfig,
  yUpperBound,
}: {
  config: ChartConfig
  data: CategoryChartDatum[]
  dataKeys: string[]
  tickFormatter?: (value: string) => string
  title: string
  tooltipConfig: ChartTooltipConfig
  yUpperBound?: number
}) {
  return (
    <ChartPanel title={title}>
      <ChartContainer className="h-full w-full aspect-auto" config={config}>
        <BarChart
          accessibilityLayer
          data={data}
          margin={{ bottom: 0, left: 8, right: 8, top: 8 }}
        >
          <CartesianGrid vertical={false} />
          <XAxis
            axisLine={false}
            dataKey="label"
            interval="preserveStartEnd"
            minTickGap={16}
            tickFormatter={(value) => tickFormatter(String(value))}
            tickLine={false}
            tickMargin={8}
          />
          <YAxis domain={[0, yUpperBound ?? "dataMax"]} hide />
          <ChartTooltip
            content={
              <FormattedChartTooltip
                config={tooltipConfig}
                labelFormatter={(value) => String(value ?? "")}
              />
            }
          />
          {dataKeys.length > 1 ? (
            <ChartLegend
              content={
                <ChartLegendContent className="flex-wrap gap-x-5 gap-y-1" />
              }
              verticalAlign="top"
            />
          ) : null}
          {dataKeys.map((dataKey) => (
            <Bar
              dataKey={dataKey}
              fill={`var(--color-${dataKey})`}
              isAnimationActive={false}
              key={dataKey}
              name={config[dataKey]?.label?.toString() ?? dataKey}
              radius={dataKeys.length === 1 ? [4, 4, 0, 0] : [3, 3, 0, 0]}
            >
              {dataKeys.length === 1
                ? data.map((_, cellIndex) => (
                    <Cell
                      fill={`var(--color-${dataKey})`}
                      key={`${dataKey}-${cellIndex}`}
                    />
                  ))
                : null}
            </Bar>
          ))}
        </BarChart>
      </ChartContainer>
    </ChartPanel>
  )
}

function EmbeddingChartPanel({
  bodyClassName = "h-[20rem]",
  className,
  dataset,
  layout = "domain",
  pointShape,
  points,
  plotAspectRatio = 1,
  sampleLimit,
  sideLabels,
  title,
  totalPointCount,
  xDomain,
  yDomain,
}: {
  bodyClassName?: string
  className?: string
  dataset: CheckpointSummary["dataset"]
  layout?: EmbeddingPlotLayout
  pointShape: "circle" | "side"
  points: AnalysisEmbeddingPlotPoint[]
  plotAspectRatio?: number
  sampleLimit: number
  sideLabels: AnalysisSideLabels
  title: string
  totalPointCount?: number
  xDomain?: [number, number]
  yDomain?: [number, number]
}) {
  const [activeTooltipPoint, setActiveTooltipPoint] =
    useState<EmbeddingChartPoint | null>(null)
  const activeTooltipKeyRef = useRef<string | null>(null)
  const tooltipElementRef = useRef<HTMLDivElement | null>(null)
  const tooltipPositionRef = useRef<ChartPointerPosition>({ x: 0, y: 0 })
  const sampledPoints = useMemo(
    () => sampleEmbeddingPoints(points, sampleLimit),
    [points, sampleLimit]
  )
  const baseChartPoints = useMemo(
    () =>
      sampledPoints.map((point) => ({
        ...point,
        className: classLabelFor(dataset, point.label),
        correctText: point.correct ? "correct" : "incorrect",
        predictionName: classLabelFor(dataset, point.prediction),
        sideName: sideLabel(point.side, sideLabels),
      })),
    [dataset, sampledPoints, sideLabels]
  )
  const chartLayout = useMemo(
    () =>
      layout === "rotated-equal-scale"
        ? rotatedEqualScaleEmbeddingLayout(baseChartPoints, plotAspectRatio)
        : {
            points: baseChartPoints,
            xDomain:
              xDomain ?? paddedNumericDomain(baseChartPoints.map((point) => point.x)),
            yDomain:
              yDomain ?? paddedNumericDomain(baseChartPoints.map((point) => point.y)),
          },
    [baseChartPoints, layout, plotAspectRatio, xDomain, yDomain]
  )
  const chartPoints = chartLayout.points
  const legendItems = useMemo(
    () => embeddingLegendItems(points, dataset),
    [dataset, points]
  )
  const config = useMemo(() => chartConfigFromLegendItems(legendItems), [legendItems])
  const groupedPoints = useMemo(
    () => groupedEmbeddingPoints(chartPoints),
    [chartPoints]
  )
  const resolvedTotalPointCount = totalPointCount ?? points.length
  const moveTooltipOverlay = useCallback((position: ChartPointerPosition) => {
    tooltipPositionRef.current = position
    const element = tooltipElementRef.current
    if (element) {
      element.style.transform = `translate3d(${position.x}px, ${position.y}px, 0)`
    }
  }, [])
  const setTooltipElement = useCallback((element: HTMLDivElement | null) => {
    tooltipElementRef.current = element
    if (!element) {
      return
    }

    const position = tooltipPositionRef.current
    element.style.transform = `translate3d(${position.x}px, ${position.y}px, 0)`
  }, [])
  const updateTooltipPosition = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const bounds = event.currentTarget.getBoundingClientRect()
      const pointer = {
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      }
      moveTooltipOverlay({
        x: pointer.x + EMBEDDING_TOOLTIP_OFFSET,
        y: pointer.y + EMBEDDING_TOOLTIP_OFFSET,
      })

      const nearestPoint = nearestEmbeddingPoint(
        chartPoints,
        chartLayout.xDomain,
        chartLayout.yDomain,
        bounds.width,
        bounds.height,
        pointer
      )
      const nearestKey = nearestPoint ? embeddingTooltipKey(nearestPoint) : null
      if (nearestKey === activeTooltipKeyRef.current) {
        return
      }

      activeTooltipKeyRef.current = nearestKey
      setActiveTooltipPoint(nearestPoint)
    },
    [chartLayout.xDomain, chartLayout.yDomain, chartPoints, moveTooltipOverlay]
  )
  const clearTooltipPosition = useCallback(() => {
    activeTooltipKeyRef.current = null
    setActiveTooltipPoint(null)
  }, [])

  return (
    <ChartPanel
      bodyClassName={bodyClassName}
      className={className}
      title={title}
    >
      <div
        className="relative h-full min-h-[18rem] w-full"
        onPointerLeave={clearTooltipPosition}
        onPointerMove={updateTooltipPosition}
      >
        <EmbeddingScatterPlot
          chartLayout={chartLayout}
          config={config}
          groupedPoints={groupedPoints}
          pointShape={pointShape}
        />
        <EmbeddingLegendOverlay items={legendItems} />
        {activeTooltipPoint ? (
          <div
            className="pointer-events-none absolute top-0 left-0 z-20 will-change-transform"
            ref={setTooltipElement}
          >
            <EmbeddingTooltipCard point={activeTooltipPoint} />
          </div>
        ) : null}
        <span className="pointer-events-none absolute bottom-0 left-0 text-[0.7rem] text-muted-foreground/60 tabular-nums">
          Sampled {formatInteger(sampledPoints.length)} /{" "}
          {formatInteger(resolvedTotalPointCount)}
        </span>
      </div>
    </ChartPanel>
  )
}

const EmbeddingScatterPlot = memo(function EmbeddingScatterPlot({
  chartLayout,
  config,
  groupedPoints,
  pointShape,
}: {
  chartLayout: EmbeddingChartLayout
  config: ChartConfig
  groupedPoints: EmbeddingPointGroup[]
  pointShape: "circle" | "side"
}) {
  return (
    <ChartContainer className="h-full w-full aspect-auto" config={config}>
      <ScatterChart accessibilityLayer margin={EMBEDDING_CHART_MARGIN}>
        <XAxis dataKey="x" domain={chartLayout.xDomain} hide type="number" />
        <YAxis dataKey="y" domain={chartLayout.yDomain} hide type="number" />
        <ZAxis range={[24, 24]} />
        {groupedPoints.map((group) => (
          <Scatter
            data={group.points}
            fill={group.color}
            isAnimationActive={false}
            key={group.label}
            legendType="circle"
            name={group.label}
            shape={
              pointShape === "circle"
                ? EmbeddingCircleShape
                : EmbeddingScatterShape
            }
          />
        ))}
      </ScatterChart>
    </ChartContainer>
  )
})

function EmbeddingLegendOverlay({ items }: { items: ChartLegendItem[] }) {
  return (
    <div className="pointer-events-none absolute top-0 left-1/2 z-10 flex max-w-[calc(100%-2rem)] -translate-x-1/2 flex-wrap justify-center gap-x-4 gap-y-1 text-xs">
      {items.map((item) => (
        <div className="flex items-center gap-1.5" key={item.label}>
          <span
            aria-hidden="true"
            className="size-2 rounded-[2px]"
            style={{ backgroundColor: item.color }}
          />
          <span className="text-foreground">{item.label}</span>
        </div>
      ))}
    </div>
  )
}

function rotatedEqualScaleEmbeddingLayout(
  points: EmbeddingChartPoint[],
  aspectRatio: number
): {
  points: EmbeddingChartPoint[]
  xDomain: [number, number]
  yDomain: [number, number]
} {
  if (points.length === 0) {
    return { points, xDomain: [0, 1], yDomain: [0, 1] }
  }

  const center = meanEmbeddingPoint(points)
  const angle = principalAxisAngle(points, center)
  const cos = Math.cos(-angle)
  const sin = Math.sin(-angle)
  const rotatedPoints = points.map((point) => {
    const x = point.x - center.x
    const y = point.y - center.y
    return {
      ...point,
      x: x * cos - y * sin,
      y: x * sin + y * cos,
    }
  })
  const { xDomain, yDomain } = aspectMatchedEmbeddingDomains(
    rotatedPoints,
    aspectRatio
  )

  return { points: rotatedPoints, xDomain, yDomain }
}

function meanEmbeddingPoint(points: Pick<EmbeddingChartPoint, "x" | "y">[]): {
  x: number
  y: number
} {
  const sum = points.reduce(
    (total, point) => ({
      x: total.x + point.x,
      y: total.y + point.y,
    }),
    { x: 0, y: 0 }
  )

  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
  }
}

function principalAxisAngle(
  points: Pick<EmbeddingChartPoint, "x" | "y">[],
  center: { x: number; y: number }
): number {
  let covarianceXx = 0
  let covarianceXy = 0
  let covarianceYy = 0

  for (const point of points) {
    const x = point.x - center.x
    const y = point.y - center.y
    covarianceXx += x * x
    covarianceXy += x * y
    covarianceYy += y * y
  }

  if (covarianceXx === 0 && covarianceXy === 0 && covarianceYy === 0) {
    return 0
  }

  return 0.5 * Math.atan2(2 * covarianceXy, covarianceXx - covarianceYy)
}

function aspectMatchedEmbeddingDomains(
  points: Pick<EmbeddingChartPoint, "x" | "y">[],
  aspectRatio: number
): { xDomain: [number, number]; yDomain: [number, number] } {
  const xDomain = paddedNumericDomain(points.map((point) => point.x))
  const yDomain = paddedNumericDomain(points.map((point) => point.y))
  const centerX = (xDomain[0] + xDomain[1]) / 2
  const centerY = (yDomain[0] + yDomain[1]) / 2
  const safeAspectRatio =
    Number.isFinite(aspectRatio) && aspectRatio > 0 ? aspectRatio : 1
  let width = xDomain[1] - xDomain[0]
  let height = yDomain[1] - yDomain[0]

  if (width / height > safeAspectRatio) {
    height = width / safeAspectRatio
  } else {
    width = height * safeAspectRatio
  }

  return {
    xDomain: [centerX - width / 2, centerX + width / 2],
    yDomain: [centerY - height / 2, centerY + height / 2],
  }
}

function EmbeddingTooltipCard({ point }: { point: EmbeddingChartPoint }) {
  return (
    <div className="grid min-w-40 items-start gap-1.5 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl">
      <div className="font-medium">{point.sideName}</div>
      <div className="grid gap-1">
        <div className="flex justify-between gap-4">
          <span className="text-muted-foreground">Class</span>
          <span className="font-medium">{point.className}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-muted-foreground">Prediction</span>
          <span className="font-medium">{point.predictionName}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-muted-foreground">Result</span>
          <span className="font-medium">{point.correctText}</span>
        </div>
      </div>
    </div>
  )
}

function nearestEmbeddingPoint(
  points: EmbeddingChartPoint[],
  xDomain: [number, number],
  yDomain: [number, number],
  width: number,
  height: number,
  pointer: ChartPointerPosition
): EmbeddingChartPoint | null {
  const plotWidth =
    width - EMBEDDING_CHART_MARGIN.left - EMBEDDING_CHART_MARGIN.right
  const plotHeight =
    height - EMBEDDING_CHART_MARGIN.top - EMBEDDING_CHART_MARGIN.bottom
  const xSpan = xDomain[1] - xDomain[0]
  const ySpan = yDomain[1] - yDomain[0]

  if (
    plotWidth <= 0 ||
    plotHeight <= 0 ||
    xSpan <= 0 ||
    ySpan <= 0 ||
    !Number.isFinite(xSpan) ||
    !Number.isFinite(ySpan)
  ) {
    return null
  }

  const hitRadiusSquared = EMBEDDING_TOOLTIP_HIT_RADIUS ** 2
  let nearestPoint: EmbeddingChartPoint | null = null
  let nearestDistanceSquared = hitRadiusSquared

  for (const point of points) {
    const pointX =
      EMBEDDING_CHART_MARGIN.left +
      ((point.x - xDomain[0]) / xSpan) * plotWidth
    const pointY =
      EMBEDDING_CHART_MARGIN.top +
      ((yDomain[1] - point.y) / ySpan) * plotHeight
    const deltaX = pointX - pointer.x
    const deltaY = pointY - pointer.y
    const distanceSquared = deltaX * deltaX + deltaY * deltaY

    if (distanceSquared <= nearestDistanceSquared) {
      nearestPoint = point
      nearestDistanceSquared = distanceSquared
    }
  }

  return nearestPoint
}

function embeddingTooltipKey(point: EmbeddingChartPoint): string {
  return `${point.side}:${point.index}`
}

function EmbeddingScatterShape(props: ScatterShapeProps & { fill?: string }) {
  const point: unknown = props.payload
  const color = props.fill ?? "currentColor"
  const cx = props.cx
  const cy = props.cy

  if (typeof cx !== "number" || typeof cy !== "number") {
    return null
  }

  if (isEmbeddingChartPoint(point) && point.side === "right") {
    const halfSize = EMBEDDING_CROSS_HALF_SIZE
    return (
      <path
        d={`M ${cx - halfSize} ${cy - halfSize} L ${cx + halfSize} ${
          cy + halfSize
        } M ${cx + halfSize} ${cy - halfSize} L ${cx - halfSize} ${
          cy + halfSize
        }`}
        fill="none"
        opacity={EMBEDDING_POINT_OPACITY}
        stroke={color}
        strokeLinecap="round"
        strokeWidth={1.5}
      />
    )
  }

  return (
    <circle
      cx={cx}
      cy={cy}
      fill={color}
      fillOpacity={EMBEDDING_POINT_OPACITY}
      r={EMBEDDING_POINT_RADIUS}
    />
  )
}

function EmbeddingCircleShape(props: ScatterShapeProps & { fill?: string }) {
  const cx = props.cx
  const cy = props.cy

  if (typeof cx !== "number" || typeof cy !== "number") {
    return null
  }

  return (
    <circle
      cx={cx}
      cy={cy}
      fill={props.fill ?? "currentColor"}
      fillOpacity={EMBEDDING_POINT_OPACITY}
      r={EMBEDDING_POINT_RADIUS}
    />
  )
}

function ConfusionHeatmapChart({
  className,
  dataset,
  deltaLabel,
  matrix,
  title,
}: {
  className?: string
  dataset: CheckpointSummary["dataset"]
  deltaLabel?: string
  matrix: number[][]
  title: string
}) {
  const data = useMemo(
    () => confusionHeatmapData(matrix, dataset, Boolean(deltaLabel)),
    [dataset, deltaLabel, matrix]
  )
  const [tooltip, setTooltip] = useState<ConfusionHeatmapTooltipState | null>(
    null
  )
  const classCount = Math.max(
    CLASS_COUNT,
    matrix.length,
    ...matrix.map((row) => row.length)
  )
  const isDelta = Boolean(deltaLabel)

  const updateTooltip = useCallback(
    (event: PointerEvent<HTMLDivElement>, cell: ConfusionHeatmapDatum) => {
      setTooltip({
        cell,
        x: event.clientX,
        y: event.clientY,
      })
    },
    []
  )

  return (
    <ChartPanel
      bodyClassName="aspect-square h-auto w-full"
      className={cn("overflow-visible", className)}
      title={title}
    >
      <div className="relative h-full w-full">
        <div
          aria-label={title}
          className="grid h-full w-full overflow-hidden rounded-sm"
          role="img"
          style={{
            gridTemplateColumns: `repeat(${classCount}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${classCount}, minmax(0, 1fr))`,
          }}
        >
          {data.map((cell) => (
            <div
              aria-label={`${cell.trueClass} predicted ${cell.predictedClass}: ${heatmapCellLabel(cell.count, isDelta)}`}
              className="grid min-h-0 min-w-0 place-items-center overflow-hidden"
              key={`${cell.y}-${cell.x}`}
              role="img"
              style={{ backgroundColor: cell.fill }}
              onPointerEnter={(event) => updateTooltip(event, cell)}
              onPointerMove={(event) => updateTooltip(event, cell)}
              onPointerLeave={() => setTooltip(null)}
            >
              <span
                className="select-none truncate px-0.5 text-[0.64rem] leading-none font-semibold whitespace-nowrap text-white tabular-nums"
                style={{
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {heatmapCellLabel(cell.count, isDelta)}
              </span>
            </div>
          ))}
        </div>
        {tooltip ? (
          <div
            className="pointer-events-none fixed z-50"
            style={{
              left: tooltip.x,
              top: tooltip.y,
              transform: "translate(12px, 12px)",
            }}
          >
            <ConfusionHeatmapTooltip
              cell={tooltip.cell}
              deltaLabel={deltaLabel}
            />
          </div>
        ) : null}
      </div>
    </ChartPanel>
  )
}

function ConfusionHeatmapTooltip({
  cell,
  deltaLabel,
}: {
  cell: ConfusionHeatmapDatum
  deltaLabel?: string
}) {
  const isDelta = Boolean(deltaLabel)
  const valueClassName = isDelta
    ? cell.count > 0
      ? "text-blue-600 dark:text-blue-300"
      : cell.count < 0
        ? "text-red-600 dark:text-red-300"
        : "text-foreground"
    : "text-foreground"

  return (
    <div className="grid min-w-44 items-start gap-1 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl">
      <div className="grid gap-1">
        <div className="flex justify-between gap-4">
          <span className="text-muted-foreground">True</span>
          <span className="font-medium">{cell.trueClass}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-muted-foreground">Predicted</span>
          <span className="font-medium">{cell.predictedClass}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-muted-foreground">
            {isDelta ? "Difference" : "Count"}
          </span>
          <span className={cn("font-mono font-medium tabular-nums", valueClassName)}>
            {isDelta ? formatSignedInteger(cell.count) : formatInteger(cell.count)}
          </span>
        </div>
      </div>
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
    loss: readCssColor(styles, "--plot-loss", fallback.loss),
    mutationStep: readCssColor(
      styles,
      "--plot-mutation-step",
      fallback.mutationStep
    ),
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

const OVERVIEW_RUNTIME_LABELS = new Set([
  "Elapsed time",
  "Device",
  "Peak memory",
])

function analysisOverviewRows(
  report: AnalysisComparisonReport,
  rowHeaderLabel: string
): AnalysisTableRow[] {
  let rows = report.metadata.map(normalizeAnalysisTableRow)
  if (rowHeaderLabel === "Optimizer") {
    rows = rows.filter((row) => row.label !== "Optimizer")
  }

  rows = insertMissingAnalysisRows(
    rows,
    [
      {
        label: "Batch size",
        left: formatInteger(report.left.config.batch_size),
        right: formatInteger(report.right.config.batch_size),
      },
    ],
    "Seed"
  )

  const accuracyRows = [
    {
      label: "Validation accuracy",
      left: formatOptionalPercent(report.left.accuracy),
      right: formatOptionalPercent(report.right.accuracy),
    },
    {
      label: "Test accuracy",
      left: formatOptionalPercent(report.metrics.left.accuracy),
      right: formatOptionalPercent(report.metrics.right.accuracy),
    },
  ]
  rows = insertMissingAnalysisRows(rows, accuracyRows, "Steps")

  const runtimeRows = report.runtime.rows
    .map(normalizeAnalysisTableRow)
    .filter((row) => OVERVIEW_RUNTIME_LABELS.has(row.label))
  rows = insertMissingAnalysisRows(rows, runtimeRows, "Steps")

  return orderAnalysisOverviewRows(rows)
}

function normalizeAnalysisTableRow(row: AnalysisTableRow): AnalysisTableRow {
  return {
    ...row,
    label: normalizeAnalysisRowLabel(row.label),
  }
}

function normalizeAnalysisRowLabel(label: string): string {
  switch (label) {
    case "Step":
      return "Steps"
    case "Saved":
      return "Saved at"
    case "Elapsed":
      return "Elapsed time"
    case "Peak Memory":
      return "Peak memory"
    default:
      return label
  }
}

function insertMissingAnalysisRows(
  rows: AnalysisTableRow[],
  newRows: AnalysisTableRow[],
  afterLabel: string
): AnalysisTableRow[] {
  const existingLabels = new Set(rows.map((row) => row.label))
  const missingRows = newRows.filter((row) => !existingLabels.has(row.label))

  if (missingRows.length === 0) {
    return rows
  }

  const insertIndex = rows.findIndex((row) => row.label === afterLabel)
  if (insertIndex === -1) {
    return [...rows, ...missingRows]
  }

  const nextRows = [...rows]
  nextRows.splice(insertIndex + 1, 0, ...missingRows)
  return nextRows
}

const ANALYSIS_OVERVIEW_ROW_ORDER = [
  "Dataset",
  "Optimizer",
  "Seed",
  "Batch size",
  "Steps",
  "Elapsed time",
  "Validation accuracy",
  "Test accuracy",
  "Device",
  "Peak memory",
  "Saved at",
]

const ANALYSIS_OVERVIEW_ROW_RANK = new Map(
  ANALYSIS_OVERVIEW_ROW_ORDER.map((label, index) => [label, index])
)

function orderAnalysisOverviewRows(rows: AnalysisTableRow[]): AnalysisTableRow[] {
  const savedAtRank = ANALYSIS_OVERVIEW_ROW_RANK.get("Saved at") ?? 10_000

  return [...rows].sort((left, right) => {
    const leftRank =
      ANALYSIS_OVERVIEW_ROW_RANK.get(left.label) ??
      (left.label === "Saved at" ? savedAtRank : savedAtRank - 1)
    const rightRank =
      ANALYSIS_OVERVIEW_ROW_RANK.get(right.label) ??
      (right.label === "Saved at" ? savedAtRank : savedAtRank - 1)

    return leftRank - rightRank
  })
}

function checkpointSideLabels(
  left: CheckpointSummary,
  right: CheckpointSummary
): AnalysisSideLabels {
  if (left.optimizer !== right.optimizer) {
    return { left: left.optimizer, right: right.optimizer }
  }
  return { left: `${left.optimizer} 1`, right: `${right.optimizer} 2` }
}

function sideLabel(
  side: AnalysisSide,
  sideLabels?: AnalysisSideLabels
): string {
  return sideLabels?.[side] ?? (side === "left" ? "Model 1" : "Model 2")
}

function sideShortLabel(
  side: AnalysisSide,
  sideLabels?: AnalysisSideLabels
): string {
  const label = sideLabel(side, sideLabels)
  if (label === "Model 1") {
    return "1"
  }
  if (label === "Model 2") {
    return "2"
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

function composeStatusUpdates(
  first: ExperimentStatusUpdater,
  second: ExperimentStatusUpdater
): ExperimentStatusUpdater {
  return (current) => second(first(current))
}

function isExperimentStatusCompactEvent(
  value: unknown
): value is ExperimentStatusCompactEvent {
  return (
    typeof value === "object" &&
    value !== null &&
    ("status_patch" in value ||
      "history_delta" in value ||
      "replace_history" in value)
  )
}

function applyExperimentStatusCompactEvent(
  current: ExperimentStatus,
  event: ExperimentStatusCompactEvent
): ExperimentStatus {
  const baseHistory = event.replace_history
    ? emptyTrainingHistory()
    : current.history

  return {
    ...current,
    ...event.status_patch,
    history: applyTrainingHistoryDelta(baseHistory, event.history_delta),
  }
}

function emptyTrainingHistory(): TrainingHistory {
  return {
    loss: [],
    acc: [],
    train_acc: [],
    val_loss: [],
    memory_mb: [],
    mutation_step: [],
  }
}

function applyTrainingHistoryDelta(
  history: TrainingHistory,
  delta: TrainingHistoryDelta | null | undefined
): TrainingHistory {
  if (!delta) {
    return history
  }

  return {
    loss: mergeLossDelta(history.loss, delta.loss ?? []),
    acc: mergePointDelta(history.acc, delta.acc ?? []),
    train_acc: mergePointDelta(history.train_acc, delta.train_acc ?? []),
    val_loss: mergePointDelta(history.val_loss, delta.val_loss ?? []),
    memory_mb: mergePointDelta(history.memory_mb, delta.memory_mb ?? []),
    mutation_step: mergePointDelta(
      history.mutation_step,
      delta.mutation_step ?? []
    ),
  }
}

function mergeLossDelta(
  values: number[],
  points: { i: number; value: number }[]
): number[] {
  let nextValues: number[] | null = null

  for (const point of points) {
    if (!Number.isInteger(point.i) || point.i < 1) {
      continue
    }

    if (nextValues === null) {
      nextValues = [...values]
    }

    const index = point.i - 1
    if (index === nextValues.length) {
      nextValues.push(point.value)
    } else if (index < nextValues.length) {
      nextValues[index] = point.value
    } else {
      while (nextValues.length < index) {
        nextValues.push(Number.NaN)
      }
      nextValues.push(point.value)
    }
  }

  return nextValues ?? values
}

function mergePointDelta<T extends { i: number; value: number }>(
  values: T[],
  points: T[]
): T[] {
  let nextValues: T[] | null = null

  for (const point of points) {
    if (!Number.isInteger(point.i) || point.i < 0) {
      continue
    }

    if (nextValues === null) {
      nextValues = [...values]
    }

    const lastPoint: T | undefined = nextValues[nextValues.length - 1]
    if (!lastPoint || point.i > lastPoint.i) {
      nextValues.push(point)
    } else if (point.i === lastPoint.i) {
      nextValues[nextValues.length - 1] = point
    } else {
      const index = pointSeriesIndex(nextValues, point.i)
      if (index < nextValues.length && nextValues[index]?.i === point.i) {
        nextValues[index] = point
      } else {
        nextValues.splice(index, 0, point)
      }
    }
  }

  return nextValues ?? values
}

function pointSeriesIndex<T extends { i: number }>(points: T[], step: number): number {
  let low = 0
  let high = points.length

  while (low < high) {
    const mid = Math.floor((low + high) / 2)
    if (points[mid].i < step) {
      low = mid + 1
    } else {
      high = mid
    }
  }

  return low
}

function useTrainingTelemetryChartState({
  accuracyPoints,
  currentLoss,
  currentMutationStep,
  currentStep,
  losses,
  mutationStepPoints,
  runId,
  selectedInitialMutationStep,
  showMutationStepAxis,
}: {
  accuracyPoints: { i: number; value: number }[]
  currentLoss: number
  currentMutationStep: number | null | undefined
  currentStep: number
  losses: number[]
  mutationStepPoints: { i: number; value: number }[]
  runId: string | null | undefined
  selectedInitialMutationStep: number | null | undefined
  showMutationStepAxis: boolean
}): TrainingTelemetryChartState {
  const cacheRef = useRef<TrainingTelemetryCache | null>(null)
  const [chartState, setChartState] = useState<TrainingTelemetryChartState>(() =>
    emptyTrainingTelemetryChartState()
  )

  useEffect(() => {
    const cache =
      cacheRef.current ?? createTrainingTelemetryCache(runId, currentStep)
    if (cache.runId !== runId || currentStep < cache.currentStep) {
      resetTrainingTelemetryCache(cache, runId, currentStep)
    }
    cacheRef.current = cache

    setChartState(
      updateTrainingTelemetryChartCache(cache, {
        accuracyPoints,
        currentLoss,
        currentMutationStep,
        currentStep,
        losses,
        mutationStepPoints,
        selectedInitialMutationStep,
        showMutationStepAxis,
      })
    )
  }, [
    accuracyPoints,
    currentLoss,
    currentMutationStep,
    currentStep,
    losses,
    mutationStepPoints,
    runId,
    selectedInitialMutationStep,
    showMutationStepAxis,
  ])

  return chartState
}

function emptyTrainingTelemetryChartState(): TrainingTelemetryChartState {
  return {
    data: [],
    lossAxisUpperBound: 1,
    mutationStepAxisUpperBound: MUTATION_STEP_AXIS_MIN_UPPER_BOUND,
  }
}

function updateTrainingTelemetryChartCache(
  cache: TrainingTelemetryCache,
  {
    accuracyPoints,
    currentLoss,
    currentMutationStep,
    currentStep,
    losses,
    mutationStepPoints,
    selectedInitialMutationStep,
    showMutationStepAxis,
  }: {
    accuracyPoints: { i: number; value: number }[]
    currentLoss: number
    currentMutationStep: number | null | undefined
    currentStep: number
    losses: number[]
    mutationStepPoints: { i: number; value: number }[]
    selectedInitialMutationStep: number | null | undefined
    showMutationStepAxis: boolean
  }
): TrainingTelemetryChartState {
  const targetPointCount = trainingTelemetrySeriesPointLimit(showMutationStepAxis)
  updateIncrementalSeriesCache({
    cache: cache.series.loss,
    length: losses.length,
    maxX: Math.max(currentStep, losses.length),
    pointAt: (index) => ({ x: index + 1, y: losses[index] }),
    targetPointCount,
  })
  updateIncrementalSeriesCache({
    cache: cache.series.accuracy,
    length: accuracyPoints.length,
    maxX: currentStep,
    pointAt: (index) => accuracyPointToSeriesPoint(accuracyPoints[index]),
    targetPointCount,
  })
  updateIncrementalSeriesCache({
    cache: cache.series.mutationStep,
    length: mutationStepPoints.length,
    maxX: currentStep,
    pointAt: (index) => accuracyPointToSeriesPoint(mutationStepPoints[index]),
    targetPointCount,
  })
  cache.currentStep = currentStep

  const series = [
    {
      dataKey: "loss",
      points: sampledIncrementalSeriesPoints(cache.series.loss),
    },
    {
      dataKey: "accuracy",
      points: sampledIncrementalSeriesPoints(cache.series.accuracy),
    },
  ]

  if (showMutationStepAxis) {
    series.push({
      dataKey: "mutationStep",
      points: sampledIncrementalSeriesPoints(cache.series.mutationStep),
    })
  }

  return {
    data: mergeNumericSeries(series),
    lossAxisUpperBound: numericAxisUpperBoundFor(
      [cache.series.loss.maxY],
      currentLoss
    ),
    mutationStepAxisUpperBound: mutationStepAxisUpperBoundFor(
      [cache.series.mutationStep.maxY],
      currentMutationStep,
      selectedInitialMutationStep
    ),
  }
}

function createTrainingTelemetryCache(
  runId: string | null | undefined,
  currentStep: number
): TrainingTelemetryCache {
  return {
    currentStep,
    runId,
    series: {
      accuracy: createIncrementalSeriesCache(),
      loss: createIncrementalSeriesCache(),
      mutationStep: createIncrementalSeriesCache(),
    },
  }
}

function resetTrainingTelemetryCache(
  cache: TrainingTelemetryCache,
  runId: string | null | undefined,
  currentStep: number
) {
  cache.currentStep = currentStep
  cache.runId = runId
  resetIncrementalSeriesCache(cache.series.accuracy, 1)
  resetIncrementalSeriesCache(cache.series.loss, 1)
  resetIncrementalSeriesCache(cache.series.mutationStep, 1)
}

function createIncrementalSeriesCache(): IncrementalSeriesCache {
  return {
    bucketSize: 1,
    buckets: new Map(),
    maxY: 0,
    processedLength: 0,
  }
}

function trainingTelemetrySeriesPointLimit(
  showMutationStepAxis: boolean
): number {
  return Math.max(2, Math.floor(LINE_RENDER_POINT_LIMIT / (showMutationStepAxis ? 3 : 2)))
}

function updateIncrementalSeriesCache({
  cache,
  length,
  maxX,
  pointAt,
  targetPointCount,
}: {
  cache: IncrementalSeriesCache
  length: number
  maxX: number
  pointAt: (index: number) => NumericSeriesPoint | null
  targetPointCount: number
}) {
  const nextBucketSize = incrementalSeriesBucketSize(maxX, targetPointCount)
  if (length < cache.processedLength || nextBucketSize !== cache.bucketSize) {
    resetIncrementalSeriesCache(cache, nextBucketSize)
  }

  const startIndex = cache.processedLength > 0 ? cache.processedLength - 1 : 0
  for (let index = startIndex; index < length; index += 1) {
    const point = pointAt(index)
    if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
      upsertIncrementalSeriesPoint(cache, point)
    }
  }
  cache.processedLength = length
}

function incrementalSeriesBucketSize(
  maxX: number,
  targetPointCount: number
): number {
  const bucketCount = Math.max(1, Math.floor((targetPointCount - 2) / 2))
  const safeMaxX = Number.isFinite(maxX) && maxX > 0 ? maxX : 1
  return Math.max(1, Math.ceil(safeMaxX / bucketCount))
}

function resetIncrementalSeriesCache(
  cache: IncrementalSeriesCache,
  bucketSize: number
) {
  cache.bucketSize = bucketSize
  cache.buckets = new Map()
  cache.maxY = 0
  cache.processedLength = 0
}

function upsertIncrementalSeriesPoint(
  cache: IncrementalSeriesCache,
  point: NumericSeriesPoint
) {
  const bucketIndex = Math.floor(Math.max(0, point.x - 1) / cache.bucketSize)
  const bucket = cache.buckets.get(bucketIndex) ?? createNumericSeriesBucket()
  bucket.values.set(point.x, point.y)
  recomputeNumericSeriesBucket(bucket)
  cache.buckets.set(bucketIndex, bucket)
  cache.maxY = Math.max(cache.maxY, point.y)
}

function createNumericSeriesBucket(): NumericSeriesBucket {
  return {
    max: null,
    min: null,
    values: new Map(),
  }
}

function recomputeNumericSeriesBucket(bucket: NumericSeriesBucket) {
  let min: NumericSeriesPoint | null = null
  let max: NumericSeriesPoint | null = null

  for (const [x, y] of bucket.values) {
    if (!Number.isFinite(y)) {
      continue
    }
    const point = { x, y }
    if (
      min === null ||
      y < min.y ||
      (y === min.y && x < min.x)
    ) {
      min = point
    }
    if (
      max === null ||
      y > max.y ||
      (y === max.y && x > max.x)
    ) {
      max = point
    }
  }

  bucket.min = min
  bucket.max = max
}

function sampledIncrementalSeriesPoints(
  cache: IncrementalSeriesCache
): NumericSeriesPoint[] {
  const points: NumericSeriesPoint[] = []
  const bucketIndexes = [...cache.buckets.keys()].sort((left, right) => left - right)

  for (const bucketIndex of bucketIndexes) {
    const bucket = cache.buckets.get(bucketIndex)
    if (!bucket?.min || !bucket.max) {
      continue
    }
    if (bucket.min.x === bucket.max.x) {
      points.push(bucket.min)
    } else if (bucket.min.x < bucket.max.x) {
      points.push(bucket.min, bucket.max)
    } else {
      points.push(bucket.max, bucket.min)
    }
  }

  return points
}

function accuracyPointToSeriesPoint(
  point: { i: number; value: number } | undefined
): NumericSeriesPoint | null {
  if (!point || !Number.isFinite(point.i) || !Number.isFinite(point.value)) {
    return null
  }

  return { x: point.i, y: point.value }
}

function indexedNumberSeries(values: number[]): NumericSeriesPoint[] {
  return values.flatMap((value, index) =>
    Number.isFinite(value) ? [{ x: index + 1, y: value }] : []
  )
}

function accuracyPointSeries(
  points: { i: number; value: number }[]
): NumericSeriesPoint[] {
  return points.flatMap((point) =>
    Number.isFinite(point.i) && Number.isFinite(point.value)
      ? [{ x: point.i, y: point.value }]
      : []
  )
}

function downsampleNumericSeries(
  points: NumericSeriesPoint[],
  targetPointCount = LINE_RENDER_POINT_LIMIT
): NumericSeriesPoint[] {
  if (points.length <= targetPointCount) {
    return points
  }

  const selectedIndexes = new Set<number>([0, points.length - 1])
  const bucketCount = Math.max(1, Math.floor((targetPointCount - 2) / 2))
  const interiorCount = points.length - 2

  for (let bucketIndex = 0; bucketIndex < bucketCount; bucketIndex += 1) {
    const start = 1 + Math.floor((bucketIndex * interiorCount) / bucketCount)
    const end = 1 + Math.floor(((bucketIndex + 1) * interiorCount) / bucketCount)
    if (start >= end) {
      continue
    }

    let minIndex = start
    let maxIndex = start
    for (let index = start + 1; index < end; index += 1) {
      if (points[index].y < points[minIndex].y) {
        minIndex = index
      }
      if (points[index].y > points[maxIndex].y) {
        maxIndex = index
      }
    }
    selectedIndexes.add(minIndex)
    selectedIndexes.add(maxIndex)
  }

  return [...selectedIndexes]
    .sort((left, right) => left - right)
    .map((index) => points[index])
}

function mergeNumericSeries(
  series: { dataKey: string; points: NumericSeriesPoint[] }[]
): NumericChartDatum[] {
  const rows = new Map<number, NumericChartDatum>()

  for (const item of series) {
    for (const point of item.points) {
      const row = rows.get(point.x) ?? { x: point.x }
      row[item.dataKey] = point.y
      rows.set(point.x, row)
    }
  }

  return [...rows.values()].sort((left, right) => left.x - right.x)
}

function maxSeriesX(points: NumericSeriesPoint[]): number {
  return Math.max(0, ...points.map((point) => point.x))
}

function chartConfigFromSeries(series: ChartSeries[]): ChartConfig {
  return Object.fromEntries(
    series.map((item) => [
      item.dataKey,
      { label: item.label, color: item.color },
    ])
  ) as ChartConfig
}

function chartTooltipConfigFromSeries(
  series: ChartSeries[],
  formatters: Partial<Record<string, ChartValueFormatter>> = {}
): ChartTooltipConfig {
  return Object.fromEntries(
    series.map((item) => [
      item.dataKey,
      {
        color: item.color,
        formatter: formatters[item.dataKey],
        label: item.label,
      },
    ])
  )
}

function chartConfigFromLegendItems(items: ChartLegendItem[]): ChartConfig {
  return Object.fromEntries(
    items.map((item) => [item.label, { label: item.label, color: item.color }])
  ) as ChartConfig
}

function confusionHeatmapData(
  matrix: number[][],
  dataset: CheckpointSummary["dataset"],
  isDelta: boolean
): ConfusionHeatmapDatum[] {
  const maxMagnitude = Math.max(
    1,
    ...matrix.flatMap((row) =>
      row.map((value) => Math.abs(Number.isFinite(value) ? value : 0))
    )
  )

  return matrix.flatMap((row, trueIndex) =>
    row.map((value, predictedIndex) => {
      const count = Number.isFinite(value) ? value : 0
      return {
        count,
        fill: confusionHeatmapFill(count, maxMagnitude, isDelta),
        predictedClass: classLabelFor(dataset, predictedIndex),
        trueClass: classLabelFor(dataset, trueIndex),
        x: predictedIndex,
        y: trueIndex,
      }
    })
  )
}

function confusionHeatmapFill(
  value: number,
  maxMagnitude: number,
  isDelta: boolean
): string {
  if (isDelta) {
    if (value === 0) {
      return "rgba(115, 115, 115, 0.16)"
    }
    const alpha = 0.16 + 0.72 * Math.sqrt(Math.abs(value) / maxMagnitude)
    return value > 0
      ? `rgba(37, 99, 235, ${alpha})`
      : `rgba(220, 38, 38, ${alpha})`
  }

  const alpha = value <= 0 ? 0.08 : 0.16 + 0.72 * Math.sqrt(value / maxMagnitude)
  return `rgba(8, 145, 178, ${alpha})`
}

function heatmapCellLabel(value: number, isDelta: boolean): string {
  const integer = Math.trunc(value)
  if (!isDelta) {
    return String(integer)
  }
  if (integer > 0) {
    return `+${integer}`
  }
  return String(integer)
}

function calibrationChartData(report: AnalysisComparisonReport): NumericChartDatum[] {
  const rows = new Map<number, NumericChartDatum>()
  const setValue = (x: number, dataKey: string, value: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(value)) {
      return
    }
    const row = rows.get(x) ?? { x }
    row[dataKey] = value
    rows.set(x, row)
  }

  for (const bin of report.metrics.left.calibration.bins) {
    setValue(bin.confidence, "left", bin.accuracy)
  }
  for (const bin of report.metrics.right.calibration.bins) {
    setValue(bin.confidence, "right", bin.accuracy)
  }
  rows.set(0, { ...(rows.get(0) ?? { x: 0 }), ideal: 0 })
  rows.set(1, { ...(rows.get(1) ?? { x: 1 }), ideal: 1 })

  return [...rows.values()]
    .map((row) => ({ ...row, ideal: row.x }))
    .sort((left, right) => left.x - right.x)
}

type AnalysisEmbeddingPlotPoint =
  AnalysisEmbeddingProjection["left"][number] & {
    side: AnalysisSide
  }

function embeddingSidePoints(
  points: AnalysisEmbeddingProjection["left"],
  side: AnalysisSide
): AnalysisEmbeddingPlotPoint[] {
  return points.map((point) => ({ ...point, side }))
}

function embeddingProjectionPoints(
  projection: AnalysisEmbeddingProjection
): AnalysisEmbeddingPlotPoint[] {
  return [
    ...projection.left.map((point) => ({ ...point, side: "left" as const })),
    ...projection.right.map((point) => ({ ...point, side: "right" as const })),
  ]
}

function embeddingLegendItems(
  points: AnalysisEmbeddingPlotPoint[],
  dataset: CheckpointSummary["dataset"]
): ChartLegendItem[] {
  return Array.from(new Set(points.map((point) => point.label)))
    .sort((leftLabel, rightLabel) => leftLabel - rightLabel)
    .map((label) => ({
      color: embeddingClassColor(label),
      label: classLabelFor(dataset, label),
    }))
}

function groupedEmbeddingPoints(points: EmbeddingChartPoint[]): EmbeddingPointGroup[] {
  const grouped = new Map<number, EmbeddingChartPoint[]>()
  for (const point of points) {
    const group = grouped.get(point.label)
    if (group) {
      group.push(point)
    } else {
      grouped.set(point.label, [point])
    }
  }

  return [...grouped.entries()]
    .sort(([leftLabel], [rightLabel]) => leftLabel - rightLabel)
    .map(([label, groupPoints]) => ({
      color: embeddingClassColor(label),
      label: groupPoints[0]?.className ?? String(label),
      points: groupPoints,
    }))
}

function embeddingClassColor(label: number): string {
  return EMBEDDING_CLASS_COLORS[
    Math.abs(label) % EMBEDDING_CLASS_COLORS.length
  ]
}

function sampleEmbeddingPoints(
  points: AnalysisEmbeddingPlotPoint[],
  limit: number
): AnalysisEmbeddingPlotPoint[] {
  if (points.length <= limit) {
    return sortEmbeddingPoints(points)
  }

  const groups = new Map<string, AnalysisEmbeddingPlotPoint[]>()
  for (const point of points) {
    const key = embeddingSampleGroupKey(point)
    const group = groups.get(key)
    if (group) {
      group.push(point)
    } else {
      groups.set(key, [point])
    }
  }

  const sortedGroups = [...groups.entries()]
    .map(([key, groupPoints]) => ({
      key,
      points: sortEmbeddingPoints(groupPoints),
      quota: 0,
    }))
    .sort((left, right) => left.key.localeCompare(right.key))

  if (sortedGroups.length === 0 || limit <= 0) {
    return []
  }

  if (sortedGroups.length >= limit) {
    return sortEmbeddingPoints(
      sortedGroups
        .slice(0, limit)
        .flatMap((group) => evenlySpacedSample(group.points, 1))
    )
  }

  for (const group of sortedGroups) {
    group.quota = 1
  }

  let remaining = limit - sortedGroups.length
  const totalCapacity = sortedGroups.reduce(
    (sum, group) => sum + Math.max(0, group.points.length - 1),
    0
  )
  const remainders = sortedGroups.map((group) => {
    const capacity = Math.max(0, group.points.length - 1)
    const exactShare = totalCapacity > 0 ? (capacity / totalCapacity) * remaining : 0
    const extra = Math.min(capacity, Math.floor(exactShare))
    group.quota += extra
    return {
      capacity: capacity - extra,
      fraction: exactShare - Math.floor(exactShare),
      group,
    }
  })

  remaining = limit - sortedGroups.reduce((sum, group) => sum + group.quota, 0)
  for (const item of remainders
    .filter((item) => item.capacity > 0)
    .sort((left, right) => right.fraction - left.fraction)) {
    if (remaining <= 0) {
      break
    }
    item.group.quota += 1
    remaining -= 1
  }

  return sortEmbeddingPoints(
    sortedGroups.flatMap((group) =>
      evenlySpacedSample(group.points, group.quota)
    )
  )
}

function embeddingSampleGroupKey(point: AnalysisEmbeddingPlotPoint): string {
  return `${point.side}:${point.label}:${point.correct ? "1" : "0"}`
}

function sortEmbeddingPoints<T extends AnalysisEmbeddingPlotPoint>(points: T[]): T[] {
  return [...points].sort(
    (left, right) =>
      left.side.localeCompare(right.side) ||
      left.label - right.label ||
      Number(left.correct) - Number(right.correct) ||
      left.index - right.index
  )
}

function evenlySpacedSample<T>(items: T[], count: number): T[] {
  if (count <= 0) {
    return []
  }
  if (items.length <= count) {
    return items
  }
  if (count === 1) {
    return [items[Math.floor((items.length - 1) / 2)]]
  }

  const selected: T[] = []
  for (let index = 0; index < count; index += 1) {
    const itemIndex = Math.round((index * (items.length - 1)) / (count - 1))
    selected.push(items[itemIndex])
  }
  return selected
}

function paddedNumericDomain(values: number[]): [number, number] {
  const finiteValues = values.filter((value) => Number.isFinite(value))
  if (finiteValues.length === 0) {
    return [0, 1]
  }

  const min = Math.min(...finiteValues)
  const max = Math.max(...finiteValues)
  if (min === max) {
    const padding = Math.max(1, Math.abs(min) * EMBEDDING_DOMAIN_PADDING_RATIO)
    return [min - padding, max + padding]
  }

  const padding = (max - min) * EMBEDDING_DOMAIN_PADDING_RATIO
  return [min - padding, max + padding]
}

function tooltipDataKey(
  item: TooltipPayloadEntry<TooltipValueType, string | number>
): string {
  if (typeof item.dataKey === "string" || typeof item.dataKey === "number") {
    return String(item.dataKey)
  }
  if (typeof item.name === "string" || typeof item.name === "number") {
    return String(item.name)
  }
  return "value"
}

function formatTooltipValue(
  value: TooltipValueType | undefined,
  formatter?: ChartValueFormatter
): string {
  const numericValue = tooltipValueNumber(value)
  if (numericValue !== null) {
    return formatter ? formatter(numericValue) : formatCompactNumber(numericValue)
  }
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).join(", ")
  }
  return value === undefined ? MISSING_VALUE_LABEL : String(value)
}

function tooltipValueNumber(value: TooltipValueType | undefined): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string") {
    const numericValue = Number(value)
    return Number.isFinite(numericValue) ? numericValue : null
  }
  return null
}

function formatAxisInteger(value: unknown): string {
  const numericValue = typeof value === "number" ? value : Number(value)
  return Number.isFinite(numericValue) ? formatInteger(numericValue) : String(value)
}

function formatAxisCompact(value: unknown): string {
  const numericValue = typeof value === "number" ? value : Number(value)
  return Number.isFinite(numericValue)
    ? formatCompactNumber(numericValue)
    : String(value)
}

function compactCategoryTick(value: string): string {
  if (value.length <= 14) {
    return value
  }

  return `${value.slice(0, 12)}...`
}

function isEmbeddingChartPoint(value: unknown): value is EmbeddingChartPoint {
  if (typeof value !== "object" || value === null) {
    return false
  }
  const point = value as Partial<EmbeddingChartPoint>
  return (
    typeof point.x === "number" &&
    typeof point.y === "number" &&
    typeof point.label === "number" &&
    typeof point.prediction === "number" &&
    typeof point.className === "string" &&
    typeof point.predictionName === "string" &&
    typeof point.correctText === "string" &&
    typeof point.sideName === "string" &&
    (point.side === "left" || point.side === "right")
  )
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

function formatAnalysisTableRowValues(row: AnalysisTableRow): {
  left: string
  right: string
} {
  const normalizedLabel = normalizeAnalysisRowLabel(row.label)
  const includeYear =
    normalizedLabel === "Saved at"
      ? shouldIncludeAnalysisSavedAtYear(row.left, row.right)
      : undefined

  return {
    left: formatAnalysisTableValue(row.label, row.left, { includeYear }),
    right: formatAnalysisTableValue(row.label, row.right, { includeYear }),
  }
}

function formatAnalysisTableValue(
  label: string,
  value: string,
  options: { includeYear?: boolean } = {}
): string {
  const normalizedLabel = normalizeAnalysisRowLabel(label)

  if (isMissingText(value)) {
    return MISSING_VALUE_LABEL
  }

  if (normalizedLabel === "Steps") {
    const steps = integerFromText(value)
    return steps === null ? value : formatInteger(steps)
  }

  if (normalizedLabel === "Elapsed time") {
    const seconds = secondsFromText(value)
    return seconds === null ? value : formatDuration(seconds)
  }

  if (normalizedLabel === "Saved at") {
    return formatReadableDateTime(value, {
      compact: true,
      includeYear: options.includeYear,
    })
  }

  if (normalizedLabel === "Peak memory") {
    return formatPeakMemoryValue(value)
  }

  return value
}

function shouldIncludeAnalysisSavedAtYear(left: string, right: string): boolean {
  const currentYear = new Date().getFullYear()
  const leftYear = yearFromDateText(left)
  const rightYear = yearFromDateText(right)
  const hasNonCurrentYear =
    (leftYear !== null && leftYear !== currentYear) ||
    (rightYear !== null && rightYear !== currentYear)
  const hasDifferentYears =
    leftYear !== null && rightYear !== null && leftYear !== rightYear

  return hasNonCurrentYear || hasDifferentYears
}

function yearFromDateText(value: string): number | null {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return null
  }
  return date.getFullYear()
}

function formatPeakMemoryValue(value: string): string {
  const match = /^\s*([+-]?\d+(?:\.\d+)?)\s*(MB|GB)?\s*$/i.exec(value)
  if (!match) {
    return value
  }

  const amount = Number(match[1])
  if (!Number.isFinite(amount)) {
    return value
  }

  const unit = match[2]?.toUpperCase() ?? "MB"
  const memoryMb = unit === "GB" ? amount * 1024 : amount
  return formatMemoryMb(memoryMb)
}

function formatMemoryMb(memoryMb: number): string {
  if (!Number.isFinite(memoryMb)) {
    return MISSING_VALUE_LABEL
  }

  if (memoryMb >= 1024) {
    return `${formatSignificantNumber(memoryMb / 1024, 3)} GB`
  }

  return `${formatInteger(Math.round(memoryMb))} MB`
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

function integerFromText(value: string): number | null {
  const normalized = value.trim().replaceAll(",", "")
  if (!/^-?\d+$/.test(normalized)) {
    return null
  }

  const parsed = Number(normalized)
  return Number.isSafeInteger(parsed) ? parsed : null
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
  if (dataset === "fashion_mnist") {
    return "Fashion MNIST"
  }

  if (dataset === "cifar10") {
    return "CIFAR-10"
  }

  return dataset.toUpperCase()
}

function classLabelFor(dataset: CheckpointSummary["dataset"], label: number): string {
  if (dataset === "fashion_mnist") {
    return FASHION_MNIST_CLASS_LABELS[label] ?? String(label)
  }

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

function formatSignedInteger(value: number): string {
  const formatted = formatInteger(Math.abs(value))
  if (value > 0) {
    return `+${formatted}`
  }
  if (value < 0) {
    return `-${formatted}`
  }
  return formatted
}

function optimizerParamEntries(
  checkpoint: CheckpointSummary,
  schema: SchemaResponse
): [string, string, OptimizerParamValue][] {
  const params = checkpoint.optimizer_params[checkpoint.optimizer] ?? {}
  const fields = schema.optimizers_schema[checkpoint.optimizer] ?? []
  const fieldKeys = new Set(fields.map((field) => field.key))
  const schemaEntries: [string, string, OptimizerParamValue][] = []
  for (const field of fields) {
    const value = params[field.key] ?? field.default
    if (field.type === "boolean") {
      if (typeof value === "boolean") {
        schemaEntries.push([field.key, field.label, value])
      }
      continue
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      schemaEntries.push([field.key, field.label, value])
    }
  }
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
  options: { compact?: boolean; includeYear?: boolean } = {}
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

  const includeYear =
    options.includeYear ??
    (!options.compact || date.getFullYear() !== new Date().getFullYear())

  if (includeYear) {
    dateOptions.year = "numeric"
  }

  return date.toLocaleString(undefined, dateOptions)
}

function nextStepAxisUpperBound(step: number): number {
  const safeStep = Number.isFinite(step) && step > 0 ? step : 0
  return (Math.floor(safeStep / 10) + 1) * 10
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
    <div className="min-w-0 rounded-lg border px-4 py-3">
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
