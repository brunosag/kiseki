import { forwardRef, useEffect, useMemo, useRef, useState } from "react"
import ReactPlotly from "react-plotly.js"
import katex from "katex"
import {
  ArrowDown,
  ArrowUpDown,
  Funnel,
  FolderOpen,
  LoaderCircle,
  Moon,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Sun,
  Trash2,
} from "lucide-react"
import type {
  ChangeEvent,
  ComponentPropsWithoutRef,
  ComponentType,
  KeyboardEvent,
  ReactElement,
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
import { Input } from "@/components/ui/input"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group"
import { Label } from "@/components/ui/label"
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
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { useTheme } from "@/components/theme-provider"
import {
  apiUrl,
  computeLrpAnalysis,
  computeTsneAnalysis,
  configDefaults,
  deleteCheckpointRun,
  defaultStatus,
  fallbackSchema,
  fetchCheckpoints,
  fetchSchema,
  fetchStatus,
  loadCheckpointStatus,
  type CheckpointSelection,
  type CheckpointListMode,
  type CheckpointSummary,
  optimizerParamDefaults,
  resetExperimentStatus,
  type ConfigField,
  type ExperimentConfig,
  type ExperimentStatus,
  type LrpAnalysisResponse,
  type LrpParams,
  type LrpSample,
  type OptimizerParams,
  type SchemaResponse,
  type SelectOption,
  type TsneAnalysisResponse,
  type TsneLearningRateMode,
  type TsneParams,
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

type ResolvedTheme = "dark" | "light"
type CheckpointSortKey = "saved_at" | "accuracy" | "step" | "elapsed"
type SortDirection = "asc" | "desc"
type CheckpointOptimizerFilter = ExperimentConfig["optimizer"] | "all"
type CheckpointDatasetFilter = ExperimentConfig["dataset"] | "all"
type AppTab = "training" | "analysis"
type AnalysisMethod = "tsne" | "lrp"
type AnalysisCardId = "left" | "right"
type AnalysisPlotStatus = "empty" | "loading" | "ready" | "error"
type AnalysisParams = TsneParams | LrpParams
type AnalysisResponse = TsneAnalysisResponse | LrpAnalysisResponse
type AnalysisLoadOptions = {
  method?: AnalysisMethod
  params?: AnalysisParams
}

type AnalysisCardState = {
  checkpoint: CheckpointSummary | null
  error: string | null
  method: AnalysisMethod | null
  requestId: number
  requestParams: AnalysisParams | null
  response: AnalysisResponse | null
  status: AnalysisPlotStatus
}

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
const ANALYSIS_CARD_IDS: AnalysisCardId[] = ["left", "right"]
const DEFAULT_TSNE_PARAMS: TsneParams = {
  perplexity: 30,
  max_iter: 1000,
  learning_rate_mode: "auto",
  learning_rate: 200,
  angle: 0.5,
  pca_components: 50,
  seed: null,
  use_pca: true,
}
const DEFAULT_LRP_PARAMS: LrpParams = {
  sample_count: 20,
  seed: null,
}

const defaultAnalysisCards: Record<AnalysisCardId, AnalysisCardState> = {
  left: {
    checkpoint: null,
    error: null,
    method: null,
    requestId: 0,
    requestParams: null,
    response: null,
    status: "empty",
  },
  right: {
    checkpoint: null,
    error: null,
    method: null,
    requestId: 0,
    requestParams: null,
    response: null,
    status: "empty",
  },
}

const ANALYSIS_CLASS_COLORS = [
  "#2563eb",
  "#0891b2",
  "#059669",
  "#65a30d",
  "#ca8a04",
  "#ea580c",
  "#dc2626",
  "#c026d3",
  "#7c3aed",
  "#475569",
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
const LRP_SEED_MAX = 2_147_483_647

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
  const [analysisMethod, setAnalysisMethod] = useState<AnalysisMethod>("tsne")
  const [tsneParams, setTsneParams] =
    useState<TsneParams>(DEFAULT_TSNE_PARAMS)
  const [lrpParams, setLrpParams] = useState<LrpParams>(() => ({
    ...DEFAULT_LRP_PARAMS,
    seed: randomAnalysisSeed(),
  }))
  const [analysisLockedClass, setAnalysisLockedClass] = useState<
    number | null
  >(null)
  const [analysisHoveredClass, setAnalysisHoveredClass] = useState<
    number | null
  >(null)
  const [analysisCards, setAnalysisCards] =
    useState<Record<AnalysisCardId, AnalysisCardState>>(defaultAnalysisCards)
  const analysisRequestCounter = useRef(0)

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
  const tsneValidationError = useMemo(
    () => validateTsneParams(tsneParams),
    [tsneParams]
  )
  const lrpValidationError = useMemo(
    () => validateLrpParams(lrpParams),
    [lrpParams]
  )
  const analysisValidationError =
    analysisMethod === "tsne" ? tsneValidationError : lrpValidationError
  const staleAnalysisCards = useMemo(
    () =>
      ANALYSIS_CARD_IDS.filter((cardId) =>
        isAnalysisCardStale(
          analysisCards[cardId],
          analysisMethod,
          currentAnalysisParams(analysisMethod, tsneParams, lrpParams)
        )
      ),
    [analysisCards, analysisMethod, lrpParams, tsneParams]
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

  function updateTsneParam<K extends keyof TsneParams>(
    key: K,
    value: TsneParams[K]
  ) {
    setTsneParams((current) => ({ ...current, [key]: value }))
  }

  function updateLrpParam<K extends keyof LrpParams>(
    key: K,
    value: LrpParams[K]
  ) {
    setLrpParams((current) => ({ ...current, [key]: value }))
  }

  const effectiveAnalysisClass = analysisHoveredClass ?? analysisLockedClass

  function toggleAnalysisClassLock(label: number) {
    setAnalysisLockedClass((current) => (current === label ? null : label))
  }

  async function loadAnalysisCheckpoint(
    cardId: AnalysisCardId,
    checkpoint: CheckpointSummary,
    options: AnalysisLoadOptions = {}
  ) {
    const method = options.method ?? analysisMethod
    let requestParams =
      options.params ?? currentAnalysisParams(method, tsneParams, lrpParams)
    if (method === "lrp" && (requestParams as LrpParams).seed == null) {
      const seed = randomAnalysisSeed()
      requestParams = { ...(requestParams as LrpParams), seed }
      setLrpParams((current) => ({ ...current, seed }))
    }
    const error =
      method === "tsne"
        ? validateTsneParams(requestParams as TsneParams)
        : validateLrpParams(requestParams as LrpParams)
    if (error) {
      setAnalysisCards((current) => ({
        ...current,
        [cardId]: {
          ...current[cardId],
          checkpoint,
          error,
          method,
          requestParams: null,
          response: null,
          status: "error",
        },
      }))
      return
    }

    const requestId = analysisRequestCounter.current + 1
    analysisRequestCounter.current = requestId
    const normalizedRequestParams =
      method === "tsne"
        ? tsneRequestParams(requestParams as TsneParams)
        : lrpRequestParams(requestParams as LrpParams)

    setAnalysisCards((current) => ({
      ...current,
      [cardId]: {
        ...current[cardId],
        checkpoint,
        error: null,
        method,
        requestId,
        requestParams: normalizedRequestParams,
        response: null,
        status: "loading",
      },
    }))

    try {
      const selection = selectionFromCheckpoint(checkpoint) ?? {
        run_id: checkpoint.run_id,
        kind: checkpoint.kind,
      }
      const response =
        method === "tsne"
          ? await computeTsneAnalysis(
              selection,
              normalizedRequestParams as TsneParams
            )
          : await computeLrpAnalysis(
              selection,
              normalizedRequestParams as LrpParams
            )
      setAnalysisCards((current) => {
        if (current[cardId].requestId !== requestId) {
          return current
        }

        return {
          ...current,
          [cardId]: {
            checkpoint: response.checkpoint,
            error: null,
            method,
            requestId,
            requestParams: normalizedRequestParams,
            response,
            status: "ready",
          },
        }
      })
    } catch {
      setAnalysisCards((current) => {
        if (current[cardId].requestId !== requestId) {
          return current
        }

        return {
          ...current,
          [cardId]: {
            ...current[cardId],
            checkpoint,
            error: `Failed to compute ${analysisMethodLabel(method)}`,
            response: null,
            method,
            requestParams: normalizedRequestParams,
            status: "error",
          },
        }
      })
    }
  }

  function recomputeLoadedAnalysisCards() {
    const options =
      analysisMethod === "lrp"
        ? {
            method: "lrp" as const,
            params: lrpRequestParams({
              ...lrpParams,
              seed: randomAnalysisSeed(),
            }),
          }
        : undefined
    if (options) {
      setLrpParams(options.params as LrpParams)
    }

    for (const cardId of ANALYSIS_CARD_IDS) {
      const checkpoint = analysisCards[cardId].checkpoint
      if (checkpoint) {
        void loadAnalysisCheckpoint(cardId, checkpoint, options)
      }
    }
  }

  async function newExperiment() {
    const nextStatus = await resetExperimentStatus()
    setLoadedCheckpoint(null)
    setConfig(readStoredConfig(schema))
    setOptParams(readStoredOptimizerParams(schema))
    setStatus(nextStatus)
  }

  return (
    <Tabs
      className="min-h-dvh gap-0 p-4"
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
          ) : (
            <Select
              value={analysisMethod}
              onValueChange={(value) =>
                setAnalysisMethod(value as AnalysisMethod)
              }
            >
              <SelectTrigger
                aria-label="Analysis method"
                className="h-9 w-36"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent
                align="start"
                className="w-(--radix-select-trigger-width) min-w-(--radix-select-trigger-width)"
                position="popper"
              >
                <SelectItem value="tsne">t-SNE</SelectItem>
                <SelectItem value="lrp">LRP</SelectItem>
              </SelectContent>
            </Select>
          )}
        </div>
        <TabsList>
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>
        <div className="flex justify-end">
          <ThemeToggle resolvedTheme={resolvedTheme} />
        </div>
      </div>
      <TabsContent className="mt-0" value="training">
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
              <Metric label="Step" value={formatInteger(status.current_step)} />
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
      </TabsContent>
      <TabsContent className="mt-0 flex min-h-0 flex-1" value="analysis">
        <AnalysisTab
          cards={analysisCards}
          focusedClass={effectiveAnalysisClass}
          lrpParams={lrpParams}
          lockedClass={analysisLockedClass}
          method={analysisMethod}
          pausedRunId={isPaused ? status.run_id : null}
          plotPalette={plotPalette}
          schema={schema}
          staleCardCount={staleAnalysisCards.length}
          tsneParams={tsneParams}
          validationError={analysisValidationError}
          onLoadCheckpoint={loadAnalysisCheckpoint}
          onRecomputeLoaded={recomputeLoadedAnalysisCards}
          onHoverClass={setAnalysisHoveredClass}
          onUpdateLrpParam={updateLrpParam}
          onToggleClass={toggleAnalysisClassLock}
          onUpdateTsneParam={updateTsneParam}
        />
      </TabsContent>
    </Tabs>
  )
}

type AnalysisTabProps = {
  cards: Record<AnalysisCardId, AnalysisCardState>
  focusedClass: number | null
  lrpParams: LrpParams
  lockedClass: number | null
  method: AnalysisMethod
  pausedRunId: string | null | undefined
  plotPalette: PlotPalette
  schema: SchemaResponse
  staleCardCount: number
  tsneParams: TsneParams
  validationError: string | null
  onLoadCheckpoint: (
    cardId: AnalysisCardId,
    checkpoint: CheckpointSummary
  ) => Promise<void>
  onHoverClass: (label: number | null) => void
  onRecomputeLoaded: () => void
  onToggleClass: (label: number) => void
  onUpdateLrpParam: <K extends keyof LrpParams>(
    key: K,
    value: LrpParams[K]
  ) => void
  onUpdateTsneParam: <K extends keyof TsneParams>(
    key: K,
    value: TsneParams[K]
  ) => void
}

function AnalysisTab({
  cards,
  focusedClass,
  lrpParams,
  lockedClass,
  method,
  pausedRunId,
  plotPalette,
  schema,
  staleCardCount,
  tsneParams,
  validationError,
  onLoadCheckpoint,
  onHoverClass,
  onRecomputeLoaded,
  onToggleClass,
  onUpdateLrpParam,
  onUpdateTsneParam,
}: AnalysisTabProps) {
  const loadedCardCount = ANALYSIS_CARD_IDS.filter(
    (cardId) => cards[cardId].checkpoint !== null
  ).length
  const hasLoadingCard = ANALYSIS_CARD_IDS.some(
    (cardId) =>
      cards[cardId].status === "loading" && cards[cardId].method === method
  )
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="mx-auto flex max-w-full shrink-0 flex-wrap items-center justify-center gap-3">
        <Card className="w-fit max-w-full" size="sm">
          <CardContent className="py-0">
            {method === "tsne" ? (
              <TsneControls
                hasLoadingCard={hasLoadingCard}
                loadedCardCount={loadedCardCount}
                staleCardCount={staleCardCount}
                tsneParams={tsneParams}
                validationError={validationError}
                onRecomputeLoaded={onRecomputeLoaded}
                onUpdateTsneParam={onUpdateTsneParam}
              />
            ) : (
              <LrpControls
                loadedCardCount={loadedCardCount}
                lrpParams={lrpParams}
                onRecomputeLoaded={onRecomputeLoaded}
                onUpdateLrpParam={onUpdateLrpParam}
              />
            )}
            {validationError ? (
              <div className="mt-3 rounded-lg border border-destructive/40 bg-destructive/10 p-2 text-sm text-destructive">
                {validationError}
              </div>
            ) : null}
          </CardContent>
        </Card>
        {method === "tsne" ? (
          <Card className="w-fit max-w-full" size="sm">
            <CardContent className="py-0">
              <AnalysisClassLegend
                focusedClass={focusedClass}
                lockedClass={lockedClass}
                onHoverClass={onHoverClass}
                onToggleClass={onToggleClass}
              />
            </CardContent>
          </Card>
        ) : null}
      </div>

      <div className="grid min-h-0 flex-1 auto-rows-fr items-stretch gap-4 xl:grid-cols-2">
        {ANALYSIS_CARD_IDS.map((cardId) => (
          <AnalysisComparisonCard
            cardId={cardId}
            key={cardId}
            method={method}
            pausedRunId={pausedRunId}
            focusedClass={focusedClass}
            plotPalette={plotPalette}
            schema={schema}
            state={cards[cardId]}
            onLoadCheckpoint={onLoadCheckpoint}
          />
        ))}
      </div>
    </div>
  )
}

function TsneControls({
  hasLoadingCard,
  loadedCardCount,
  staleCardCount,
  tsneParams,
  validationError,
  onRecomputeLoaded,
  onUpdateTsneParam,
}: {
  hasLoadingCard: boolean
  loadedCardCount: number
  staleCardCount: number
  tsneParams: TsneParams
  validationError: string | null
  onRecomputeLoaded: () => void
  onUpdateTsneParam: <K extends keyof TsneParams>(
    key: K,
    value: TsneParams[K]
  ) => void
}) {
  return (
    <div className="flex flex-wrap items-end justify-center gap-x-3 gap-y-2">
      <AnalysisNumberField
        className="w-24"
        label="Perplexity"
        max={50}
        min={5}
        step={1}
        value={tsneParams.perplexity}
        onChange={(value) => onUpdateTsneParam("perplexity", value)}
      />
      <AnalysisNumberField
        className="w-24"
        label="Max iter"
        min={250}
        step={50}
        value={tsneParams.max_iter}
        onChange={(value) => onUpdateTsneParam("max_iter", value)}
      />
      <AnalysisNumberField
        className="w-20"
        label="Angle"
        max={0.8}
        min={0.2}
        step={0.05}
        value={tsneParams.angle}
        onChange={(value) => onUpdateTsneParam("angle", value)}
      />
      <div className="grid w-28 gap-1">
        <Label className="text-xs leading-none">Learning rate</Label>
        <Select
          value={tsneParams.learning_rate_mode}
          onValueChange={(value) =>
            onUpdateTsneParam(
              "learning_rate_mode",
              value as TsneLearningRateMode
            )
          }
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent
            className="w-(--radix-select-trigger-width) min-w-(--radix-select-trigger-width)"
            position="popper"
          >
            <SelectItem value="auto">Auto</SelectItem>
            <SelectItem value="numeric">Numeric</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {tsneParams.learning_rate_mode === "numeric" ? (
        <AnalysisNumberField
          className="w-24"
          label="LR value"
          min={0}
          step={10}
          value={tsneParams.learning_rate ?? NaN}
          onChange={(value) =>
            onUpdateTsneParam(
              "learning_rate",
              Number.isFinite(value) ? value : null
            )
          }
        />
      ) : null}
      <div className="flex h-8 items-center gap-2 px-1">
        <Checkbox
          checked={tsneParams.use_pca}
          id="analysis-use-pca"
          onCheckedChange={(checked) =>
            onUpdateTsneParam("use_pca", checked === true)
          }
        />
        <Label
          className="whitespace-nowrap text-sm font-normal"
          htmlFor="analysis-use-pca"
        >
          Use PCA
        </Label>
      </div>
      {tsneParams.use_pca ? (
        <AnalysisNumberField
          className="w-20"
          label="PCA dims"
          max={120}
          min={2}
          step={1}
          value={tsneParams.pca_components}
          onChange={(value) => onUpdateTsneParam("pca_components", value)}
        />
      ) : null}
      {loadedCardCount > 0 ? (
        <Button
          className="h-8"
          disabled={
            staleCardCount === 0 ||
            hasLoadingCard ||
            validationError !== null
          }
          onClick={onRecomputeLoaded}
        >
          Recompute
        </Button>
      ) : null}
    </div>
  )
}

function LrpControls({
  loadedCardCount,
  lrpParams,
  onRecomputeLoaded,
  onUpdateLrpParam,
}: {
  loadedCardCount: number
  lrpParams: LrpParams
  onRecomputeLoaded: () => void
  onUpdateLrpParam: <K extends keyof LrpParams>(
    key: K,
    value: LrpParams[K]
  ) => void
}) {
  return (
    <div className="flex flex-wrap items-end justify-center gap-x-3 gap-y-2">
      <AnalysisNumberField
        className="w-28"
        label="Sample count"
        max={50}
        min={1}
        step={1}
        value={lrpParams.sample_count}
        onChange={(value) => onUpdateLrpParam("sample_count", value)}
      />
      {loadedCardCount > 0 ? (
        <Button className="h-8" onClick={onRecomputeLoaded}>
          Resample
        </Button>
      ) : null}
    </div>
  )
}

function AnalysisNumberField({
  className,
  disabled = false,
  label,
  max,
  min,
  step,
  value,
  onChange,
}: {
  className?: string
  disabled?: boolean
  label: string
  max?: number
  min?: number
  step: number
  value: number
  onChange: (value: number) => void
}) {
  return (
    <div className={cn("grid gap-1", className)}>
      <Label className="text-xs leading-none">{label}</Label>
      <Input
        className="h-8"
        disabled={disabled}
        max={max}
        min={min}
        step={step}
        type="number"
        value={Number.isFinite(value) ? value : ""}
        onChange={(event) =>
          onChange(
            event.currentTarget.value === ""
              ? NaN
              : Number(event.currentTarget.value)
          )
        }
      />
    </div>
  )
}

type AnalysisComparisonCardProps = {
  cardId: AnalysisCardId
  focusedClass: number | null
  method: AnalysisMethod
  pausedRunId: string | null | undefined
  plotPalette: PlotPalette
  schema: SchemaResponse
  state: AnalysisCardState
  onLoadCheckpoint: (
    cardId: AnalysisCardId,
    checkpoint: CheckpointSummary
  ) => Promise<void>
}

function AnalysisComparisonCard({
  cardId,
  focusedClass,
  method,
  pausedRunId,
  plotPalette,
  schema,
  state,
  onLoadCheckpoint,
}: AnalysisComparisonCardProps) {
  const hasCurrentMethodResult = state.method === method
  const currentResponse = hasCurrentMethodResult ? state.response : null
  const resultAccuracy = analysisAccuracyFor(currentResponse)
  const resultAccuracyLabel = method === "lrp" ? "Sample acc." : "Test acc."
  const isMethodStale =
    state.checkpoint !== null && state.method !== null && state.method !== method

  if (state.status === "empty") {
    return (
      <Card className="h-full min-h-[34rem] gap-0 [--card-spacing:--spacing(0)]">
        <CardContent className="flex min-h-0 flex-1 p-4">
          <CheckpointPicker
            closeOnLoadStart
            currentSelection={selectionFromCheckpoint(state.checkpoint)}
            disabled={false}
            mode="analysis"
            pausedRunId={pausedRunId}
            schema={schema}
            trigger={<AnalysisEmptyState method={method} />}
            onLoad={(checkpoint) => onLoadCheckpoint(cardId, checkpoint)}
          />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="h-full min-h-[34rem]">
      <CardHeader className="gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-xl">
            {state.checkpoint?.optimizer ?? "Select checkpoint"}
          </CardTitle>
          <CheckpointPicker
            closeOnLoadStart
            currentSelection={selectionFromCheckpoint(state.checkpoint)}
            disabled={state.status === "loading"}
            mode="analysis"
            pausedRunId={pausedRunId}
            schema={schema}
            onLoad={(checkpoint) => onLoadCheckpoint(cardId, checkpoint)}
          />
        </div>
        {state.checkpoint ? (
          <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
            <CheckpointMetric
              label="Val. acc."
              value={formatOptionalPercent(
                state.checkpoint.best_acc ?? state.checkpoint.accuracy
              )}
            />
            {state.status !== "loading" ? (
              <CheckpointMetric
                label={resultAccuracyLabel}
                value={formatOptionalPercent(resultAccuracy)}
              />
            ) : null}
            <CheckpointMetric
              label="Step"
              value={formatInteger(state.checkpoint.step)}
            />
            <CheckpointMetric
              label="Elapsed"
              value={formatOptionalDuration(
                state.checkpoint.total_elapsed_seconds
              )}
            />
          </div>
        ) : null}
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col">
        {isMethodStale && state.checkpoint ? (
          <AnalysisMethodPendingState
            method={method}
            onCompute={() => onLoadCheckpoint(cardId, state.checkpoint!)}
          />
        ) : null}
        {!isMethodStale && state.status === "loading" ? (
          <AnalysisLoadingState method={state.method ?? method} />
        ) : null}
        {!isMethodStale && state.status === "error" ? (
          <div className="grid min-h-80 flex-1 place-items-center rounded-lg border border-destructive/40 bg-destructive/10 p-6 text-center text-sm text-destructive">
            {state.error ??
              `Failed to compute ${analysisMethodLabel(state.method ?? method)}`}
          </div>
        ) : null}
        {!isMethodStale &&
        state.status === "ready" &&
        method === "tsne" &&
        isTsneResponse(state.response) ? (
          <AnalysisPlot
            focusedClass={focusedClass}
            response={state.response}
            plotPalette={plotPalette}
          />
        ) : null}
        {!isMethodStale &&
        state.status === "ready" &&
        method === "lrp" &&
        isLrpResponse(state.response) ? (
          <LrpGallery response={state.response} />
        ) : null}
      </CardContent>
    </Card>
  )
}

const AnalysisEmptyState = forwardRef<
  HTMLButtonElement,
  ComponentPropsWithoutRef<"button"> & { method: AnalysisMethod }
>(function AnalysisEmptyState({ className, method, ...props }, ref) {
  return (
    <button
      {...props}
      className={cn(
        "flex min-h-0 w-full flex-1 cursor-pointer rounded-xl border border-dashed bg-transparent text-center transition-colors hover:bg-muted/20 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none",
        className
      )}
      ref={ref}
      type="button"
    >
      <Empty className="min-h-80 flex-1 border-0 bg-transparent">
        <EmptyHeader>
          <EmptyMedia className="mb-3 size-12" variant="icon">
            <FolderOpen className="size-6" />
          </EmptyMedia>
          <EmptyTitle className="text-lg">Select checkpoint</EmptyTitle>
          <EmptyDescription className="mt-0 text-sm">
            {analysisEmptyDescription(method)}
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    </button>
  )
})

function AnalysisLoadingState({ method }: { method: AnalysisMethod }) {
  return (
    <div className="grid min-h-80 flex-1 place-items-center rounded-lg">
      <div className="grid justify-items-center gap-3 text-sm text-muted-foreground">
        <LoaderCircle className="size-5 animate-spin" />
        <span>Computing {analysisMethodLabel(method)}</span>
      </div>
    </div>
  )
}

function AnalysisMethodPendingState({
  method,
  onCompute,
}: {
  method: AnalysisMethod
  onCompute: () => void
}) {
  return (
    <div className="grid min-h-80 flex-1 place-items-center rounded-lg border border-dashed p-6 text-center">
      <div className="grid justify-items-center gap-3">
        <div className="text-sm font-medium">
          {analysisMethodLabel(method)} not computed
        </div>
        <Button className="h-8" onClick={onCompute}>
          Compute {analysisMethodLabel(method)}
        </Button>
      </div>
    </div>
  )
}

function AnalysisPlot({
  focusedClass,
  response,
  plotPalette,
}: {
  focusedClass: number | null
  response: TsneAnalysisResponse
  plotPalette: PlotPalette
}) {
  const { markerStates, revision } = useAnimatedClassMarkers(focusedClass)
  const data = useMemo(
    () => analysisPlotData(response, plotPalette, markerStates),
    [markerStates, plotPalette, response]
  )
  const layout = useMemo(
    () => analysisPlotLayout(plotPalette, response.checkpoint.run_id),
    [plotPalette, response.checkpoint.run_id]
  )

  return (
    <div className="flex min-h-80 flex-1 flex-col">
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
          layout={layout}
          revision={revision}
          style={{ height: "100%", width: "100%" }}
          useResizeHandler
        />
      </div>
    </div>
  )
}

function LrpGallery({ response }: { response: LrpAnalysisResponse }) {
  if (response.samples.length === 0) {
    return (
      <div className="grid min-h-80 flex-1 place-items-center rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
        No LRP samples returned.
      </div>
    )
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto pr-1">
      <div className="grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-3">
        {response.samples.map((sample) => {
          const label = classLabelFor(response.checkpoint.dataset, sample.label)
          const prediction = classLabelFor(
            response.checkpoint.dataset,
            sample.prediction
          )

          return (
            <div
              className={cn(
                "rounded-lg border bg-background p-2",
                sample.correct ? "border-border" : "border-destructive/50"
              )}
              key={sample.index}
            >
              <div className="aspect-square overflow-hidden rounded-md border bg-muted">
                <LrpSampleCanvas sample={sample} />
              </div>
              <div className="mt-2 flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="truncate text-sm leading-tight font-medium">
                    {label}
                  </div>
                  <div
                    className={cn(
                      "truncate text-xs",
                      sample.correct
                        ? "text-muted-foreground"
                        : "text-destructive"
                    )}
                  >
                    {prediction}
                  </div>
                </div>
                <div className="shrink-0 rounded-md bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                  #{sample.index}
                </div>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                <LrpSampleMetric
                  label="Score"
                  value={formatCompactNumber(sample.score)}
                />
                <LrpSampleMetric
                  label="Delta"
                  value={formatCompactNumber(sample.delta)}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function LrpSampleCanvas({ sample }: { sample: LrpSample }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    drawLrpSample(canvas, sample)
  }, [sample])

  return (
    <canvas
      aria-label={`LRP relevance overlay for sample ${sample.index}`}
      className="block h-full w-full"
      ref={canvasRef}
      style={{ imageRendering: "pixelated" }}
    />
  )
}

function LrpSampleMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="truncate font-medium tabular-nums">{value}</div>
      <div className="text-muted-foreground">{label}</div>
    </div>
  )
}

function drawLrpSample(canvas: HTMLCanvasElement, sample: LrpSample) {
  const height = sample.image.length
  const width = sample.image[0]?.length ?? 0
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
      const base = sample.image[y]?.[x] ?? [0, 0, 0]
      const relevance = clamp(sample.relevance[y]?.[x] ?? 0, -1, 1)
      const overlay = relevance >= 0 ? [239, 68, 68] : [37, 99, 235]
      const alpha = Math.abs(relevance) * 0.68

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

function AnalysisClassLegend({
  focusedClass,
  lockedClass,
  onHoverClass,
  onToggleClass,
}: {
  focusedClass: number | null
  lockedClass: number | null
  onHoverClass: (label: number | null) => void
  onToggleClass: (label: number) => void
}) {
  return (
    <div className="flex h-8 flex-wrap items-center justify-center gap-0">
      {ANALYSIS_CLASS_COLORS.map((color, label) => {
        const active = focusedClass === label
        const locked = lockedClass === label
        const dimmed = focusedClass !== null && !active

        return (
          <button
            aria-label={`Toggle class ${label}`}
            aria-pressed={active}
            className={cn(
              "inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-sm text-muted-foreground transition-[background-color,color,opacity] duration-200 hover:bg-muted/70 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none",
              locked ? "bg-muted text-foreground" : "",
              dimmed ? "opacity-45" : "opacity-100"
            )}
            key={label}
            type="button"
            onBlur={() => onHoverClass(null)}
            onClick={() => onToggleClass(label)}
            onFocus={() => onHoverClass(label)}
            onMouseEnter={() => onHoverClass(label)}
            onMouseLeave={() => onHoverClass(null)}
          >
            <span
              aria-hidden="true"
              className="size-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span>{label}</span>
          </button>
        )
      })}
    </div>
  )
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

function validateTsneParams(params: TsneParams): string | null {
  if (!isFiniteInRange(params.perplexity, 5, 50)) {
    return "Perplexity must be between 5 and 50"
  }

  if (!Number.isFinite(params.max_iter) || params.max_iter < 250) {
    return "Max iter must be at least 250"
  }

  if (!isFiniteInRange(params.angle, 0.2, 0.8)) {
    return "Angle must be between 0.2 and 0.8"
  }

  if (
    params.use_pca &&
    (!Number.isFinite(params.pca_components) ||
      params.pca_components < 2 ||
      params.pca_components > 120)
  ) {
    return "PCA dims must be between 2 and 120"
  }

  if (
    params.learning_rate_mode === "numeric" &&
    (!Number.isFinite(params.learning_rate) || Number(params.learning_rate) <= 0)
  ) {
    return "Learning rate must be positive"
  }

  if (
    params.seed !== null &&
    params.seed !== undefined &&
    !Number.isFinite(params.seed)
  ) {
    return "Seed must be a number"
  }

  return null
}

function validateLrpParams(params: LrpParams): string | null {
  if (
    !Number.isInteger(params.sample_count) ||
    params.sample_count < 1 ||
    params.sample_count > 50
  ) {
    return "Sample count must be an integer between 1 and 50"
  }

  if (
    params.seed !== null &&
    params.seed !== undefined &&
    (!Number.isInteger(params.seed) ||
      params.seed < 0 ||
      params.seed > LRP_SEED_MAX)
  ) {
    return "Seed must be an integer between 0 and 2147483647"
  }

  return null
}

function isFiniteInRange(value: number, min: number, max: number): boolean {
  return Number.isFinite(value) && value >= min && value <= max
}

function currentAnalysisParams(
  method: AnalysisMethod,
  tsneParams: TsneParams,
  lrpParams: LrpParams
): AnalysisParams {
  return method === "tsne" ? tsneParams : lrpParams
}

function tsneRequestParams(params: TsneParams): TsneParams {
  return {
    ...params,
    learning_rate:
      params.learning_rate_mode === "numeric" ? params.learning_rate : null,
    seed: params.seed ?? null,
  }
}

function lrpRequestParams(params: LrpParams): LrpParams {
  return {
    sample_count: params.sample_count,
    seed: params.seed ?? null,
  }
}

function randomAnalysisSeed(): number {
  const cryptoApi = globalThis.crypto
  if (cryptoApi) {
    const values = new Uint32Array(1)
    cryptoApi.getRandomValues(values)
    return values[0] & LRP_SEED_MAX
  }

  return Math.floor(Math.random() * (LRP_SEED_MAX + 1))
}

function isAnalysisCardStale(
  card: AnalysisCardState,
  method: AnalysisMethod,
  params: AnalysisParams
): boolean {
  if (
    !(
      (card.status === "ready" || card.status === "error") &&
      card.checkpoint !== null
    )
  ) {
    return false
  }

  if (card.method !== method) {
    return true
  }

  if (card.requestParams === null) {
    return true
  }

  if (method === "tsne") {
    return !sameTsneParams(card.requestParams as TsneParams, params as TsneParams)
  }

  return !sameLrpParams(card.requestParams as LrpParams, params as LrpParams)
}

function sameLrpParams(left: LrpParams, right: LrpParams): boolean {
  return (
    JSON.stringify(lrpRequestParams(left)) ===
    JSON.stringify(lrpRequestParams(right))
  )
}

function sameTsneParams(left: TsneParams, right: TsneParams): boolean {
  return (
    JSON.stringify(normalizedTsneParams(left)) ===
    JSON.stringify(normalizedTsneParams(right))
  )
}

function normalizedTsneParams(params: TsneParams): Record<string, unknown> {
  return {
    angle: params.angle,
    learning_rate:
      params.learning_rate_mode === "numeric" ? params.learning_rate : null,
    learning_rate_mode: params.learning_rate_mode,
    max_iter: params.max_iter,
    pca_components: params.use_pca ? params.pca_components : null,
    perplexity: params.perplexity,
    seed: params.seed ?? null,
    use_pca: params.use_pca,
  }
}

function isTsneResponse(
  response: AnalysisResponse | null
): response is TsneAnalysisResponse {
  return Boolean(response && "points" in response)
}

function isLrpResponse(
  response: AnalysisResponse | null
): response is LrpAnalysisResponse {
  return Boolean(response && "samples" in response)
}

function analysisMethodLabel(method: AnalysisMethod): string {
  return method === "tsne" ? "t-SNE" : "LRP"
}

function analysisEmptyDescription(method: AnalysisMethod): string {
  return method === "tsne"
    ? "t-SNE plots will appear here."
    : "LRP heatmaps will appear here."
}

type AnalysisClassMarkerState = {
  opacity: number
  size: number
}

type AnalysisClassMarkerAnimation = {
  markerStates: AnalysisClassMarkerState[]
  revision: number
}

const ACTIVE_CLASS_MARKER: AnalysisClassMarkerState = { opacity: 0.86, size: 4 }
const DIMMED_CLASS_MARKER: AnalysisClassMarkerState = { opacity: 0.06, size: 3.2 }
const CLASS_HIGHLIGHT_ANIMATION_MS = 220

function useAnimatedClassMarkers(
  focusedClass: number | null
): AnalysisClassMarkerAnimation {
  const [animation, setAnimation] = useState<AnalysisClassMarkerAnimation>(
    () => ({
      markerStates: targetClassMarkers(focusedClass),
      revision: 0,
    })
  )
  const markersRef = useRef(animation.markerStates)
  const revisionRef = useRef(animation.revision)

  useEffect(() => {
    markersRef.current = animation.markerStates
  }, [animation.markerStates])

  useEffect(() => {
    const start = markersRef.current
    const target = targetClassMarkers(focusedClass)
    let frameId = 0
    const startedAt = performance.now()

    function animateFrame(now: number) {
      const progress = Math.min(
        1,
        (now - startedAt) / CLASS_HIGHLIGHT_ANIMATION_MS
      )
      const easedProgress = easeInOutCubic(progress)
      const nextMarkers = target.map((targetMarker, index) => ({
        opacity: interpolate(
          start[index]?.opacity ?? targetMarker.opacity,
          targetMarker.opacity,
          easedProgress
        ),
        size: interpolate(
          start[index]?.size ?? targetMarker.size,
          targetMarker.size,
          easedProgress
        ),
      }))

      const nextRevision = revisionRef.current + 1

      markersRef.current = nextMarkers
      revisionRef.current = nextRevision
      setAnimation({
        markerStates: nextMarkers,
        revision: nextRevision,
      })

      if (progress < 1) {
        frameId = requestAnimationFrame(animateFrame)
      }
    }

    frameId = requestAnimationFrame(animateFrame)
    return () => cancelAnimationFrame(frameId)
  }, [focusedClass])

  return animation
}

function targetClassMarkers(focusedClass: number | null): AnalysisClassMarkerState[] {
  return ANALYSIS_CLASS_COLORS.map((_, label) =>
    focusedClass === null || focusedClass === label
      ? ACTIVE_CLASS_MARKER
      : DIMMED_CLASS_MARKER
  )
}

function interpolate(start: number, end: number, progress: number): number {
  return start + (end - start) * progress
}

function easeInOutCubic(progress: number): number {
  if (progress < 0.5) {
    return 4 * progress * progress * progress
  }

  return 1 - (-2 * progress + 2) ** 3 / 2
}

function analysisPlotData(
  response: TsneAnalysisResponse,
  plotPalette: PlotPalette,
  markerStates: AnalysisClassMarkerState[]
): Data[] {
  const points = response.points

  return ANALYSIS_CLASS_COLORS.map((color, label) => {
    const classPoints = points.filter((point) => point.label === label)
    const markerState = markerStates[label] ?? ACTIVE_CLASS_MARKER

    return {
      type: "scattergl",
      mode: "markers",
      name: String(label),
      uid: `${response.checkpoint.run_id}-${response.checkpoint.kind}-tsne-${label}`,
      x: classPoints.map((point) => point.x),
      y: classPoints.map((point) => point.y),
      customdata: classPoints.map((point) => [
        point.prediction,
        point.correct ? "correct" : "incorrect",
      ]),
      hoverlabel: {
        align: "left",
        bgcolor: plotPalette.hoverBackground,
        bordercolor: plotPalette.hoverBorder,
        font: { color: plotPalette.hoverText, size: 12 },
      },
      hovertemplate: `<b>Label ${label}</b><br>Prediction %{customdata[0]}<br>%{customdata[1]}<extra></extra>`,
      marker: {
        color,
        opacity: markerState.opacity,
        size: markerState.size,
      },
      showlegend: false,
    } satisfies Data
  })
}

function analysisPlotLayout(
  plotPalette: PlotPalette,
  runId: string
): Partial<Layout> {
  return {
    autosize: true,
    dragmode: "pan",
    font: { color: plotPalette.text, family: "Geist Variable, sans-serif" },
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
    margin: { b: 0, l: 0, r: 0, t: 0 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    showlegend: false,
    transition: {
      duration: 220,
      easing: "cubic-in-out",
    },
    uirevision: `analysis-tsne-${runId}`,
    xaxis: {
      automargin: false,
      fixedrange: false,
      visible: false,
      showgrid: false,
      showline: false,
      showticklabels: false,
      ticks: "",
      zeroline: false,
    },
    yaxis: {
      automargin: false,
      fixedrange: false,
      visible: false,
      showgrid: false,
      showline: false,
      showticklabels: false,
      ticks: "",
      zeroline: false,
    },
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

function formatOptionalPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a"
  }

  return `${value.toFixed(2)}%`
}

function analysisAccuracyFor(response: AnalysisResponse | null): number | null {
  const outcomes = isTsneResponse(response)
    ? response.points
    : isLrpResponse(response)
      ? response.samples
      : []
  if (outcomes.length === 0) {
    return null
  }

  const correctCount = outcomes.reduce(
    (total, item) => total + (item.correct ? 1 : 0),
    0
  )
  return (correctCount / outcomes.length) * 100
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

function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "n/a"
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
