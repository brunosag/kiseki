// Line charts of training curves on uPlot (canvas; thousands of points stay cheap).
// One y axis per chart. Hover shows a crosshair and every series' value at that x;
// drag zooms into a range, double-click zooms back out.
import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { accuracyRange } from "@/lib/curves";
import { count, duration } from "@/lib/format";
import { resolveColor, useDark } from "@/lib/theme";

export interface Series {
  label: string;
  color: string;
  x: (number | null)[];
  y: (number | null)[];
  /** Drawn thin and faint: context (training batches) behind the measure. */
  faint?: boolean;
}

export type XAxis = "step" | "time";
export type YAxis = "accuracy" | "loss";

/** Tick labels; `step` is the tick spacing, so sub-second ticks keep the digits that differ. */
const formatX = (axis: XAxis, value: number, step = 1) =>
  axis === "time"
    ? step < 1
      ? `${value.toFixed(Math.min(3, Math.ceil(-Math.log10(step))))}s`
      : duration(value)
    : value >= 1000
      ? `${count(value / 1000)}K`
      : count(value);
const formatY = (axis: YAxis, value: number, digits: number) =>
  axis === "accuracy" ? `${(value * 100).toFixed(digits)}%` : String(Number(value.toPrecision(3)));

/** Each series' pairs with a usable x, sorted by x (uPlot needs ascending x). */
function table(series: Series): [number[], (number | null)[]] {
  const pairs: [number, number | null][] = [];
  series.x.forEach((x, index) => x != null && pairs.push([x, series.y[index] ?? null]));
  pairs.sort((a, b) => a[0] - b[0]);
  return [pairs.map((pair) => pair[0]), pairs.map((pair) => pair[1])];
}

export function Chart({
  series,
  x,
  y,
  height = 280,
}: {
  series: Series[];
  x: XAxis;
  y: YAxis;
  height?: number;
}) {
  const box = useRef<HTMLDivElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const dark = useDark();
  const data = useMemo(() => uPlot.join(series.map(table) as uPlot.AlignedData[]), [series]);
  const shape = JSON.stringify([series.map((s) => [s.label, s.color, s.faint]), x, y, height, dark]);
  const latest = useRef({ series, data });
  latest.current = { series, data };

  useEffect(() => {
    const element = box.current!;
    const ink = resolveColor("var(--muted-foreground)");
    const grid = resolveColor("color-mix(in oklab, var(--foreground) 7%, transparent)");
    const font = "11px 'Inter Variable', sans-serif";
    const axis = (values: uPlot.Axis["values"], size: number, space = 40): uPlot.Axis => ({
      stroke: ink,
      font,
      size,
      space,
      gap: 6,
      grid: { stroke: grid, width: 1 },
      ticks: { show: false },
      border: { show: false },
      values,
    });
    const options: uPlot.Options = {
      width: element.clientWidth,
      height,
      legend: { show: false },
      padding: [8, 12, 0, 0],
      cursor: {
        y: false,
        drag: { x: true, y: false },
        points: { size: 7, width: 2, fill: resolveColor("var(--background)") },
      },
      scales: {
        x: { time: false },
        y:
          y === "accuracy"
            ? { range: (u) => accuracyRange(...measured(u, latest.current.series)) }
            : { distr: 3, log: 10 },
      },
      axes: [
        axis((_plot, splits) => splits.map((value) => formatX(x, value, (splits[1] ?? value + 1) - (splits[0] ?? value))), 28, 72),
        axis((_plot, splits) => {
          const span = (splits.at(-1) ?? 1) - (splits[0] ?? 0);
          return splits.map((value) => formatY(y, value, y === "accuracy" && span < 0.05 ? 1 : 0));
        }, 52),
      ],
      series: [
        {},
        ...series.map((s) => ({
          label: s.label,
          stroke: s.faint ? `color-mix(in oklab, ${s.color} 35%, transparent)` : s.color,
          width: s.faint ? 1 : 2,
          spanGaps: true,
          points: { show: false },
        })),
      ],
      hooks: { setCursor: [(u) => showTip(u, tip.current!, latest.current.series, x, y)] },
    };
    for (const s of options.series.slice(1)) s.stroke = resolveColor(s.stroke as string);
    plot.current = new uPlot(options, latest.current.data, element);
    const resize = new ResizeObserver(() => plot.current?.setSize({ width: element.clientWidth, height }));
    resize.observe(element);
    return () => {
      resize.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
  }, [shape]);

  useEffect(() => {
    plot.current?.setData(data);
  }, [data]);

  return (
    <div className="relative" style={{ height }}>
      <div ref={box} className="h-full w-full" />
      <div
        ref={tip}
        className="pointer-events-none absolute top-2 z-10 hidden min-w-36 rounded-lg border bg-popover px-2.5 py-2 text-xs shadow-lg/5"
      />
    </div>
  );
}

/** The y extent of the series that are not faint, within the visible x range: training
 * batches are context, and an SGD run's first noisy batches would flatten its curve. */
function measured(u: uPlot, series: Series[]): [number, number] {
  const xs = u.data[0] ?? [];
  const min = u.scales.x.min ?? -Infinity; // null until the x scale is first set
  const max = u.scales.x.max ?? Infinity;
  let low = Infinity;
  let high = -Infinity;
  series.forEach((s, i) => {
    const ys = u.data[i + 1]; // missing while new series wait for their setData
    if (!ys || (s.faint && !series.every((other) => other.faint))) return;
    for (let j = 0; j < xs.length; j++) {
      const value = ys[j];
      if (value == null || xs[j] < min || xs[j] > max) continue;
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
  });
  return [low, high];
}

/** The last value at or before index, for series sampled at other x positions. */
function valueAt(values: ArrayLike<number | null | undefined>, index: number): number | null {
  for (let i = index; i >= 0; i--) if (values[i] != null) return values[i] as number;
  return null;
}

function showTip(u: uPlot, tip: HTMLDivElement, series: Series[], x: XAxis, y: YAxis) {
  const index = u.cursor.idx;
  if (index == null || u.cursor.left == null || u.cursor.left < 0) {
    tip.style.display = "none";
    return;
  }
  const at = u.data[0][index];
  const header = document.createElement("div");
  header.className = "mb-1 text-muted-foreground";
  header.textContent = x === "time" ? duration(at) : `Step ${count(at)}`;
  const rows = series.map((s, i) => {
    const value = valueAt(u.data[i + 1], index);
    const row = document.createElement("div");
    row.className = "flex items-center gap-2 py-0.5";
    const key = document.createElement("span");
    key.className = "h-0.5 w-3 shrink-0 rounded-full";
    key.style.background = u.series[i + 1].stroke as string;
    const strong = document.createElement("span");
    strong.className = "font-medium tabular-nums text-foreground";
    strong.textContent = value == null ? "—" : formatY(y, value, y === "accuracy" ? 2 : 0);
    const label = document.createElement("span");
    label.className = "truncate text-muted-foreground";
    label.textContent = s.label;
    row.append(key, strong, label);
    return row;
  });
  tip.replaceChildren(header, ...rows);
  tip.style.display = "block";
  const left = u.cursor.left + u.over.offsetLeft;
  const flip = left > u.width - 200;
  tip.style.left = flip ? "" : `${left + 14}px`;
  tip.style.right = flip ? `${u.width - left + 14}px` : "";
}
