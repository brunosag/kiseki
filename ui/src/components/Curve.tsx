import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { colors } from "../core";
export interface Line {
  label: string;
  x: number[];
  y: number[];
  color?: string;
  dash?: number[];
}
function plotData(lines: Line[], log: boolean) {
  return [
    null,
    ...lines.map((line) => {
      const x: number[] = [],
        y: number[] = [];
      for (let i = 0; i < line.x.length; i++)
        if (
          Number.isFinite(line.x[i]) &&
          Number.isFinite(line.y[i]) &&
          (!log || line.x[i] > 0)
        ) {
          x.push(line.x[i]);
          y.push(line.y[i]);
        }
      return [x, y];
    }),
  ] as unknown as uPlot.AlignedData;
}
// Mode 2 keeps independent x coordinates; data updates retain the instance and user zoom.
export function Curve({
  lines,
  accuracy = false,
  log = false,
  xLabel = "Step",
  height = 320,
}: {
  lines: Line[];
  accuracy?: boolean;
  log?: boolean;
  xLabel?: string;
  height?: number;
}) {
  const host = useRef<HTMLDivElement>(null),
    plot = useRef<uPlot | null>(null),
    zoomed = useRef(false);
  const signature = JSON.stringify(
    lines.map(({ label, color, dash }) => ({ label, color, dash })),
  );
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    zoomed.current = false;
    const instance = new uPlot(
      {
        mode: 2,
        width: el.clientWidth,
        height,
        padding: [20, 20, 0, 0],
        cursor: { drag: { x: true, y: false } },
        legend: { show: true },
        hooks: {
          setSelect: [
            (u) => {
              if (u.select.width > 0) zoomed.current = true;
            },
          ],
        },
        scales: {
          x: { time: false, distr: log ? 3 : 1 },
          // accuracy: fit the data (runs near 99% would look flat on 0–100%), within [0, 1]
          y: {
            range: accuracy
              ? (_u, min, max) => {
                  const pad = Math.max((max - min) * 0.08, 0.005);
                  return [Math.max(0, min - pad), Math.min(1, max + pad)];
                }
              : undefined,
          },
        },
        axes: [
          {
            label: xLabel,
            stroke: "#8e9698",
            grid: { stroke: "#283033", width: 1 },
            ticks: { stroke: "#353e40" },
            font: "11px system-ui",
            labelFont: "11px system-ui",
          },
          {
            stroke: "#8e9698",
            grid: { stroke: "#283033", width: 1 },
            ticks: { show: false },
            size: 62,
            font: "11px system-ui",
            values: (_u, values) => {
              // as many decimals as the tick spacing needs (0.25% steps: two)
              const step = Math.abs((values[1] ?? 1) - (values[0] ?? 0)) * 100;
              let decimals = 0;
              while (
                decimals < 3 &&
                Math.abs(step * 10 ** decimals - Math.round(step * 10 ** decimals)) > 1e-6
              )
                decimals++;
              return values.map((v) =>
                accuracy
                  ? `${(v * 100).toFixed(decimals)}%`
                  : Number(v.toPrecision(3)).toString(),
              );
            },
          },
        ],
        series: [
          {},
          ...lines.map((line, index) => ({
            label: line.label,
            stroke: line.color || colors[index % colors.length],
            width: 1.7,
            dash: line.dash,
            paths: uPlot.paths.linear!(),
            facets: [
              { scale: "x", auto: true },
              { scale: "y", auto: true },
            ],
            value: (_u: uPlot, value: number) =>
              value == null
                ? "—"
                : accuracy
                  ? `${(value * 100).toFixed(2)}%`
                  : value.toPrecision(4),
            points: { show: false },
          })),
        ],
      },
      plotData(lines, log),
      el,
    );
    plot.current = instance;
    const reset = () => {
      zoomed.current = false;
    };
    el.addEventListener("dblclick", reset);
    const observer = new ResizeObserver(() => {
      if (el.clientWidth > 0)
        instance.setSize({ width: el.clientWidth, height });
    });
    observer.observe(el);
    return () => {
      observer.disconnect();
      el.removeEventListener("dblclick", reset);
      instance.destroy();
      plot.current = null;
    };
  }, [signature, accuracy, log, xLabel, height]);
  useEffect(() => {
    plot.current?.setData(plotData(lines, log), !zoomed.current);
  }, [lines, log]);
  const hasSamples = lines.some((l) =>
    l.x.some(
      (x, i) =>
        Number.isFinite(x) && Number.isFinite(l.y[i]) && (!log || x > 0),
    ),
  );
  return (
    <div className="curve-wrap">
      <div className="curve" ref={host} />
      {!hasSamples && (
        <div className="chart-empty">
          No samples yet. Curves appear after the first validation.
        </div>
      )}
    </div>
  );
}
