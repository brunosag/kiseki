import { useEffect, useRef, useState } from "react";
import { fileURL, useViewTiming } from "../api";
import { loadNpy, type NpyArray } from "../npy";
import type { ReportDocument, Section } from "../types";
import { Scatter } from "./Scatter";
import { ErrorBox, Loading } from "./Common";
interface GalleryEntry {
  index: number;
  label: number;
  group: string;
  [label: string]: number | string | { prediction: number; confidence: number };
}
function useArrays(id: string, paths: string[]) {
  const key = paths.join("|");
  const [state, set] = useState<{
    key: string;
    data?: Record<string, NpyArray>;
    error?: string;
  }>({ key });
  useEffect(() => {
    const abort = new AbortController();
    Promise.all(
      paths.map(
        async (path) =>
          [path, await loadNpy(fileURL(id, path), abort.signal)] as const,
      ),
    ).then(
      (items) => set({ key, data: Object.fromEntries(items) }),
      (e) => {
        if (!abort.signal.aborted) set({ key, error: String(e) });
      },
    );
    return () => abort.abort();
  }, [id, key]);
  return state.key === key ? state : { key };
}
export function Embedding({
  id,
  section,
  report,
}: {
  id: string;
  section: Section;
  report: ReportDocument;
}) {
  const [subject, setSubject] = useState(report.subjects[0].label),
    [kind, setKind] = useState("tsne"),
    [colorBy, setColorBy] = useState<"label" | "prediction">("label");
  const data = section.data as {
      indices: string;
      labels?: string;
      [label: string]: string | Record<string, string> | undefined;
    },
    names = data[subject] as Record<string, string>,
    paths = [data.indices, names[kind], names.predictions, ...(data.labels ? [data.labels] : [])];
  const arrays = useArrays(id, paths);
  useViewTiming("embedding", !!arrays.data);
  return (
    <div className="interactive-view">
      <div className="chart-controls">
        <label>
          Model{" "}
          <select
            aria-label="Embedding model"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
          >
            {report.subjects.map((s) => (
              <option key={s.label}>{s.label}</option>
            ))}
          </select>
        </label>
        <div className="segmented">
          {["tsne", "pca"].map((k) => (
            <button
              key={k}
              className={kind === k ? "current" : ""}
              onClick={() => setKind(k)}
            >
              {k === "tsne" ? "t-SNE" : "PCA"}
            </button>
          ))}
        </div>
      </div>
      {data.labels ? (
        <div className="segmented">
          {(["label", "prediction"] as const).map((k) => (
            <button
              key={k}
              className={colorBy === k ? "current" : ""}
              onClick={() => setColorBy(k)}
            >
              {k === "label" ? "Color by class" : "Color by prediction"}
            </button>
          ))}
        </div>
      ) : (
        <p className="muted">
          Colored by prediction: this report was built before true classes
          were saved with embeddings. Rebuild it to color by class.
        </p>
      )}
      <ErrorBox error={arrays.error} />
      {arrays.data ? (
        <Scatter
          key={subject + kind}
          points={arrays.data[names[kind]]}
          predictions={arrays.data[names.predictions]}
          labels={data.labels ? arrays.data[data.labels] : undefined}
          indices={arrays.data[data.indices]}
          classNames={report.class_names}
          colorBy={data.labels ? colorBy : "prediction"}
        />
      ) : (
        !arrays.error && <Loading />
      )}
    </div>
  );
}
function ImageCanvas({
  array,
  index,
  heat = false,
  image,
  opacity = 1,
}: {
  array: NpyArray;
  index: number;
  heat?: boolean;
  image?: NpyArray;
  opacity?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const height = array.shape[1],
      width = array.shape[2];
    if (!height || !width) return;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d")!;
    const data = context.createImageData(width, height),
      channels = array.shape[3] ?? 1,
      offset = index * width * height * channels;
    for (let p = 0; p < width * height; p++) {
      const value = array.data[offset + p * channels];
      for (let c = 0; c < 3; c++) {
        let color: number;
        if (heat) {
          const v = Math.max(-1, Math.min(1, value)),
            colorHeat =
              v > 0
                ? c === 0
                  ? 255
                  : 255 * (1 - v)
                : c === 2
                  ? 255
                  : 255 * (1 + v);
          const channel = image?.shape[3] ?? 1;
          const base = image
            ? 255 *
              image.data[
                index * width * height * channel +
                  p * channel +
                  (channel === 1 ? 0 : c)
              ]
            : 255;
          color = base * (1 - opacity) + colorHeat * opacity;
        } else {
          color =
            255 * array.data[offset + p * channels + (channels === 1 ? 0 : c)];
        }
        data.data[p * 4 + c] = Math.round(color);
      }
      data.data[p * 4 + 3] = 255;
    }
    context.putImageData(data, 0, 0);
  }, [array, index, heat, image, opacity]);
  return (
    <canvas
      ref={ref}
      className="pixel-image"
      aria-label={heat ? "Relevance map" : "Input image"}
    />
  );
}
function GalleryContent({
  id,
  section,
  report,
}: {
  id: string;
  section: Section;
  report: ReportDocument;
}) {
  const data = section.data as {
      gallery: GalleryEntry[];
      arrays: Record<string, string | { gallery: string; classes: string }>;
    },
    [index, setIndex] = useState(0),
    [group, setGroup] = useState("all"),
    [opacity, setOpacity] = useState(0.75),
    [mode, setMode] = useState("gallery"),
    [classIndex, setClassIndex] = useState(0);
  const paths = [
      ...(mode === "gallery" ? [data.arrays.images as string] : []),
      ...report.subjects.map(
        (s) =>
          (data.arrays[s.label] as { gallery: string; classes: string })[
            mode === "gallery" ? "gallery" : "classes"
          ],
      ),
    ].filter(Boolean),
    arrays = useArrays(id, paths),
    entry = data.gallery[index];
  const available = data.gallery
    .map((e, i) => ({ e, i }))
    .filter(({ e }) => group === "all" || e.group === group);
  useViewTiming("lrp", !!arrays.data);
  if (!data.arrays || !Object.keys(data.arrays).length)
    return <p className="muted">No relevance arrays were requested.</p>;
  return (
    <div className="interactive-view">
      <div className="chart-controls">
        <div className="segmented">
          <button
            className={mode === "gallery" ? "current" : ""}
            onClick={() => setMode("gallery")}
          >
            Sample gallery
          </button>
          <button
            className={mode === "classes" ? "current" : ""}
            onClick={() => setMode("classes")}
          >
            Class averages
          </button>
        </div>
        {mode === "gallery" ? (
          <select
            aria-label="Gallery group"
            value={group}
            onChange={(e) => {
              setGroup(e.target.value);
              const i = data.gallery.findIndex(
                (row) =>
                  e.target.value === "all" || row.group === e.target.value,
              );
              if (i >= 0) setIndex(i);
            }}
          >
            <option value="all">All image groups</option>
            {[...new Set(data.gallery.map((e) => e.group))].map((g) => (
              <option key={g}>{g}</option>
            ))}
          </select>
        ) : (
          <select
            aria-label="LRP class"
            value={classIndex}
            onChange={(e) => setClassIndex(Number(e.target.value))}
          >
            {report.class_names.map((c, i) => (
              <option key={i} value={i}>
                {c}
              </option>
            ))}
          </select>
        )}
        <label>
          Relevance opacity{" "}
          <input
            aria-label="Relevance opacity"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
          />
        </label>
      </div>
      <p className="muted">
        Red supports the target class. Blue speaks against it.{" "}
        {mode === "gallery" && entry
          ? `Sample #${entry.index} · label ${report.class_names[entry.label]} · ${entry.group.replaceAll("_", " ")}`
          : "Mean relevance per class."}
      </p>
      <ErrorBox error={arrays.error} />
      {arrays.data ? (
        <>
          <div className="relevance-grid">
            {mode === "gallery" && entry && (
              <div>
                <h3>Input · {report.class_names[entry.label]}</h3>
                <ImageCanvas
                  array={arrays.data[data.arrays.images as string]}
                  index={index}
                />
              </div>
            )}
            {report.subjects.map((subject) => {
              const path = (
                  data.arrays[subject.label] as {
                    gallery: string;
                    classes: string;
                  }
                )[mode === "gallery" ? "gallery" : "classes"],
                array = arrays.data![path],
                prediction = entry?.[subject.label] as
                  { prediction: number; confidence: number } | undefined;
              return (
                <div key={subject.label}>
                  <h3>{subject.label}</h3>
                  {array.shape[0] >
                  (mode === "gallery" ? index : classIndex) ? (
                    <ImageCanvas
                      array={array}
                      index={mode === "gallery" ? index : classIndex}
                      heat
                      image={
                        mode === "gallery"
                          ? arrays.data![data.arrays.images as string]
                          : undefined
                      }
                      opacity={opacity}
                    />
                  ) : (
                    <p>No maps requested.</p>
                  )}
                  {mode === "gallery" && prediction && (
                    <small>
                      → {report.class_names[prediction.prediction]} ·{" "}
                      {(prediction.confidence * 100).toFixed(1)}% confidence
                    </small>
                  )}
                </div>
              );
            })}
          </div>
          {mode === "gallery" && (
            <div className="gallery-strip">
              {available.map(({ e, i }) => (
                <button
                  aria-label={`View sample ${e.index}`}
                  className={index === i ? "current" : ""}
                  key={i}
                  onClick={() => setIndex(i)}
                >
                  <ImageCanvas
                    array={arrays.data![data.arrays.images as string]}
                    index={i}
                  />
                  <small>{report.class_names[e.label]}</small>
                </button>
              ))}
            </div>
          )}
        </>
      ) : (
        !arrays.error && <Loading />
      )}
    </div>
  );
}

export function LrpGallery(props: {
  id: string;
  section: Section;
  report: ReportDocument;
}) {
  const data = props.section.data as {
    arrays?: Record<string, unknown>;
    gallery?: unknown[];
  };
  return !data.arrays || !Object.keys(data.arrays).length ? (
    <p className="padded muted">No relevance arrays were requested.</p>
  ) : (
    <GalleryContent {...props} />
  );
}
