import { useEffect, useRef, useState } from "react";
import type { NpyArray } from "../npy";
import { colors } from "../core";
export function Scatter({
  points,
  predictions,
  labels,
  indices,
  classNames,
  colorBy,
}: {
  points: NpyArray;
  predictions: NpyArray;
  labels?: NpyArray; // true classes; reports built before they were saved lack them
  indices: NpyArray;
  classNames: string[];
  colorBy: "label" | "prediction";
}) {
  const classes = colorBy === "label" && labels ? labels : predictions;
  const canvas = useRef<HTMLCanvasElement>(null),
    [error, setError] = useState(""),
    [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(
      null,
    ),
    [filter, setFilter] = useState(-1),
    [reset, setReset] = useState(0);
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const gl = el.getContext("webgl", {
      antialias: false,
      alpha: false,
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      setError("WebGL is unavailable. Open the saved figure below.");
      return;
    }
    let destroyed = false;
    function shader(type: number, source: string) {
      const s = gl!.createShader(type)!;
      gl!.shaderSource(s, source);
      gl!.compileShader(s);
      if (!gl!.getShaderParameter(s, gl!.COMPILE_STATUS))
        throw new Error(gl!.getShaderInfoLog(s) || "Shader compile failed");
      return s;
    }
    let program: WebGLProgram, vertex: WebGLShader, fragment: WebGLShader;
    try {
      vertex = shader(
        gl.VERTEX_SHADER,
        "attribute vec2 p; attribute vec3 color; attribute float category; uniform vec3 transform; uniform float pointSize; uniform float selected; varying vec3 c; varying float visible; void main(){gl_Position=vec4(p*transform.x+transform.yz,0.,1.); gl_PointSize=pointSize; c=color; visible=selected<0.||abs(category-selected)<.1?1.:.06;}",
      );
      fragment = shader(
        gl.FRAGMENT_SHADER,
        "precision mediump float; varying vec3 c; varying float visible; void main(){if(distance(gl_PointCoord,vec2(.5))>.5) discard; gl_FragColor=vec4(c,visible*.85);}",
      );
      program = gl.createProgram()!;
      gl.attachShader(program, vertex);
      gl.attachShader(program, fragment);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        throw new Error("Shader link failed");
    } catch (e) {
      setError(String(e));
      return;
    }
    const n = points.shape[0],
      norm = new Float32Array(n * 2);
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = points.data[i * 2],
        y = points.data[i * 2 + 1];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    const range = Math.max(maxX - minX, maxY - minY) || 1;
    const rgb = new Float32Array(n * 3),
      category = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      norm[i * 2] = ((points.data[i * 2] - (minX + maxX) / 2) / range) * 1.8;
      norm[i * 2 + 1] =
        ((points.data[i * 2 + 1] - (minY + maxY) / 2) / range) * 1.8;
      const k = classes.data[i];
      category[i] = k;
      const c = colors[k % colors.length] || "#999999";
      for (let j = 0; j < 3; j++)
        rgb[i * 3 + j] = parseInt(c.slice(1 + j * 2, 3 + j * 2), 16) / 255;
    }
    gl.useProgram(program);
    const buffers: WebGLBuffer[] = [];
    function attribute(name: string, data: Float32Array, size: number) {
      const buffer = gl!.createBuffer()!;
      buffers.push(buffer);
      gl!.bindBuffer(gl!.ARRAY_BUFFER, buffer);
      gl!.bufferData(gl!.ARRAY_BUFFER, data, gl!.STATIC_DRAW);
      const location = gl!.getAttribLocation(program, name);
      gl!.enableVertexAttribArray(location);
      gl!.vertexAttribPointer(location, size, gl!.FLOAT, false, 0, 0);
    }
    attribute("p", norm, 2);
    attribute("color", rgb, 3);
    attribute("category", category, 1);
    const transform = gl.getUniformLocation(program, "transform"),
      size = gl.getUniformLocation(program, "pointSize"),
      selected = gl.getUniformLocation(program, "selected");
    let zoom = 1,
      panX = 0,
      panY = 0,
      width = 1,
      height = 1,
      drag: { x: number; y: number; px: number; py: number } | null = null,
      frame = 0;
    function draw() {
      if (destroyed) return;
      gl!.viewport(0, 0, el!.width, el!.height);
      gl!.clearColor(0.065, 0.086, 0.094, 1);
      gl!.clear(gl!.COLOR_BUFFER_BIT);
      gl!.enable(gl!.BLEND);
      gl!.blendFunc(gl!.SRC_ALPHA, gl!.ONE_MINUS_SRC_ALPHA);
      gl!.uniform3f(transform, zoom, panX, panY);
      gl!.uniform1f(
        size,
        Math.max(3, Math.min(7, 3 * Math.sqrt(zoom))) *
          Math.min(devicePixelRatio, 2),
      );
      gl!.uniform1f(selected, filter);
      gl!.drawArrays(gl!.POINTS, 0, n);
    }
    const resize = new ResizeObserver(() => {
      width = el.clientWidth;
      height = el.clientHeight;
      const ratio = Math.min(devicePixelRatio, 2);
      el.width = width * ratio;
      el.height = height * ratio;
      draw();
    });
    resize.observe(el);
    function wheel(e: WheelEvent) {
      e.preventDefault();
      const r = el!.getBoundingClientRect(),
        x = ((e.clientX - r.left) / width) * 2 - 1,
        y = 1 - ((e.clientY - r.top) / height) * 2,
        old = zoom;
      zoom = Math.min(60, Math.max(0.5, zoom * Math.exp(-e.deltaY * 0.001)));
      panX = x - ((x - panX) * zoom) / old;
      panY = y - ((y - panY) * zoom) / old;
      draw();
      setHover(null);
    }
    function move(e: PointerEvent) {
      if (drag) {
        panX = drag.px + ((e.clientX - drag.x) / width) * 2;
        panY = drag.py - ((e.clientY - drag.y) / height) * 2;
        draw();
        setHover(null);
        return;
      }
      cancelAnimationFrame(frame);
      const r = el!.getBoundingClientRect(),
        mx = e.clientX - r.left,
        my = e.clientY - r.top;
      frame = requestAnimationFrame(() => {
        let nearest = -1,
          distance = 81;
        for (let i = 0; i < n; i++) {
          if (filter >= 0 && category[i] !== filter) continue;
          const dx = ((norm[2 * i] * zoom + panX + 1) * width) / 2 - mx,
            dy = ((1 - norm[2 * i + 1] * zoom - panY) * height) / 2 - my,
            d = dx * dx + dy * dy;
          if (d < distance) {
            distance = d;
            nearest = i;
          }
        }
        setHover(
          nearest < 0
            ? null
            : {
                i: nearest,
                x: Math.min(mx, width - 240),
                y: Math.max(8, my - 80),
              },
        );
      });
    }
    function down(e: PointerEvent) {
      drag = { x: e.clientX, y: e.clientY, px: panX, py: panY };
      el!.setPointerCapture(e.pointerId);
    }
    function up() {
      drag = null;
    }
    function leave() {
      if (!drag) setHover(null);
    }
    function lost(e: Event) {
      e.preventDefault();
      setError(
        "WebGL context lost. Reset the view to restore it, or use the saved figure.",
      );
    }
    el.addEventListener("wheel", wheel, { passive: false });
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointerleave", leave);
    el.addEventListener("webglcontextlost", lost);
    draw();
    return () => {
      destroyed = true;
      cancelAnimationFrame(frame);
      resize.disconnect();
      el.removeEventListener("wheel", wheel);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointerleave", leave);
      el.removeEventListener("webglcontextlost", lost);
      buffers.forEach((b) => gl.deleteBuffer(b));
      gl.deleteProgram(program);
      gl.deleteShader(vertex);
      gl.deleteShader(fragment);
    };
  }, [points, classes, filter, reset]);
  const index = hover ? indices.data[hover.i] : undefined,
    label = hover && labels ? labels.data[hover.i] : undefined;
  return (
    <>
      <div className="chart-controls">
        <span className="muted">
          {points.shape[0].toLocaleString()} points · WebGL · scroll to zoom ·
          drag to pan
        </span>
        <select
          aria-label="Filter class"
          value={filter}
          onChange={(e) => setFilter(Number(e.target.value))}
        >
          <option value={-1}>
            All {classes === labels ? "classes" : "predicted classes"}
          </option>
          {classNames.map((c, i) => (
            <option key={i} value={i}>
              {c}
            </option>
          ))}
        </select>
        <button
          onClick={() => {
            setError("");
            setReset((x) => x + 1);
          }}
        >
          Reset view
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="scatter-wrap">
        <canvas aria-label="Interactive embedding scatter plot" ref={canvas} />
        {hover && (
          <output
            className="scatter-tooltip"
            style={{ left: hover.x, top: hover.y }}
          >
            Sample #{index}
            <br />
            Class:{" "}
            {label === undefined
              ? "not saved in this report"
              : classNames[label]}
            <br />
            Prediction: {classNames[predictions.data[hover.i]]}
            {label !== undefined && label !== predictions.data[hover.i] && " ✗"}
          </output>
        )}
      </div>
      <div className="legend">
        {classNames.map((c, i) => (
          <span key={i}>
            <i style={{ background: colors[i % colors.length] }} />
            {c}
          </span>
        ))}
      </div>
    </>
  );
}
