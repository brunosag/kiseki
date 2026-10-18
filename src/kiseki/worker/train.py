"""Training a run: the compiled block, and the host loop around it.

The device runs a *block*: a while-loop of up to `count` steps, each drawing its minibatch,
stepping the optimizer and, every validation_interval steps (and at the last one),
validating the current network, keeping the best one and calling the optimizer's
validation hook. The loop ends early once validation reaches target_accuracy. Metrics go
into fixed-size device buffers the host reads once per block; `count` is a runtime
argument, so one compilation serves every block of the run.

The host sizes blocks to about BLOCK_SECONDS, cuts them at checkpoint boundaries, appends
metrics to the run's files, refreshes status.json, writes files on a background thread,
and between blocks honours stop and pause requests and SIGTERM.

Randomness is a pure function of (seed, step), so a resumed run continues exactly as if it
had never stopped (bit-exact with spec.deterministic).
"""

import math
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import msgspec
import numpy as np
from jax import lax

from kiseki import datasets, models, optimizers
from kiseki.spec import ModelRef, RunSpec, encode
from kiseki.store import Run, now, read_ndjson, write_atomic
from kiseki.worker import checkpoint, runtime
from kiseki.worker import data as device_data
from kiseki.worker.optimizer import Optimizer, Problem
from kiseki.worker.population import Scorer

MAX_BLOCK = 1024
BLOCK_SECONDS = 2.0
MEMORY_HEADROOM = 0.85  # share of device memory a compiled block may plan to use
LOG_SECONDS = 10.0

Carry = dict[str, Any]


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


# -- the compiled block ----------------------------------------------------------------------


@dataclass(frozen=True)
class Keys:
    data: jax.Array
    init: jax.Array
    step: jax.Array

    @classmethod
    def from_seed(cls, seed: int) -> Keys:
        root = jax.random.key(seed)
        return cls(*(jax.random.fold_in(root, i) for i in range(3)))


def initial_carry(opt: Optimizer, data: device_data.DeviceData, keys: Keys) -> Carry:
    state = opt.init(keys.init)
    return {
        "state": state,
        "step": jnp.int32(0),
        "order": data.order(keys.data, jnp.int32(0)),
        "best": opt.current(state),
        "best_accuracy": jnp.float32(-1.0),
        "best_step": jnp.int32(0),
        "reached": jnp.array(False),
    }


def metric_names(opt: Optimizer, carry: Carry) -> tuple[tuple[str, ...], tuple[str, ...]]:
    extra = jax.eval_shape(opt.validation_metrics, carry["state"])
    train = ("loss", "accuracy", *opt.extra_metrics)
    return train, ("val_loss", "val_accuracy", *extra)


def make_block(spec: RunSpec, opt: Optimizer, scorer: Scorer, keys: Keys, train_names, val_names):
    interval = spec.validation_interval
    target = spec.target_accuracy

    def block(carry: Carry, data: device_data.DeviceData, count, limit):
        buffers = {n: jnp.full(MAX_BLOCK, jnp.nan, jnp.float32) for n in (*train_names, *val_names)}

        def body(loop):
            i, carry, buffers = loop
            step = carry["step"]
            order = lax.cond(
                step % data.batches_per_epoch == 0,
                lambda: data.order(keys.data, step),
                lambda: carry["order"],
            )
            x, y = data.batch(order, step)
            state, metrics = opt.step(carry["state"], jax.random.fold_in(keys.step, step), x, y)
            step = step + 1

            def validate(state):
                flat = opt.current(state)
                loss, accuracy = scorer.evaluate(flat, data.val_x, data.val_y)
                improved = accuracy > carry["best_accuracy"]
                state = opt.on_validation(state, improved)
                values = {
                    "val_loss": loss,
                    "val_accuracy": accuracy,
                    **opt.validation_metrics(state),
                }
                best = (
                    jnp.where(improved, flat, carry["best"]),
                    jnp.where(improved, accuracy, carry["best_accuracy"]),
                    jnp.where(improved, step, carry["best_step"]),
                )
                return state, {k: jnp.float32(v) for k, v in values.items()}, best

            def skip(state):
                values = {name: jnp.float32(jnp.nan) for name in val_names}
                return state, values, (carry["best"], carry["best_accuracy"], carry["best_step"])

            due = (step % interval == 0) | (step == limit)
            state, values, (best, best_accuracy, best_step) = lax.cond(due, validate, skip, state)
            reached = jnp.array(False) if target is None else best_accuracy >= target
            for name in train_names:
                buffers[name] = buffers[name].at[i].set(jnp.float32(metrics[name]))
            for name in val_names:
                buffers[name] = buffers[name].at[i].set(values[name])
            carry = {
                "state": state,
                "step": step,
                "order": order,
                "best": best,
                "best_accuracy": best_accuracy,
                "best_step": best_step,
                "reached": reached,
            }
            return i + 1, carry, buffers

        def cond(loop):
            i, carry, _ = loop
            return (i < count) & (carry["step"] < limit) & ~carry["reached"]

        executed, carry, buffers = lax.while_loop(cond, body, (jnp.int32(0), carry, buffers))
        return carry, buffers, executed

    return block


@dataclass
class Program:
    """Everything compiled for one run at one chunk size."""

    chunk: int
    opt: Optimizer
    scorer: Scorer
    block: Any
    init: Any
    evaluate: Any
    train_names: tuple[str, ...]
    val_names: tuple[str, ...]
    planned_bytes: int
    compile_seconds: float


def device_limit() -> int:
    stats = jax.devices()[0].memory_stats() or {}
    return int(stats.get("bytes_limit", 0))


def first_chunk(net, spec: RunSpec, population: int, limit: int) -> int:
    """A chunk whose activations use about 30% of device memory (1 to 256 networks)."""
    if spec.chunk_size:
        return min(spec.chunk_size, population)
    if not limit:
        return min(population, 64)
    flat = jax.ShapeDtypeStruct((net.size,), jnp.float32)
    images = jax.ShapeDtypeStruct((spec.batch_size, *net.input_shape), jnp.float32)
    taps = jax.eval_shape(lambda f, x: net.forward(f, x, taps=net.names)[1], flat, images)
    itemsize = 2 if spec.precision == "bf16" else 4
    per_network = itemsize * sum(math.prod(t.shape) for t in taps.values())
    chunk = 2 ** int(math.log2(max(1.0, 0.3 * limit / per_network)))
    return max(1, min(chunk, 256, population))


def starting_network(spec: RunSpec, net) -> jax.Array | None:
    """The saved network `spec.start_from` names, checked to share this run's dataset, model
    and train/validation split (else its held-out images could be training ones here)."""
    if not spec.start_from:
        return None
    ref = ModelRef.parse(spec.start_from)
    source = Run.find(ref.run_id).spec()
    for field in ("dataset", "model", "split_seed", "validation_size"):
        if getattr(source, field) != getattr(spec, field):
            raise ValueError(
                f"start_from {ref}: its {field} is {getattr(source, field)!r}, "
                f"this run's is {getattr(spec, field)!r}"
            )
    weights = checkpoint.load_network(Run.find(ref.run_id).models_dir / f"{ref.name}.safetensors")
    if weights.shape != (net.size,):
        raise ValueError(f"start_from {ref}: {weights.shape[0]} weights, the model has {net.size}")
    return jnp.asarray(weights)


def compile_program(spec, net, impl, params, data, keys: Keys, chunk: int, start=None) -> Program:
    started = time.perf_counter()
    dtype = jnp.bfloat16 if spec.precision == "bf16" else jnp.float32
    scorer = Scorer(net, chunk, dtype)
    opt = impl.build(params, Problem(net, scorer, spec.batch_size, start))
    init = jax.jit(lambda d: initial_carry(opt, d, keys))
    carry_shape = jax.eval_shape(init, data)
    train_names, val_names = metric_names(opt, carry_shape)
    block = jax.jit(make_block(spec, opt, scorer, keys, train_names, val_names), donate_argnums=0)
    scalar = jax.ShapeDtypeStruct((), jnp.int32)
    compiled = block.lower(carry_shape, data, scalar, scalar).compile()
    memory = compiled.memory_analysis()
    planned = 0
    if memory is not None:
        planned = (
            memory.argument_size_in_bytes
            + memory.output_size_in_bytes
            - memory.alias_size_in_bytes
            + memory.temp_size_in_bytes
        )
    evaluate = jax.jit(scorer.evaluate)
    return Program(
        chunk=chunk,
        opt=opt,
        scorer=scorer,
        block=compiled,
        init=init,
        evaluate=evaluate,
        train_names=train_names,
        val_names=val_names,
        planned_bytes=int(planned),
        compile_seconds=time.perf_counter() - started,
    )


# -- the host loop ---------------------------------------------------------------------------


class Interrupted:
    def __init__(self) -> None:
        self.signal: int | None = None
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, number, frame) -> None:
        self.signal = number


def truncate_rows(run: Run, name: str, last_step: int) -> None:
    """Drop rows past `last_step` (written after the checkpoint a run resumes from)."""
    path = run.path / f"{name}.ndjson"
    if not path.exists():
        return
    rows, end = read_ndjson(path)
    torn = end < path.stat().st_size  # a partial last line from a killed worker
    if torn or (rows and rows[-1]["step"] > last_step):
        kept = [row for row in rows if row["step"] <= last_step]
        write_atomic(path, b"".join(msgspec.json.encode(row) + b"\n" for row in kept))


def train(run: Run) -> int:
    spec = run.spec()
    status = run.status()
    if status.state == "completed":
        log("run already completed")
        return 0
    request = run.pending_request()
    if request:  # stopped or paused while the runner was starting this worker
        run.clear_request()
        state = "stopped" if request == "stop" else "paused"
        run.update_status(state=state, pid=None)
        run.event(state, step=status.step)
        log(f"{state} before starting")
        return 0
    interrupted = Interrupted()
    info = datasets.info(spec.dataset)
    net = models.load(spec.model, info.image_shape, len(info.class_names))
    params = msgspec.convert(spec.params, optimizers.params_type(spec.optimizer))
    impl = optimizers.load_impl(spec.optimizer)
    population = int(getattr(params, "population_size", 1))
    assert spec.validation_size is not None
    loading = time.perf_counter()
    data = device_data.load(
        spec.dataset,
        validation_size=spec.validation_size,
        split_seed=spec.split_seed,
        batch_size=spec.batch_size,
    )
    keys = Keys.from_seed(spec.seed)
    device = runtime.device_summary()
    log(
        f"{spec.dataset}/{spec.model}/{spec.optimizer} on {device['device']}; "
        f"{net.size} weights, data in {time.perf_counter() - loading:.1f}s"
    )
    manifest = checkpoint.load_manifest(run.state_dir)
    resumed_from = manifest.step if manifest else 0
    for name in ("train", "val"):
        truncate_rows(run, name, resumed_from)
    elapsed_before = manifest.elapsed_seconds if manifest else 0.0
    run.update_status(
        state="running",
        pid=os.getpid(),
        device=str(device["device"]),
        started_at=status.started_at or now(),
        error=None,
        step=resumed_from,
    )
    run.event("started" if not manifest else "resumed", step=resumed_from, device=device)

    start = starting_network(spec, net)
    limit = device_limit()
    chunk = first_chunk(net, spec, population, limit)
    while True:  # compile, then run one step; halve the chunk until it fits
        program = compile_program(spec, net, impl, params, data, keys, chunk, start)
        log(
            f"compiled in {program.compile_seconds:.1f}s: chunk {chunk}, "
            f"plans {program.planned_bytes / 2**30:.2f} of {limit / 2**30:.2f} GiB"
        )
        if (
            limit
            and program.planned_bytes > MEMORY_HEADROOM * limit
            and chunk > 1
            and not spec.chunk_size
        ):
            chunk //= 2
            continue
        if manifest:
            like = jax.eval_shape(program.init, data)
            carry = jax.device_put(
                checkpoint.unflatten(like, checkpoint.load_state(run.state_dir, manifest))
            )
        else:
            carry = program.init(data)
        try:
            first = run_block(program, carry, data, 1, spec.iterations)
            break
        except jax.errors.JaxRuntimeError as error:
            if "RESOURCE_EXHAUSTED" not in str(error) or chunk == 1 or spec.chunk_size:
                raise
            log(f"out of memory at chunk {chunk}; halving")
            chunk //= 2
    run.event(
        "compiled",
        chunk_size=chunk,
        seconds=round(program.compile_seconds, 2),
        planned_bytes=program.planned_bytes,
        device_bytes=limit,
    )
    run.update_status(chunk_size=chunk)
    loop = Loop(run, spec, program, data, resumed_from, elapsed_before)
    return loop.execute(first, interrupted)


def run_block(program: Program, carry: Carry, data, count: int, limit: int):
    started = time.perf_counter()
    carry, buffers, executed = program.block(carry, data, jnp.int32(count), jnp.int32(limit))
    host = jax.device_get(
        {
            "buffers": buffers,
            "executed": executed,
            "step": carry["step"],
            "best_accuracy": carry["best_accuracy"],
            "best_step": carry["best_step"],
            "reached": carry["reached"],
        }
    )
    return carry, host, time.perf_counter() - started


class Loop:
    def __init__(self, run: Run, spec: RunSpec, program: Program, data, step: int, elapsed: float):
        self.run, self.spec, self.program, self.data = run, spec, program, data
        self.step = step
        self.elapsed = elapsed
        self.writer = checkpoint.Writer()
        self.rate: float | None = None
        self.saved_best_step = -1
        self.last_log = 0.0
        self.best_accuracy: float | None = None
        self.best_step: int | None = None

    def execute(self, first, interrupted: Interrupted) -> int:
        spec = self.spec
        carry, host, seconds = first
        count = 1
        try:
            while True:
                self.record(host, seconds)
                carry_step = int(host["step"])
                done = carry_step >= spec.iterations or bool(host["reached"])
                request = self.run.pending_request()
                if interrupted.signal is not None:
                    request = request or "pause"
                at_checkpoint = carry_step % spec.checkpoint_interval == 0
                if host["best_step"] != self.saved_best_step and host["best_step"] > 0:
                    self.save_network(
                        carry["best"],
                        "best",
                        int(host["best_step"]),
                        val_accuracy=float(host["best_accuracy"]),
                    )
                    self.saved_best_step = int(host["best_step"])
                if at_checkpoint and carry_step > 0:
                    self.save_network(
                        self.program.opt.current(carry["state"]), f"step-{carry_step}", carry_step
                    )
                if done or request or at_checkpoint:
                    self.save_state(carry)
                if done or request == "stop":
                    return self.finish(carry, "completed" if done else "stopped")
                if request:
                    self.writer.flush()
                    self.run.clear_request()
                    self.run.update_status(state="paused", pid=None, steps_per_second=self.rate)
                    self.run.event("paused", step=self.step)
                    log(f"paused at step {self.step}")
                    return 0
                count = self.next_count(seconds, int(host["executed"]), count)
                carry, host, seconds = run_block(
                    self.program, carry, self.data, count, spec.iterations
                )
        except BaseException as error:
            self.writer.flush()
            self.run.update_status(
                state="failed", pid=None, error=f"{type(error).__name__}: {error}"
            )
            self.run.event("failed", step=self.step, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self.writer.close()

    def next_count(self, seconds: float, executed: int, count: int) -> int:
        per_step = seconds / max(executed, 1)
        wanted = 2 ** int(math.log2(max(1.0, BLOCK_SECONDS / max(per_step, 1e-6))))
        wanted = min(wanted, MAX_BLOCK, count * 8)  # grow gradually: early timings are noisy
        to_checkpoint = self.spec.checkpoint_interval - self.step % self.spec.checkpoint_interval
        return max(1, min(wanted, to_checkpoint))

    def record(self, host, seconds: float) -> None:
        executed = int(host["executed"])
        buffers = host["buffers"]
        start, start_elapsed = self.step, self.elapsed
        self.step += executed
        self.elapsed += seconds
        if executed:
            rate = executed / seconds
            self.rate = rate if self.rate is None else 0.7 * self.rate + 0.3 * rate
        steps = range(start + 1, start + executed + 1)
        # steps of a block take equal shares of its time
        elapsed = (start_elapsed + seconds * np.arange(1, executed + 1) / executed).round(3)
        names = self.program.train_names
        columns = [buffers[name][:executed].tolist() for name in names]
        self.run.append(
            "train",
            [
                {"step": s, "elapsed": e, **dict(zip(names, values, strict=True))}
                for s, e, *values in zip(steps, elapsed.tolist(), *columns, strict=True)
            ],
        )
        val_rows = []
        for i in np.flatnonzero(~np.isnan(buffers["val_accuracy"][:executed])):
            row = {"step": start + int(i) + 1, "elapsed": float(elapsed[i])}
            row |= {name: float(buffers[name][i]) for name in self.program.val_names}
            val_rows.append(row)
        self.run.append("val", val_rows)
        if host["best_step"] > 0:
            self.best_accuracy, self.best_step = (
                float(host["best_accuracy"]),
                int(host["best_step"]),
            )
        last = val_rows[-1] if val_rows else None
        eta = (self.spec.iterations - self.step) / self.rate if self.rate else None
        changes: dict[str, Any] = dict(
            step=self.step,
            steps_per_second=self.rate,
            eta_seconds=eta,
            elapsed_seconds=round(self.elapsed, 3),
            best_accuracy=self.best_accuracy,
            best_step=self.best_step,
        )
        if last:
            changes |= dict(last_accuracy=last["val_accuracy"], last_loss=last["val_loss"])
        self.run.update_status(**changes)
        if time.monotonic() - self.last_log > LOG_SECONDS:
            self.last_log = time.monotonic()
            loss = buffers["loss"][executed - 1] if executed else float("nan")
            log(
                f"step {self.step}/{self.spec.iterations} loss {loss:.4f} "
                f"best val {self.best_accuracy or 0:.4f} {self.rate or 0:.1f} steps/s"
            )

    def save_network(self, flat, name: str, step: int, **extra) -> None:
        weights = np.asarray(jax.device_get(flat))
        spec = self.spec
        path = self.run.models_dir / f"{name}.safetensors"
        self.writer.submit(
            checkpoint.save_network,
            path,
            weights,
            run=self.run.id,
            model=spec.model,
            dataset=spec.dataset,
            step=step,
            **extra,
        )

    def save_state(self, carry: Carry) -> None:
        tensors = checkpoint.flatten(jax.device_get(carry))
        self.writer.submit(
            checkpoint.save_state,
            self.run.state_dir,
            tensors,
            step=self.step,
            chunk_size=self.program.chunk,
            elapsed_seconds=round(self.elapsed, 3),
        )

    def finish(self, carry: Carry, state: str) -> int:
        data, evaluate = self.data, self.program.evaluate
        final = self.program.opt.current(carry["state"])
        self.save_network(final, "final", self.step)
        summary: dict[str, Any] = {"steps": self.step, "elapsed_seconds": round(self.elapsed, 3)}
        best = carry["best"] if self.best_step is not None else final  # never validated
        for name, flat in (("final", final), ("best", best)):
            val_loss, val_acc = jax.device_get(evaluate(flat, data.val_x, data.val_y))
            test_loss, test_acc = jax.device_get(evaluate(flat, data.test_x, data.test_y))
            summary[name] = {
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
            }
        summary["best"]["step"] = self.best_step
        if self.best_step is None:
            self.save_network(final, "best", self.step)
        self.writer.flush()
        write_atomic(self.run.path / "summary.json", encode(summary))
        self.run.update_status(
            state=state,
            pid=None,
            finished_at=now(),
            eta_seconds=0.0,
            test_accuracy=summary["best"]["test_accuracy"],
        )
        self.run.event(state, step=self.step, summary=summary)
        log(
            f"{state} at step {self.step}: best val {summary['best']['val_accuracy']:.4f} "
            f"test {summary['best']['test_accuracy']:.4f}"
        )
        return 0


def main(run: Run) -> int:
    try:
        return train(run)
    except BaseException as error:
        traceback.print_exc()
        status = run.status()
        if status.state != "failed":
            run.update_status(state="failed", pid=None, error=f"{type(error).__name__}: {error}")
            run.event("failed", error=f"{type(error).__name__}: {error}")
        return 1 if not isinstance(error, KeyboardInterrupt) else 130


if __name__ == "__main__":
    sys.exit("run through `python -m kiseki.worker train <run_dir>`")
