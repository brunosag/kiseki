# `kiseki`

PyTorch experiments for comparing gradient-based and evolutionary optimization on the same neural network architectures. Provides a FastAPI backend, a React dashboard, and a CLI for headless training.

> **/kiseki/** `noun`
>
> 1. 軌跡: locus, trajectory, or path; traces left by a process.
> 2. 奇跡: miracle, wonder, or marvel.

## What is included

- **Datasets:** MNIST, Fashion MNIST, and CIFAR-10 through `torchvision`.
- **Models:** PyTorch CNNs selected by dataset.
- **Optimizers:** `SGD`, `LEEA`, and `CoSyNE`.
- **Training runtime:** Experiment start, stop, pause, resume, checkpoints, target accuracy, deterministic seeding, and CPU/GPU selection.
- **Analysis:** Checkpoint comparison reports with metrics, calibration, prediction overlap, activation and weight summaries, t-SNE/PCA projections, robustness curves, and LRP relevance samples.
- **Dashboard:** React, Vite, shadcn/ui, Recharts, and server-sent event streams from the backend.

## Requirements

- Python 3.14+
- `uv`
- Node.js and npm
- tmux for the repository-level `npm run train` helper

On NixOS, run this once from the repository root so `shell.nix` exposes CUDA libraries and the system CA bundle before Python starts:

```bash
direnv allow
```

## Setup

Install frontend dependencies:

```bash
npm --prefix frontend install
```

Backend dependencies are managed by `uv` from `backend/pyproject.toml`.

## Run the app

Start the backend and frontend together from the repository root:

```bash
npm run dev
```

This starts FastAPI at `http://127.0.0.1:8000` and Vite at
`http://127.0.0.1:5173`. The frontend proxies `/api` requests to the backend.

Run only the backend from `backend/`:

```bash
uv run fastapi dev main.py
```

Run only the frontend from `frontend/`:

```bash
npm run dev -- --host 127.0.0.1 --port 5173
```

## Headless training

Use the tmux helper from the repository root for SSH-safe runs:

```bash
npm run train -- --device gpu --optimizer LEEA --iterations 100000
```

It creates a `kiseki-train` session, runs `uv run kiseki train` from `backend/`, and writes a log under `logs/`. Detach with `Ctrl-b d`; reattach with:

```bash
tmux attach -t kiseki-train
```

Resume from the newest checkpoint, or pass a specific run ID:

```bash
npm run train -- --resume
npm run train -- --resume <run_id>
```

Run the Python CLI directly from `backend/` when tmux is not needed:

```bash
uv run kiseki train --device cpu --optimizer SGD --iterations 100
```

Useful CLI options include `--dataset`, `--batch-size`, `--target-acc`, `--checkpoint-interval`, `--seed`, `--deterministic`, and optimizer-specific parameters such as `--learning-rate`, `--population-size`, `--mutation-probability`, `--tournament-size`, and `--elitism-ratio`.

## Checks

Run backend tests and linting from `backend/`:

```bash
uv run pytest
uv run ruff check .
```

Run frontend checks from `frontend/`:

```bash
npm run lint
npm run build
```

## Project layout

- `backend/kiseki/`: FastAPI app, experiment manager, models, optimizers, checkpointing, analysis, data loading, and CLI.
- `backend/tests/`: Backend unit and API tests.
- `frontend/src/`: Dashboard application and API client.
- `scripts/dev.sh`: Starts backend and frontend together.
- `scripts/train-tmux.sh`: Runs CLI training inside tmux.

Runtime artifacts are written to `backend/data/`, `backend/checkpoints/`, and `logs/`.
