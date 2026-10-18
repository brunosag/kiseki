# `kiseki`

Experiments comparing gradient descent with evolutionary optimization on the same small
CNNs, and reports that compare the networks they produce. JAX on the GPU, a CLI built for
agents, a desktop app, and runs on remote machines over SSH.

> **/kiseki/** `noun`
>
> 1. 軌跡: locus, trajectory, or path; traces left by a process.
> 2. 奇跡: miracle, wonder, or marvel.

## What is included

- **Datasets:** MNIST, Fashion-MNIST, KMNIST, CIFAR-10.
- **Models:** small_cnn, wide_cnn, vgg_mini, declared as layer lists.
- **Optimizers:** SGD, LEEA, CoSyNE, OpenAI-ES. Populations are scored in compiled,
  memory-sized chunks; many steps run per round trip to the GPU.
- **Runs:** queue, stop, pause and resume (bit-exact), checkpoints, target accuracy, on
  this machine or a remote one, surviving sleep and dropped connections.
- **Reports:** accuracy and per-class metrics, calibration, agreement (McNemar), weights,
  activations and CKA, t-SNE/PCA embeddings, robustness to corruptions, LRP heatmaps,
  loss landscapes and linear mode connectivity.

## Setup

On NixOS, `direnv allow` (or `nix develop`) provides Python 3.14, uv, Node, pnpm and Electron.
Then:

```bash
uv sync --extra cuda      # drop --extra cuda on machines without an NVIDIA GPU
uv run kiseki datasets --fetch mnist
```

## CLI

```bash
kiseki options                                   # datasets, models, optimizers and their params
kiseki run -o sgd,leea --preset smoke --watch    # a few seconds each: does it train?
kiseki run -o leea -s population_size=500,1000 -s iterations=5000
kiseki ls                                        # runs here and mirrored from remotes
kiseki show RUN | watch RUN... | compare RUN... | plot RUN... -o curves.png
kiseki stop | pause | resume RUN...
kiseki query "select optimizer, max(best_accuracy) from runs group by 1"
kiseki report RUN_A RUN_B:final                  # writes report.md, report.json and figures
```

`RUN` is any unique part of a run id. Every command accepts `--json`. The commands assume
the venv is active (`source .venv/bin/activate`); otherwise prefix them with `uv run`.

## Remote machines

Machines are listed in `~/.config/kiseki/machines.toml` by their `~/.ssh/config` alias
(see `src/kiseki/machines.py` for the format). `kiseki run --on lugia ...` deploys the
code, makes the dataset available and queues the runs there; `kiseki machines` shows each
machine's GPU, and `kiseki pull` mirrors results. On a shared server, set `notice` and
the runner keeps the required usage notice up to date while runs are active.

## The app

```bash
cd ui && pnpm install --frozen-lockfile && cd ..   # once
ui/kiseki-app
```

It starts `kiseki serve` and opens the app: live runs and curves, new runs, comparisons,
reports and machines, with everything one search away (Ctrl+K).

## Checks

```bash
uv run pytest
uv run ruff format src tests && uv run ruff check src tests
uv run basedpyright
```

`AGENTS.md` maps the code for agents; the design and its reasons are in `docs/design/`.
