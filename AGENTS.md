# kiseki

Experiments comparing SGD with evolutionary optimizers (LEEA, CoSyNE, OpenAI-ES) on small
CNNs, and reports that compare the trained networks. Preferences: `PREFERENCES.md`. Terms:
`GLOSSARY.md`. Design and its reasons: `docs/design/`.

## Areas

| Directory | What it is | Design | Runs it |
|---|---|---|---|
| `src/kiseki/` (core) | CLI, run store, queue runner, remotes, app server, catalogs. Never imports JAX | `docs/design/architecture.md` | `uv run kiseki ...` |
| `src/kiseki/worker/` | JAX processes: training (`train.py`) and reports (`report.py`) | `docs/design/training.md` | the queue runner, `kiseki report` |
| `src/kiseki/{datasets,models,optimizers,analyses}/` | Plugins: one module or package each; light `__init__` + JAX `impl.py` | each package's docstring | the worker |
| `ui/` | React app in an Electron shell, talking to `kiseki serve` | `docs/design/architecture.md` | `ui/kiseki-app` |
| `tests/` | pytest, JAX on CPU | | `uv run pytest` |

Add a dataset, model, optimizer or analysis by copying the nearest sibling and listing it
in its package's `NAMES`; the CLI, server and UI pick it up from the catalogs.

## Commands

- `uv sync --extra cuda` sets up; `uv run pytest` (≈40 s), `uv run ruff format src tests`,
  `uv run ruff check src tests`, `uv run basedpyright`.
- `uv run kiseki run --preset smoke -o sgd,leea` checks that configurations train (seconds);
  `kiseki ls`, `show`, `compare`, `query`, `report RUN... ` read results; `kiseki --help`
  lists the rest. Every command takes `--json`.
- Scratch experiments: `KISEKI_HOME=/tmp/<dir>` keeps them out of the real store. Its runs
  still wait for the machine's running one: one run at a time per machine, across homes.

## Machines

- The laptop (RTX 3050, 4 GB) and the vps (2 CPUs, no GPU) each have a checkout at
  `~/src/kiseki`, the vps's for agents working there. Each lists the other in its
  `~/.config/kiseki/machines.toml` (`--on vps`, `--on laptop`).
- From the vps, train on the laptop's GPU with `--on laptop`. The laptop sleeps when
  unattended: check `kiseki machines` first; SSH to a sleeping laptop times out after
  about 20 s, and its runs freeze during sleep and continue on wake.

## Rules

- The GitHub repo is public. No email addresses or other contact details in tracked files:
  they live in `~/.config/kiseki/machines.toml` (untracked).
- lugia's rule, binding on every run there: "O arquivo de texto /etc/motd.tail pode ser
  editado por qualquer usuário e DEVE ser editado por qualquer pessoa que coloque um
  experimento para rodar, informando: nome, e-mail, e previsão do intervalo de uso do
  servidor (data de início e previsão de final)." The queue runner does this
  (`kiseki/notice.py`); never start work on lugia outside it.
- lugia's `~/kiseki` is the old project's checkout; leave it alone. The rewrite deploys to
  `~/.local/share/kiseki-remote/`; runs sent before 2026-10-18 (all failed) are in
  `~/.local/share/kiseki/` there.
- Don't delegate UI work to Sol (Codex GPT-6.1): "I shouldn't have let Sol do the UI work"
  (2026-10-18; its UI was rewritten).

## Gotchas

- `uv add`, `uv remove` and `uv lock` re-sync without extras and uninstall the CUDA
  libraries, and `uv run` in a checkout without `.venv` creates one without them; run
  `uv sync --extra cuda` after them, or JAX silently runs on the CPU (on the laptop).
- lugia (RTX 5070 Ti, driver 580 = CUDA 13.0) lost its GPU on 2026-10-17 during the first
  run with the CUDA 13.4 user-space libraries (`CUDA_ERROR_LAUNCH_FAILED` in autotuning,
  then "No devices were found"; needs an admin reset). `pyproject.toml` pins them to 13.0.
  After a reset, test a tiny kernel before a run.
- On NixOS, libcuda is in `/run/opengl-driver/lib`; workers re-exec themselves with it on
  `LD_LIBRARY_PATH` (`worker/runtime.py`), so launchers need not set it.
- Datasets download from their original hosts; lugia reaches some slowly (CIFAR-10 from
  Toronto), and `Remote.ensure_dataset` then pushes this machine's parsed copy.
- A remote needs uv on PATH or in `~/.local/bin` (`curl -LsSf https://astral.sh/uv/install.sh | sh`);
  the vps got it that way on 2026-10-17. Its first SSH connection takes ~12 s over
  Tailscale; later commands reuse the ControlMaster connection (~0.5 s).
