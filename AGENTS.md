# kiseki

Retired since 19 October 2026: new research code is written from scratch in kiseki-research, in CUDA C++ (its `docs/design/compute.md`). Keep kiseki working for the experiments that still run on it; build nothing new here.

Experiments comparing SGD with evolutionary optimizers (LEEA, CoSyNE, OpenAI-ES) on small CNNs, and reports that compare the trained networks. The design and its reasons are in `docs/design/`; the words to use in code, the CLI, the UI and docs in `docs/terms.md`. Open work: `TODO.md`.

The code is written for agents to navigate, not humans, and everything, the app as much as the training and analysis runs, is optimized for performance on this laptop.

## You may

Train on any machine in `~/.config/kiseki/machines.toml`, inhibit the laptop's sleep while a run trains (the queue runner does), and run scratch experiments with `KISEKI_HOME=/tmp/<dir>` to keep them out of the real store.

## Limits

- Research that uses kiseki lives in the private repo `brunosag/kiseki-research` (`~/src/kiseki-research`). Its findings, unpublished results and idea-revealing code stay there until the user publishes; only general tool improvements land here.
- The GitHub repo is public. No email addresses or other contact details in tracked files: they live in `~/.config/kiseki/machines.toml` (untracked).
- lugia's rule, binding on every run there: "O arquivo de texto /etc/motd.tail pode ser editado por qualquer usuário e DEVE ser editado por qualquer pessoa que coloque um experimento para rodar, informando: nome, e-mail, e previsão do intervalo de uso do servidor (data de início e previsão de final)." The queue runner does this (`kiseki/notice.py`); never start work on lugia outside it.
- lugia's `~/kiseki` is the old project's checkout; leave it alone. The rewrite deploys to `~/.local/share/kiseki-remote/`.

## Machines

- The laptop (RTX 3050, 4 GB) and the vps (6 CPUs, no GPU) each have a checkout at `~/src/kiseki` and list each other in `machines.toml` (`--on vps`, `--on laptop`). From the vps, train on the laptop's GPU with `--on laptop`; the laptop sleeps when unattended, so check `kiseki machines` first (SSH to a sleeping laptop times out after about 20 s).
- One run at a time per machine, across homes: scratch runs wait for the machine's running one.
- A suspend mid-run can leave the worker hung with the GPU idle on wake; the runner kills it after 15 min and resumes it from its checkpoint.
- The laptop's app (GNOME menu, tray) runs from `~/src/kiseki`, which nothing pulls on its own: a landed change reaches it once that checkout is pulled and the app restarted.
- A `v<version>` tag publishes the AppImage as a GitHub Release (bump `ui/package.json`, `pyproject.toml` and kiseki's entry in `uv.lock` together). Build or test one on the vps (`ui/README.md`), not the laptop.

## Gotchas

- `uv add`, `uv remove` and `uv lock` re-sync without extras and uninstall the CUDA libraries, and `uv run` in a checkout without `.venv` creates one without them; run `uv sync --extra cuda` after them, or JAX silently runs on the CPU.
- lugia's GPU falls off the bus under XLA's autotuner (`~/.agents/skills/references/lugia.md`), and kiseki's workers don't turn it off: lugia's entry in `machines.toml` needs `env = { XLA_FLAGS = "--xla_gpu_autotune_level=0" }`.
- A remote needs uv on PATH or in `~/.local/bin`. Datasets download from their original hosts; when a remote reaches one slowly, `Remote.ensure_dataset` pushes this machine's parsed copy.
