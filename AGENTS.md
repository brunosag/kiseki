# kiseki

Experiments comparing SGD with evolutionary optimizers (LEEA, CoSyNE, OpenAI-ES) on small
CNNs, and reports that compare the trained networks. Words to use in code, the CLI, the UI
and docs: `docs/terms.md`.

## Production

- The laptop (RTX 3050, 4 GB) and the vps (6 CPUs, no GPU) each have a checkout at
  `~/src/kiseki`, the vps's for agents working there. Each lists the other in its
  `~/.config/kiseki/machines.toml` (`--on vps`, `--on laptop`).
- The laptop's app (GNOME menu, tray) runs from `~/src/kiseki`, which nothing pulls on its
  own: a landed change reaches the app once that checkout is pulled and the app restarted.
- From the vps, train on the laptop's GPU with `--on laptop`. The laptop sleeps when
  unattended: check `kiseki machines` first; SSH to a sleeping laptop times out after
  about 20 s. While a run trains, its queue runner holds a sleep inhibitor ("yes, you can
  inhibit sleep if you need to", the user, 2026-10-18); T3 holds one only during an agent
  turn. A suspend mid-run (lid, manual) can leave the worker hung with the GPU idle on
  wake (2026-10-18); the runner then kills it after 15 min and resumes it from its
  checkpoint.
- A `v<version>` tag publishes the AppImage as a GitHub Release (bump `ui/package.json`,
  `pyproject.toml` and kiseki's entry in `uv.lock` together). Build or test one on the vps
  (`ui/README.md`, The AppImage), not the laptop.

## Rules

- Scratch experiments: `KISEKI_HOME=/tmp/<dir>` keeps them out of the real store. Its runs
  still wait for the machine's running one: one run at a time per machine, across homes.
- Research that uses kiseki lives in the private repo `brunosag/kiseki-research`
  (`~/src/kiseki-research`). Its findings, unpublished results and idea-revealing code
  stay there until the user publishes; only general tool improvements land here.
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
- lugia's GPU falls off the bus under XLA's autotuner (`~/.agents/skills/references/lugia.md`),
  and kiseki's workers don't turn it off: lugia's entry in `machines.toml` needs
  `env = { XLA_FLAGS = "--xla_gpu_autotune_level=0" }`.
- On NixOS, libcuda is in `/run/opengl-driver/lib`; workers re-exec themselves with it on
  `LD_LIBRARY_PATH` (`worker/runtime.py`), so launchers need not set it.
- Datasets download from their original hosts; lugia reaches some slowly (CIFAR-10 from
  Toronto), and `Remote.ensure_dataset` then pushes this machine's parsed copy.
- A remote needs uv on PATH or in `~/.local/bin` (`curl -LsSf https://astral.sh/uv/install.sh | sh`);
  the vps got it that way on 2026-10-17. Its first SSH connection takes ~12 s over
  Tailscale; later commands reuse the ControlMaster connection (~0.5 s).
