# Kiseki Backend

FastAPI and PyTorch backend for the Kiseki dashboard.

Run the API locally:

```bash
uv run fastapi dev main.py
```

Run checks:

```bash
uv run pytest
uv run ruff check .
```

Run headless training directly:

```bash
uv run kiseki train --device cpu --optimizer SGD --iterations 100 --checkpoint-interval 10
```

For unstable SSH sessions, use the repository-level tmux helper:

```bash
npm run train -- --device gpu --optimizer LEEA --iterations 100000
```

The helper starts a `kiseki-train` tmux session, runs `uv run kiseki train`
from `backend/`, tees output to `logs/`, and attaches immediately. Detach with
`Ctrl-b d` and reattach with:

```bash
tmux attach -t kiseki-train
```

Resume from a saved checkpoint with:

```bash
npm run train -- --resume <run_id>
```

On NixOS, allow direnv once from the repository root so `shell.nix` exposes CUDA
and the system CA bundle before Python starts:

```bash
direnv allow
```

Then run GPU training normally:

```bash
npm run train -- --device gpu
```
