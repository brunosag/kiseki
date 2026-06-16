# Kiseki Backend

FastAPI and PyTorch backend for the Kiseki dashboard.

Run the API locally:

```powershell
uv run fastapi dev main.py
```

Run checks:

```powershell
uv run pytest
uv run ruff check .
```

Run backend benchmarks:

```powershell
uv run kiseki benchmark
```

Benchmark defaults mirror the dashboard defaults: MNIST, LEEA, safe mode, seed 42,
batch size 1000, 100000 iterations, and LEEA population 200. The default device
is `auto`, which uses CUDA when available and CPU otherwise. For a quick smoke
run, use:

```powershell
uv run kiseki benchmark --device cpu --benchmark synthetic --optimizer SGD --iterations 1 --batch-size 4
```

On NixOS, allow direnv once from this directory so `shell.nix` exposes CUDA and the
system CA bundle before Python starts:

```bash
direnv allow
```

Then run GPU benchmarks normally:

```bash
uv run kiseki benchmark --optimizer both --benchmark both --speed-mode fast --iterations 50
```
