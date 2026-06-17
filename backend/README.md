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

Benchmark defaults use MNIST with the dashboard batch size, seed, and LEEA
parameters. By default, the benchmark runs both LEEA and SGD for 10 iterations
on CPU and, when CUDA is available, GPU. For a quick smoke run, use:

```powershell
uv run kiseki benchmark --device cpu --benchmark synthetic --optimizer SGD --iterations 1 --batch-size 4
```

On NixOS, allow direnv once from the repository root so `shell.nix` exposes CUDA
and the system CA bundle before Python starts:

```bash
direnv allow
```

Then run GPU benchmarks normally:

```bash
uv run kiseki benchmark --device gpu
```
