"""kiseki: compare gradient-based and evolutionary optimization of neural networks.

Two halves share this package and must stay apart:

- the core (everything outside `kiseki.worker`, `kiseki.models` and plugin `impl` modules)
  runs the CLI, the run store, remote machines and the app server. It never imports JAX,
  so it starts fast and never opens a CUDA context.
- the worker (`kiseki.worker`) runs one training run or report per process, with JAX.

`tests/test_boundary.py` enforces the split.
"""
