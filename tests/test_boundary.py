"""The core (CLI, server, store, remotes, catalogs) never imports JAX: it starts fast and
runs where JAX is not installed."""

import subprocess
import sys

from kiseki import analyses

LIGHT = [
    "kiseki.cli",
    "kiseki.server",
    "kiseki.actions",
    "kiseki.store",
    "kiseki.spec",
    "kiseki.runner",
    "kiseki.remote",
    "kiseki.report",
    "kiseki.query",
    "kiseki.notice",
    "kiseki.machines",
    "kiseki.plots",
    "kiseki.datasets",
    "kiseki.models",
    "kiseki.optimizers",
    "kiseki.analyses",
    *[f"kiseki.analyses.{name}" for name in analyses.NAMES],
]


def test_core_never_imports_jax():
    code = (
        f"import importlib, sys\nfor m in {LIGHT!r}: importlib.import_module(m)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
