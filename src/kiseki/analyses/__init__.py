"""Analysis catalog. A report runs analyses over trained networks (`kiseki.report`); add an
analysis as a package here, then list it in `NAMES` (report order).

Each package has two modules:
- `__init__.py` (light, no JAX): `Params` (msgspec Struct with defaults, bounds and
  descriptions), `TITLE` and `SUMMARY`.
- `impl.py` (JAX): `run(params, context) -> Section`, see `kiseki.worker.report`.
"""

import importlib
from typing import Any

import msgspec

NAMES = (
    "metrics", "calibration", "agreement", "weights", "representations", "embedding",
    "robustness", "lrp", "loss_landscape",
)


def _package(name: str):
    if name not in NAMES:
        raise KeyError(f"unknown analysis {name!r}; known: {', '.join(NAMES)}")
    return importlib.import_module(f"kiseki.analyses.{name}")


def params_type(name: str) -> type[msgspec.Struct]:
    return _package(name).Params


def title(name: str) -> str:
    return _package(name).TITLE


def summary(name: str) -> str:
    return _package(name).SUMMARY


def normalize_params(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    return msgspec.to_builtins(msgspec.convert(raw, params_type(name), strict=False))


def load_impl(name: str):
    _package(name)
    return importlib.import_module(f"kiseki.analyses.{name}.impl")
