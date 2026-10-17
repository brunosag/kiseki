"""Optimizer catalog. Add an optimizer as a package here, then list it in `NAMES`.

Each package has two modules:
- `__init__.py` (light, no JAX): `Params` (a msgspec Struct whose fields carry defaults,
  bounds and descriptions; the CLI, app forms and run validation read it) and `SUMMARY`.
- `impl.py` (JAX): `build(params, problem) -> Optimizer`, see `kiseki.worker.optimizer`.
"""

import importlib
from typing import Any

import msgspec

NAMES = ("sgd", "leea", "cosyne", "openai_es")


def _package(name: str):
    if name not in NAMES:
        raise KeyError(f"unknown optimizer {name!r}; known: {', '.join(NAMES)}")
    return importlib.import_module(f"kiseki.optimizers.{name}")


def params_type(name: str) -> type[msgspec.Struct]:
    return _package(name).Params


def summary(name: str) -> str:
    return _package(name).SUMMARY


def normalize_params(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Validate raw params and return the full set, defaults filled in."""
    params = msgspec.convert(raw, params_type(name), strict=False)
    return msgspec.to_builtins(params)


def load_impl(name: str):
    _package(name)
    return importlib.import_module(f"kiseki.optimizers.{name}.impl")
