"""Where kiseki keeps state on a machine. Override the root with KISEKI_HOME."""

import os
from pathlib import Path


def home() -> Path:
    root = os.environ.get("KISEKI_HOME")
    if root:
        return Path(root).expanduser()
    data = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(data).expanduser() / "kiseki"


def runs_dir() -> Path:
    return home() / "runs"


def reports_dir() -> Path:
    return home() / "reports"


def cache_dir() -> Path:
    root = os.environ.get("KISEKI_CACHE")
    if root:
        return Path(root).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(cache).expanduser() / "kiseki"


def datasets_dir() -> Path:
    return cache_dir() / "datasets"


def repo_root() -> Path:
    """The source checkout this package runs from (code shipped to remotes comes from here)."""
    return Path(__file__).resolve().parents[2]
