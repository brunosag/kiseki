"""Process-wide JAX setup. Must run before the first `import jax` in a worker process.

Environment it honours (set by the launcher from the machine profile):
- KISEKI_GPU_MEMORY_FRACTION: share of GPU memory to preallocate. Unset (or "0"), JAX
  allocates on demand and caches what it got, which costs a steady training loop nothing
  measurable and lets a report run beside a training run, or a shared GPU stay shared.
- KISEKI_DETERMINISTIC=1: deterministic GPU kernels.
- KISEKI_PLATFORM=cpu: force CPU (tests, machines without GPUs).
"""

import os
import sys
from pathlib import Path

from kiseki import paths

NIXOS_DRIVER = Path("/run/opengl-driver/lib")  # where NixOS keeps libcuda


def ensure_driver_path() -> None:
    """Re-exec with NixOS's driver directory on LD_LIBRARY_PATH if it is missing.

    The dynamic loader reads LD_LIBRARY_PATH only at process start, and JAX's CUDA plugin
    dlopens libcuda, so a GNOME-launched app or a bare shell would otherwise fall back to CPU.
    """
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if not NIXOS_DRIVER.is_dir() or str(NIXOS_DRIVER) in current.split(":"):
        return
    env = dict(os.environ, LD_LIBRARY_PATH=f"{NIXOS_DRIVER}:{current}".rstrip(":"))
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)


def configure(*, deterministic: bool | None = None) -> None:
    fraction = os.environ.get("KISEKI_GPU_MEMORY_FRACTION", "0")
    if fraction in {"0", "0.0", "off"}:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", fraction)
    if deterministic is None:
        deterministic = os.environ.get("KISEKI_DETERMINISTIC") == "1"
    flags = os.environ.get("XLA_FLAGS", "")
    if deterministic and "xla_gpu_deterministic_ops" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_deterministic_ops=true".strip()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    platform = os.environ.get("KISEKI_PLATFORM")

    import jax

    if platform:
        jax.config.update("jax_platforms", platform)
    cache = paths.cache_dir() / "jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def power_state() -> dict[str, str | bool]:
    """The laptop's platform power profile and whether it is on mains power, when known.

    Both change GPU throughput, so runs record them next to their timings.
    """
    state: dict[str, str | bool] = {}
    profile = Path("/sys/firmware/acpi/platform_profile")
    if profile.exists():
        state["power_profile"] = profile.read_text().strip()
    supplies = Path("/sys/class/power_supply")
    for supply in sorted(supplies.iterdir()) if supplies.is_dir() else []:
        if (supply / "type").read_text().strip() == "Mains":
            state["on_mains"] = (supply / "online").read_text().strip() == "1"
            break
    return state


def device_summary() -> dict[str, str | int | bool]:
    import jax

    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    return {
        "platform": device.platform,
        "device": getattr(device, "device_kind", str(device)),
        "memory_bytes": int(stats.get("bytes_limit", 0)),
        "jax": jax.__version__,
        **power_state(),
    }
