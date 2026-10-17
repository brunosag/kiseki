"""Machines runs can go to, from `<config>/machines.toml` (outside the repo: it holds
contact details). `local` always exists. Example:

    [contact]                  # who the server notice names; default: git config
    name = "Ada Lovelace"
    email = "ada@example.org"

    [machines.lugia]
    ssh = "lugia"              # Host alias from ~/.ssh/config (ProxyJump etc. apply)
    notice = "/etc/motd.tail"  # shared server: announce runs there while any is active

    [machines.vps]
    ssh = "vps"
    extras = []                # no GPU: plain JAX
    env = { KISEKI_PLATFORM = "cpu" }

A remote keeps its own kiseki home (`home`, default ~/.local/share/kiseki) with code/,
runs/ and its queue; datasets are copied there from this machine.
"""

import subprocess
import tomllib
from pathlib import Path

import msgspec

from kiseki import paths


class Contact(msgspec.Struct, frozen=True):
    name: str
    email: str


class Machine(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    name: str = "local"
    ssh: str | None = None
    home: str = "~/.local/share/kiseki"
    notice: str | None = None
    env: dict[str, str] = {}
    extras: tuple[str, ...] = ("cuda",)  # pyproject extras installed there
    description: str = ""

    @property
    def is_local(self) -> bool:
        return self.ssh is None


class Config(msgspec.Struct, frozen=True, kw_only=True):
    contact: Contact | None = None
    machines: dict[str, Machine] = {}


def config_path() -> Path:
    return paths.config_dir() / "machines.toml"


def load_config() -> Config:
    path = config_path()
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    for name, entry in raw.get("machines", {}).items():
        entry.setdefault("name", name)
    return msgspec.convert(raw, Config)


def all_machines() -> dict[str, Machine]:
    configured = load_config().machines
    return {"local": configured.get("local", Machine())} | {
        name: machine for name, machine in configured.items() if name != "local"
    }


def get(name: str) -> Machine:
    machines = all_machines()
    if name not in machines:
        raise KeyError(f"unknown machine {name!r}; known: {', '.join(machines)} ({config_path()})")
    return machines[name]


def contact() -> Contact:
    configured = load_config().contact
    if configured:
        return configured

    def git(key: str) -> str:
        result = subprocess.run(["git", "config", key], capture_output=True, text=True)
        return result.stdout.strip()

    name, email = git("user.name"), git("user.email")
    if not name or not email:
        raise RuntimeError(f"set [contact] name and email in {config_path()}")
    return Contact(name, email)
