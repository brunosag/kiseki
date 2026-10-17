"""The report contract: which trained networks to analyse and how, frozen into
<reports>/<id>/spec.json. The worker (`python -m kiseki.worker report <dir>`) writes

  report.json   every section's numbers (summary and details), the input to the app
  report.md     the same as prose and tables, with figure links, for agents and people
  figures/      PNGs
  arrays/       larger arrays the app draws (embeddings, relevance maps), .npy
  status.json   state while it runs: queued, running, completed, failed

Models are `run:name` refs (name: best, final or step-<n>); all must share a dataset.
"""

import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import msgspec

from kiseki import analyses, paths
from kiseki.spec import ModelRef, encode
from kiseki.store import now, write_atomic


class ReportSpec(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    models: list[str]
    labels: list[str] = []
    split: Literal["test", "val"] = "test"
    analyses: dict[str, dict[str, Any]] = {}
    seed: int = 0


class ReportStatus(msgspec.Struct, kw_only=True):
    state: Literal["queued", "running", "completed", "failed"]
    stage: str = ""
    seconds: dict[str, float] = {}
    error: str | None = None
    updated_at: str = ""


def normalize(spec: ReportSpec) -> ReportSpec:
    if not spec.models:
        raise ValueError("a report needs at least one model")
    refs = [str(ModelRef.parse(text)) for text in spec.models]
    chosen = spec.analyses or {name: {} for name in analyses.NAMES}
    unknown = set(chosen) - set(analyses.NAMES)
    if unknown:
        raise KeyError(f"unknown analyses {sorted(unknown)}; known: {', '.join(analyses.NAMES)}")
    ordered = {name: analyses.normalize_params(name, chosen[name])
               for name in analyses.NAMES if name in chosen}
    if spec.labels and len(spec.labels) != len(refs):
        raise ValueError("give one label per model")
    return msgspec.structs.replace(spec, models=refs, analyses=ordered)


def new_report_id(now_: datetime | None = None) -> str:
    return f"{(now_ or datetime.now(UTC)).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class Report:
    def __init__(self, path: Path):
        self.path = path
        self.id = path.name

    @classmethod
    def create(cls, spec: ReportSpec, root: Path | None = None) -> "Report":
        path = (root or paths.reports_dir()) / new_report_id()
        path.mkdir(parents=True)
        write_atomic(path / "spec.json", encode(spec))
        report = cls(path)
        report.write_status(ReportStatus(state="queued"))
        return report

    @classmethod
    def find(cls, report_id: str, root: Path | None = None) -> "Report":
        base = root or paths.reports_dir()
        matches = sorted(p for p in base.glob(f"*{report_id}*") if p.is_dir()) if base.exists() else []
        if len(matches) != 1:
            raise KeyError(f"{report_id!r} matches {len(matches)} reports")
        return cls(matches[0])

    def spec(self) -> ReportSpec:
        return msgspec.json.decode((self.path / "spec.json").read_bytes(), type=ReportSpec)

    def status(self) -> ReportStatus:
        return msgspec.json.decode((self.path / "status.json").read_bytes(), type=ReportStatus)

    def write_status(self, status: ReportStatus) -> None:
        status.updated_at = now()
        write_atomic(self.path / "status.json", encode(status))
