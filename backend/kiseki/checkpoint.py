import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .schemas import ExperimentConfig, ExperimentStatus


@dataclass(slots=True)
class CheckpointSaver:
    directory: Path = Path("checkpoints")

    def save(
        self,
        *,
        model: nn.Module,
        status: ExperimentStatus,
        config: ExperimentConfig,
        optimizer: str,
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"step-{status.current_step:08d}"
        pt_path = self.directory / f"{stem}.pt"
        metadata_path = self.directory / f"{stem}.json"
        payload: dict[str, Any] = {
            "model_state": model.state_dict(),
            "status": status.model_dump(),
            "config": config.model_dump(),
            "optimizer": optimizer,
        }
        torch.save(payload, pt_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "checkpoint": pt_path.name,
                    "status": status.model_dump(),
                    "config": config.model_dump(),
                    "optimizer": optimizer,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return pt_path

