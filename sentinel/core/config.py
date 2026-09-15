"""YAML config loading for cameras, models, and alert rules."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclasses.dataclass
class SourceConfig:
    name: str
    type: str
    source: Any
    roi: list[float] | None = None
    expected_flow_deg: float = 0.0


@dataclasses.dataclass
class SentinelConfig:
    cameras: dict[str, Any]
    models: dict[str, Any]
    alerts: dict[str, Any]

    @property
    def sources(self) -> list[SourceConfig]:
        flow = self.cameras.get("expected_flow_deg", {})
        out = []
        for s in self.cameras.get("sources", []):
            out.append(
                SourceConfig(
                    name=s["name"],
                    type=s["type"],
                    source=s["source"],
                    roi=s.get("roi"),
                    expected_flow_deg=float(flow.get(s["name"], 0.0)),
                )
            )
        return out


def load_config(config_dir: str | Path = CONFIG_DIR) -> SentinelConfig:
    config_dir = Path(config_dir)
    return SentinelConfig(
        cameras=_load_yaml(config_dir / "cameras.yaml"),
        models=_load_yaml(config_dir / "models.yaml"),
        alerts=_load_yaml(config_dir / "alerts.yaml"),
    )
