"""Shared anomaly event type produced by every detector in this package."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AnomalyEvent:
    event_type: str  # "fall" | "loitering" | "abandoned_object" | "crowd_density" | "wrong_way"
    track_id: int | None
    source: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    bbox: tuple[float, float, float, float] | None = None
    details: dict = field(default_factory=dict)
