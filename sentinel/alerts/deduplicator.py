"""Same track + same event type within `dedup_window_sec` collapses to one alert."""
from __future__ import annotations

from sentinel.anomaly.events import AnomalyEvent


class Deduplicator:
    def __init__(self, dedup_window_sec: float = 30) -> None:
        self.dedup_window_sec = dedup_window_sec
        self._last_seen: dict[tuple[str, str, int | None], float] = {}

    def is_duplicate(self, event: AnomalyEvent) -> bool:
        key = (event.source, event.event_type, event.track_id)
        last = self._last_seen.get(key)
        self._last_seen[key] = event.timestamp
        if last is None:
            return False
        return (event.timestamp - last) < self.dedup_window_sec

    def filter(self, events: list[AnomalyEvent]) -> list[AnomalyEvent]:
        return [e for e in events if not self.is_duplicate(e)]
