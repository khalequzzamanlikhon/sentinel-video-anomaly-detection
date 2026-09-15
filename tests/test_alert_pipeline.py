import tempfile
from pathlib import Path

from sentinel.alerts.alert_manager import AlertManager
from sentinel.alerts.deduplicator import Deduplicator
from sentinel.anomaly.events import AnomalyEvent


def make_event(event_type="fall", track_id=1, source="cam1", timestamp=0.0, confidence=0.8):
    return AnomalyEvent(
        event_type=event_type, track_id=track_id, source=source, confidence=confidence, timestamp=timestamp
    )


def test_deduplicator_collapses_repeats_within_window():
    dedup = Deduplicator(dedup_window_sec=30)
    events = [make_event(timestamp=0.0), make_event(timestamp=5.0), make_event(timestamp=40.0)]
    kept = dedup.filter(events)
    # first kept, second (within 30s of first) dropped, third (40s later) kept
    assert len(kept) == 2
    assert kept[0].timestamp == 0.0
    assert kept[1].timestamp == 40.0


def test_deduplicator_treats_different_tracks_independently():
    dedup = Deduplicator(dedup_window_sec=30)
    e1 = make_event(track_id=1, timestamp=0.0)
    e2 = make_event(track_id=2, timestamp=1.0)
    kept = dedup.filter([e1, e2])
    assert len(kept) == 2


def test_alert_manager_persists_and_scores_severity():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "alerts.db"
        alerts_config = {
            "alerting": {
                "dedup_window_sec": 30,
                "severity": {"fall": "critical"},
                "clip": {"pre_event_sec": 1, "post_event_sec": 0.1},
                "storage": {
                    "db_path": str(db_path),
                    "clips_dir": str(Path(tmpdir) / "clips"),
                    "thumbnails_dir": str(Path(tmpdir) / "thumbs"),
                },
            }
        }
        manager = AlertManager(alerts_config, frame_buffers={})
        try:
            ids = manager.process([make_event(event_type="fall", timestamp=100.0)])
            assert len(ids) == 1

            rows = manager.storage.recent()
            assert len(rows) == 1
            assert rows[0]["event_type"] == "fall"
            assert rows[0]["severity"] == "critical"

            # duplicate within window is dropped before it reaches storage
            manager.process([make_event(event_type="fall", timestamp=105.0)])
            assert len(manager.storage.recent()) == 1
        finally:
            manager.shutdown()
