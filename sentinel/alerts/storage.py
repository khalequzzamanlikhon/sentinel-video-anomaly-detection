"""SQLite-backed alert storage. Single table, no ORM -- this is a portfolio project, not a
distributed system; SQLite is the right amount of database here.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    track_id INTEGER,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp REAL NOT NULL,
    thumbnail_path TEXT,
    clip_path TEXT,
    details_json TEXT,
    false_positive INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_event_type ON alerts (event_type);
"""


class AlertStorage:
    def __init__(self, db_path: str | Path = "data/alerts.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def insert(
        self,
        source: str,
        event_type: str,
        track_id: int | None,
        severity: str,
        confidence: float,
        timestamp: float,
        thumbnail_path: str | None = None,
        clip_path: str | None = None,
        details: dict | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO alerts
               (source, event_type, track_id, severity, confidence, timestamp,
                thumbnail_path, clip_path, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source,
                event_type,
                track_id,
                severity,
                confidence,
                timestamp,
                thumbnail_path,
                clip_path,
                json.dumps(details or {}),
                time.time(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_clip_path(self, alert_id: int, clip_path: str) -> None:
        self._conn.execute("UPDATE alerts SET clip_path = ? WHERE id = ?", (clip_path, alert_id))
        self._conn.commit()

    def mark_false_positive(self, alert_id: int, is_false_positive: bool = True) -> None:
        self._conn.execute(
            "UPDATE alerts SET false_positive = ? WHERE id = ?", (int(is_false_positive), alert_id)
        )
        self._conn.commit()

    def recent(self, limit: int = 100) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()

    def all(self) -> list[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
        return cur.fetchall()

    def counts_by_event_type(self) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT event_type, COUNT(*) as count FROM alerts GROUP BY event_type"
        )
        return cur.fetchall()

    def counts_by_source(self) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT source, COUNT(*) as count FROM alerts GROUP BY source"
        )
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()
