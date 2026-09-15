"""Shared bounding-box/alert overlay drawing, used by both the CLI (`main.py`, OpenCV window)
and the Streamlit dashboard so the two don't drift apart.
"""
from __future__ import annotations

import cv2


def draw_overlay(frame, detections, events):
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        color = (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} #{det.track_id}" if det.track_id is not None else det.cls_name
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for i, event in enumerate(events):
        text = f"ALERT: {event.event_type} (track {event.track_id}, conf {event.confidence:.2f})"
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame
