"""Streamlit dashboard -- live view. Run with:

    streamlit run sentinel/dashboard/app.py

The pipeline is created once via `st.cache_resource` and driven by a blocking render loop
(the standard pattern for embedding OpenCV video in Streamlit); alert history, thumbnails,
clip playback, and analytics/CSV export live on the separate pages in `dashboard/pages/`
(pure DB reads, so they don't fight the live loop for widget state).
"""
from __future__ import annotations

import time

import cv2
import streamlit as st

from sentinel.core.config import load_config
from sentinel.core.pipeline import SentinelPipeline
from sentinel.core.render import draw_overlay

st.set_page_config(page_title="Sentinel", page_icon="🛰️", layout="wide")


@st.cache_resource
def get_pipeline() -> SentinelPipeline:
    config = load_config()
    pipeline = SentinelPipeline(config)
    pipeline.start()
    return pipeline


def main() -> None:
    st.title("🛰️ Sentinel — Live Multi-Camera Anomaly Detection")
    st.caption(
        "Rule-based fall / loitering / abandoned-object / crowd-density / wrong-way detection "
        "running on pretrained YOLOv8 + BoT-SORT. See the **Alerts** and **Analytics** pages "
        "in the sidebar for history, clip playback, and charts."
    )

    pipeline = get_pipeline()
    run = st.toggle("Live", value=True, help="Pause to free the CPU/GPU without stopping capture")

    cols = st.columns(min(len(pipeline.sources), 3) or 1)
    frame_slots = {}
    for i, s in enumerate(pipeline.sources):
        col = cols[i % len(cols)]
        col.caption(s.name)
        frame_slots[s.name] = col.empty()

    status_slot = st.empty()

    while run:
        results = pipeline.step()
        n_events = sum(len(r["events"]) for r in results.values())
        status_slot.caption(
            f"{time.strftime('%H:%M:%S')} — {n_events} alert(s) this tick"
            if n_events
            else f"{time.strftime('%H:%M:%S')} — monitoring"
        )
        for name, res in results.items():
            if res["frame"] is None:
                continue
            frame = draw_overlay(res["frame"].copy(), res["detections"], res["events"])
            frame_slots[name].image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(0.05)


if __name__ == "__main__":
    main()
