"""Alert history browser: thumbnails, severity, click-to-play clip, CSV export.
Reads directly from SQLite -- independent of the live pipeline loop on the Home page.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from sentinel.alerts.storage import AlertStorage
from sentinel.core.config import load_config

st.set_page_config(page_title="Sentinel — Alerts", page_icon="🚨", layout="wide")
st.title("🚨 Alert History")

config = load_config()
db_path = config.alerts.get("alerting", {}).get("storage", {}).get("db_path", "data/alerts.db")

if not Path(db_path).exists():
    st.info("No alerts recorded yet. Start the pipeline (Home page or `python -m sentinel.main`).")
    st.stop()

storage = AlertStorage(db_path)
rows = storage.all()

if not rows:
    st.info("No alerts recorded yet.")
    st.stop()

df = pd.DataFrame([dict(r) for r in rows])
df["time"] = pd.to_datetime(df["timestamp"], unit="s")

col_filter, col_export = st.columns([3, 1])
with col_filter:
    event_types = ["all"] + sorted(df["event_type"].unique().tolist())
    selected = st.selectbox("Filter by event type", event_types)
with col_export:
    csv_buf = io.StringIO()
    df.drop(columns=["details_json"]).to_csv(csv_buf, index=False)
    st.download_button(
        "Export CSV", data=csv_buf.getvalue(), file_name="sentinel_alerts.csv", mime="text/csv"
    )

filtered = df if selected == "all" else df[df["event_type"] == selected]
st.caption(f"{len(filtered)} alert(s)")

severity_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

for _, row in filtered.iterrows():
    badge = severity_color.get(row["severity"], "⚪")
    with st.expander(
        f"{badge} {row['event_type']} — {row['source']} — {row['time']} "
        f"(confidence {row['confidence']:.2f})"
    ):
        c1, c2 = st.columns([1, 2])
        with c1:
            if row["thumbnail_path"] and Path(row["thumbnail_path"]).exists():
                st.image(row["thumbnail_path"], caption="Snapshot at event time")
            else:
                st.caption("No thumbnail available")
        with c2:
            st.write(f"**Track ID:** {row['track_id']}")
            st.write(f"**Severity:** {row['severity']}")
            st.write(f"**Details:** {row['details_json']}")
            if row["clip_path"] and Path(row["clip_path"]).exists():
                st.video(row["clip_path"])
            else:
                st.caption("Clip not yet extracted (or buffer had too little history).")
            if st.button("Mark as false positive", key=f"fp_{row['id']}"):
                storage.mark_false_positive(int(row["id"]))
                st.success("Marked. This feeds back into rule/threshold tuning.")
                st.rerun()
