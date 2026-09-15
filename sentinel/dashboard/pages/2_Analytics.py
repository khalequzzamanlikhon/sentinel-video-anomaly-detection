"""Analytics: alert frequency over time, per-camera breakdown, top event types."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sentinel.alerts.storage import AlertStorage
from sentinel.core.config import load_config

st.set_page_config(page_title="Sentinel — Analytics", page_icon="📊", layout="wide")
st.title("📊 Analytics")

config = load_config()
db_path = config.alerts.get("alerting", {}).get("storage", {}).get("db_path", "data/alerts.db")

if not Path(db_path).exists():
    st.info("No alerts recorded yet.")
    st.stop()

storage = AlertStorage(db_path)
rows = storage.all()
if not rows:
    st.info("No alerts recorded yet.")
    st.stop()

df = pd.DataFrame([dict(r) for r in rows])
df["time"] = pd.to_datetime(df["timestamp"], unit="s")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total alerts", len(df))
col2.metric("False positives flagged", int(df["false_positive"].sum()))
col3.metric("Cameras reporting", df["source"].nunique())
col4.metric("Event types seen", df["event_type"].nunique())

st.subheader("Alerts over time")
by_time = df.set_index("time").resample("5min").size().rename("alerts")
st.line_chart(by_time)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Top event types")
    st.bar_chart(df["event_type"].value_counts())
with c2:
    st.subheader("Alerts per camera")
    st.bar_chart(df["source"].value_counts())

st.subheader("Severity breakdown")
st.bar_chart(df["severity"].value_counts())
