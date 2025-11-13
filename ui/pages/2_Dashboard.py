# ui/pages/2_Dashboard.py
import sys, os
import streamlit as st

# บังคับให้มองเห็น root ของโปรเจกต์
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from race_logger import log_event, read_events

import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import io   # 👈 ใช้ BytesIO สำหรับ export

st.set_page_config(page_title="Race Engineer Dashboard", layout="wide")

st.title("🏎️ Race Engineer Dashboard")

# ---------------- Session Selector ----------------
sessions = {
    "Bahrain GP 2024 - Qualifying": {
        "lap_file": "data/eval/lap_times.xlsx",
        "summary_file": "data/eval/report_summary.xlsx",
        "strategy_file": "data/eval/strategy_result.xlsx"
    }
}

session_choice = st.sidebar.selectbox("Select Session", list(sessions.keys()))
files = sessions[session_choice]

lap_file = files["lap_file"]
summary_file = files["summary_file"]
strategy_file = files["strategy_file"]

# ---------------- Load Data ----------------
df_laps = pd.read_excel(lap_file, sheet_name="LapTimes")
df_laps.columns = [c.lower() for c in df_laps.columns]

df_stints = pd.read_excel(lap_file, sheet_name="StintSummary")
df_stints.columns = [c.lower() for c in df_stints.columns]

df_summary = pd.read_excel(summary_file, sheet_name="Summary")
df_strategy = pd.read_excel(strategy_file)

drivers = sorted(df_laps["driver"].unique())

# ---------------- Driver Selector ----------------
col1, col2 = st.columns(2)
with col1:
    driver = st.selectbox("Select Driver", drivers, index=0)
with col2:
    driver_b = st.selectbox("Compare with (Delta)", drivers, index=1)

# ---------------- Lap Time Chart ----------------
st.subheader("Lap Time Comparison")
fig, ax = plt.subplots(figsize=(10,5))
for drv in drivers:
    subset = df_laps[df_laps["driver"] == drv]
    ax.plot(subset["lapnumber"], subset["laptime_s"], label=drv)
ax.set_xlabel("Lap")
ax.set_ylabel("Lap Time (s)")
ax.legend()
st.pyplot(fig)

# ---------------- Stint Avg Chart ----------------
st.subheader("Average Lap Time per Stint")
fig, ax = plt.subplots(figsize=(8,5))
for drv in df_stints["driver"].unique():
    subset = df_stints[df_stints["driver"] == drv]
    ax.bar(subset["stint"], subset["avglaptime_s"], label=drv, alpha=0.7)
ax.set_xlabel("Stint")
ax.set_ylabel("Avg Lap Time (s)")
ax.legend()
st.pyplot(fig)

# ---------------- Boxplot Consistency ----------------
st.subheader("Lap Time Consistency per Driver")
df_clean = df_laps[df_laps["laptime_s"] < 200]
fig, ax = plt.subplots(figsize=(10,5))
df_clean.boxplot(column="laptime_s", by="driver", ax=ax)
ax.set_title("Lap Time Consistency")
ax.set_ylabel("Lap Time (s)")
st.pyplot(fig)

# ---------------- Delta Chart ----------------
st.subheader(f"Delta Chart: {driver} vs {driver_b}")
laps_a = df_laps[df_laps["driver"] == driver][["lapnumber", "laptime_s"]].set_index("lapnumber")
laps_b = df_laps[df_laps["driver"] == driver_b][["lapnumber", "laptime_s"]].set_index("lapnumber")
delta = laps_a.join(laps_b, lsuffix=f"_{driver}", rsuffix=f"_{driver_b}")
delta["delta_s"] = delta[f"laptime_s_{driver}"] - delta[f"laptime_s_{driver_b}"]

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(delta.index, delta["delta_s"], marker="o")
ax.axhline(0, color="black", linestyle="--")
ax.set_xlabel("Lap")
ax.set_ylabel(f"Δ LapTime ({driver} - {driver_b}) [s]")
st.pyplot(fig)

# ---------------- Strategy Simulation ----------------
st.subheader("Strategy Simulation Results")
st.dataframe(df_strategy)

# ---------------- Export Report ----------------
st.subheader("Export Report")
if st.button("Generate Auto Report"):
    with st.spinner("Generating report..."):
        subprocess.run(["python", "scripts/auto_report.py"])
    st.success("Report generated!")

    if os.path.exists("data/eval/auto_report.pdf"):
        with open("data/eval/auto_report.pdf", "rb") as f:
            st.download_button("Download PDF Report", f, file_name="auto_report.pdf")

    if os.path.exists("data/eval/auto_report.xlsx"):
        with open("data/eval/auto_report.xlsx", "rb") as f:
            st.download_button("Download Excel Report", f, file_name="auto_report.xlsx")

# ---------------- Race Log Export ----------------
st.subheader("📊 Race Log Data")

events = read_events()
laps = [e["data"] for e in events if e["event"] == "lap_completed"]

if not laps:
    st.warning("ยังไม่มีข้อมูล Lap ที่ log มา")
else:
    df = pd.DataFrame(laps)
    st.dataframe(df)

    # Export CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="race_log.csv",
        mime="text/csv"
    )

    # Export Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Lap Data")

    st.download_button(
        label="⬇️ Download Excel",
        data=excel_buffer.getvalue(),
        file_name="race_log.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
