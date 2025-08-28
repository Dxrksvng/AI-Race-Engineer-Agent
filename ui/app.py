# ui/app.py
from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from agents.agent import build_agent
from tools.telemetry_tools import load_session, lap_summary, build_delta_vs_time, stint_summary

st.set_page_config(page_title="AI Race Engineer - Lap Viewer", layout="wide")

st.title("🏁 AI Race Engineer — Lap Viewer (MVP)")
st.caption("FastF1 + Streamlit + Plotly")

# --- Sidebar: เลือก Session ---
with st.sidebar:
    st.header("Session Selector")
    year = st.number_input("Year", min_value=2018, max_value=2025, value=2024, step=1)
    gp = st.text_input("Grand Prix (e.g., Bahrain, Spain, Monaco)", value="Bahrain")
    session_name = st.selectbox("Session", ["FP1", "FP2", "FP3", "Q", "SQ", "R"], index=3)
    driver_a = st.text_input("Driver A (e.g., VER, LEC, HAM)", value="VER").upper().strip()
    driver_b = st.text_input("Driver B (optional)", value="LEC").upper().strip()
    load_btn = st.button("Load session", type="primary")

# --- State ---
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session" not in st.session_state:
    st.session_state.session = None
if "loaded_meta" not in st.session_state:
    st.session_state.loaded_meta = ""

# Load session + build agent
if load_btn:
    try:
        session = load_session(int(year), gp, session_name)
        st.session_state.session = session
        st.session_state.loaded_meta = f"{session.event['EventName']} {session.name} ({session.event.year})"
        if not os.getenv("DEMO_NO_LLM"):
            st.session_state.agent = build_agent(session=session)
        else:
            st.session_state.agent = None
        st.session_state.messages = []  # reset chat
        st.success(f"Loaded: {st.session_state.loaded_meta}")
    except Exception as e:
        st.error(f"Failed to load session: {e}")

# ใช้ตัวแปรเดียวทั่วไฟล์
session = st.session_state.session

# --- Main area: กราฟ ---
st.subheader("📈 Lap Time Charts")
col1, col2 = st.columns(2)

def plot_driver(driver_code: str, column, session):
    if not driver_code:
        return
    with column:
        st.markdown(f"**{driver_code} — Lap Time (s)**")
        if not session:
            st.info("Load a session first (เลือกปี/สนาม/เซสชัน แล้วกด Load).")
            return
        df = lap_summary(session, driver_code)
        if df.empty:
            st.warning(f"No laps found for {driver_code} in this session.")
            return
        fig = px.line(df, x="LapNumber", y="LapTime_s", title=None)
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{driver_code}")
        st.dataframe(df.tail(8), use_container_width=True, hide_index=True, key=f"table_{driver_code}")

# ถ้า A=B ให้แสดงชุดเดียวกันเพื่อกันชน key
if driver_b == driver_a and driver_b:
    st.info("Driver B เหมือนกับ Driver A — แสดงกราฟเพียงชุดเดียว")
    driver_b = ""

plot_driver(driver_a, col1, session)
plot_driver(driver_b, col2, session)

# ---------- Delta (A - B) ----------
st.divider()
st.subheader("Δ LapTime (A − B)")

if session and driver_a and driver_b:
    delta_df = build_delta_vs_time(session, driver_a, driver_b)
    if not delta_df.empty:
        fig_delta = px.line(delta_df, x="LapNumber", y="Delta_s",
                            title=f"Δ LapTime  ({driver_a} − {driver_b})  (negative = A faster)")
        st.plotly_chart(fig_delta, use_container_width=True, key="plot_delta")
        st.dataframe(delta_df.tail(12), use_container_width=True, hide_index=True, key="table_delta")
    else:
        st.info(f"No aligned laps for {driver_a} vs {driver_b} in this session.")
else:
    st.caption("เลือก Driver A และ B แล้วกด Load session เพื่อดูกราฟ Δ")

# ---------- Stint Summary ----------
st.divider()
st.subheader("🛞 Stint Summary")

col_s1, col_s2 = st.columns(2)
with col_s1:
    if session and driver_a:
        sA = stint_summary(session, driver_a)
        if not sA.empty:
            st.markdown(f"**{driver_a} — Stints**")
            st.dataframe(sA, use_container_width=True, hide_index=True, key=f"stints_{driver_a}")
        else:
            st.info(f"No stint data for {driver_a}.")
with col_s2:
    if session and driver_b:
        sB = stint_summary(session, driver_b)
        if not sB.empty:
            st.markdown(f"**{driver_b} — Stints**")
            st.dataframe(sB, use_container_width=True, hide_index=True, key=f"stints_{driver_b}")
        else:
            st.info(f"No stint data for {driver_b}.")

# ---------- Chat ----------
st.divider()
st.subheader("💬 Chat with AI Race Engineer")

# แสดงประวัติ
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

prompt = st.chat_input("พิมพ์ถาม เช่น 'lap summary VER', 'VER vs LEC', 'VER pit loss 20'")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    session = st.session_state.session

    if not session:
        answer = "Load a session first (เลือกปี/สนาม/เซสชันแล้วกด Load)."
    elif st.session_state.agent:
        q = prompt.strip()
        qlow = q.lower()
        try:
            if qlow.startswith("lap summary"):
                drv = q.split()[-1].upper()
                answer = st.session_state.agent.run(f"Use tool telemetry_query with input '{drv}'")
            elif " vs " in qlow:
                pair = q.replace("VS", "vs")
                answer = st.session_state.agent.run(f"Use tool delta_compare with input '{pair}'")
            elif qlow.startswith("stint summary"):
                drv = q.split()[-1].upper()
                answer = st.session_state.agent.run(f"Use tool stint_summary with input '{drv}'")
            elif "pit loss" in qlow and any(x in qlow for x in ["undercut", "overcut"]):
                answer = st.session_state.agent.run(f"Use tool kb_ask with input '{q}'")
            elif "pit loss" in qlow:
                answer = st.session_state.agent.run(f"Use tool strategy_simulator with input '{q}'")
            elif any(x in qlow for x in ["undercut", "overcut", "tyre", "ยาง", "intermediate", "full wet"]):
                answer = st.session_state.agent.run(f"Use tool kb_ask with input '{q}'")
            elif " vs " in qlow and "pit loss" in qlow and "under" in qlow:
                answer = st.session_state.agent.run(f"Use tool undercut_evaluator with input '{q}'")

            else:
                answer = st.session_state.agent.run(q)
        except Exception as e:
            answer = f"Agent error: {e}"


    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

