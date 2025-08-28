# tools/telemetry_tools.py
from __future__ import annotations
import os
import pandas as pd
import numpy as np
import fastf1

CACHE_DIR = os.getenv("FASTF1_CACHE", "data/cache")

def _enable_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

def load_session(year: int, gp: str, session_name: str):
    """
    ตัวโหลดเซสชัน F1 เช่น (2024, 'Bahrain', 'Q')
    จะโหลด laps และ weather มาให้ พร้อมแคชลง data/cache
    """
    _enable_cache()
    session = fastf1.get_session(year, gp, session_name)
    session.load(laps=True, telemetry=False, weather=True)
    return session

def _td_to_sec(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds()

def lap_summary(session, driver: str) -> pd.DataFrame:
    """
    สรุป Lap ของไดรเวอร์: LapNumber, LapTime(s), Sector1/2/3(s), Compound, Stint
    """
    laps = session.laps.pick_driver(driver).copy()
    if laps.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "LapNumber": laps["LapNumber"].values,
        "LapTime_s": _td_to_sec(laps["LapTime"]),
        "S1_s": _td_to_sec(laps["Sector1Time"]),
        "S2_s": _td_to_sec(laps["Sector2Time"]),
        "S3_s": _td_to_sec(laps["Sector3Time"]),
        "Compound": laps.get("Compound", pd.Series(["?"]*len(laps))).values,
        "Stint": laps.get("Stint", pd.Series([np.nan]*len(laps))).values,
    })
    return df.dropna(subset=["LapTime_s"])

def stint_summary(session, driver: str) -> pd.DataFrame:
    """
    รวมสรุปการใช้ยางเป็นสติ้นท์: Stint, Compound, Laps, AvgLapTime_s, BestLap_s
    """
    df = lap_summary(session, driver)
    if df.empty:
        return df
    gb = df.groupby(["Stint", "Compound"], dropna=False)
    out = gb.agg(
        Laps=("LapNumber", "count"),
        AvgLapTime_s=("LapTime_s", "mean"),
        BestLap_s=("LapTime_s", "min")
    ).reset_index().sort_values(["Stint", "Compound"])
    return out

def build_delta_vs_time(session, driver_a: str, driver_b: str) -> pd.DataFrame:
    """
    สร้างตาราง Delta LapTime (A - B) ต่อ LapNumber โดยจับคู่ลาพ์ที่ตรงกัน
    """
    a = lap_summary(session, driver_a)
    b = lap_summary(session, driver_b)
    if a.empty or b.empty:
        return pd.DataFrame()

    merged = pd.merge(
        a[["LapNumber", "LapTime_s"]],
        b[["LapNumber", "LapTime_s"]],
        on="LapNumber",
        how="inner",
        suffixes=(f"_{driver_a}", f"_{driver_b}")
    )
    if merged.empty:
        return pd.DataFrame()

    merged["Delta_s"] = merged[f"LapTime_s_{driver_a}"] - merged[f"LapTime_s_{driver_b}"]
    return merged[["LapNumber", "Delta_s"]].sort_values("LapNumber")

def suggest_pit_lap_simple(session, driver: str, pit_loss_s: float = 20.0) -> dict:
    """
    แนะนำรอบพิทแบบ heuristic ง่าย ๆ:
    - หา lap ที่ pace ช้าลง (degradation) เกิน threshold
    - เปรียบเทียบกับ pit loss
    คืนค่า dict {recommend_lap, reason}
    """
    df = lap_summary(session, driver)
    if df.empty:
        return {"recommend_lap": None, "reason": f"No data for {driver}"}

    # คำนวณ delta ต่อ lap
    df = df.sort_values("LapNumber")
    df["LapDelta"] = df["LapTime_s"].diff()

    # หา lap ที่ช้าลง > 0.25s/lap ต่อเนื่อง
    degradation = df[df["LapDelta"] > 0.25]
    if not degradation.empty:
        lap = int(degradation["LapNumber"].iloc[0])
        return {
            "recommend_lap": lap,
            "reason": f"Pace drop detected (Δ>{0.25:.2f}s) around lap {lap}"
        }

    # fallback: แนะนำกลาง stint
    avg_lap = df["LapTime_s"].mean()
    best_lap = df["LapTime_s"].min()
    reason = f"Stable pace (best={best_lap:.2f}, avg={avg_lap:.2f}), no urgent pit need"
    return {"recommend_lap": int(df['LapNumber'].median()), "reason": reason}

def evaluate_undercut_simple(session, attacker: str, defender: str, pit_loss_s: float = 20.0) -> dict:
    """คำนวณ undercut แบบเร็ว ๆ จาก lap_summary:
    - ประเมิน deg ของ defender จากสโลป lap ล่าสุด
    - กำไรโจมตี ≈ max(0, defender_deg_per_lap) * HORIZON  (เช่น 1–2 laps)
    - ถ้า กำไร > pit_loss → viable
    """
    HORIZON = 2  # ลอง 2 laps
    a = lap_summary(session, attacker)
    d = lap_summary(session, defender)
    if a.empty or d.empty:
        return {"viable": None, "reason": "missing laps"}

    # สโลปเสื่อมของ defender จากเส้นตรงช่วงท้าย
    tail = d.tail(min(8, len(d)))  # ใช้ 6–8 laps ท้ายสุด
    x = tail["LapNumber"].to_numpy(dtype=float)
    y = tail["LapTime_s"].to_numpy(dtype=float)
    if len(x) < 3:
        return {"viable": None, "reason": "too few laps for defender"}

    slope = float(np.polyfit(x, y, 1)[0])  # s/lap
    defender_deg = max(0.0, slope)
    expected_gain = defender_deg * HORIZON

    viable = expected_gain > pit_loss_s
    reason = f"defender_deg≈{defender_deg:.3f}s/lap, horizon={HORIZON}, gain≈{expected_gain:.1f}s vs pit_loss={pit_loss_s:.1f}s"
    return {"viable": viable, "expected_gain_s": expected_gain, "pit_loss_s": pit_loss_s, "reason": reason}
