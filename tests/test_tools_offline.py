# tests/test_tools_offline.py
from __future__ import annotations
import math
import pytest

from tools.telemetry_tools import (
    load_session, lap_summary, stint_summary, build_delta_vs_time
)

# ใช้เซสชันที่ดาวน์โหลด/แคชง่าย
YEAR, GP, SESSION = 2024, "Bahrain", "Q"
DRIVERS = ["VER", "LEC"]  # เปลี่ยน/เพิ่มภายหลังได้

@pytest.fixture(scope="session")
def session():
    s = load_session(YEAR, GP, SESSION)
    assert s is not None
    return s

@pytest.mark.parametrize("drv", DRIVERS)
def test_lap_summary_schema(session, drv):
    df = lap_summary(session, drv)
    # คอลัมน์หลักต้องมี
    for col in ["LapNumber", "LapTime_s", "S1_s", "S2_s", "S3_s", "Compound", "Stint"]:
        assert col in df.columns
    # ต้องมีอย่างน้อย 1 แถว
    assert len(df) > 0
    # เวลาลาพ์ต้องเป็นค่าบวกทั้งหมด
    assert (df["LapTime_s"] > 0).all()

@pytest.mark.parametrize("drv", DRIVERS)
def test_stint_summary_basic(session, drv):
    stints = stint_summary(session, drv)
    # อนุโลมไม่มีสติ้นท์ในบางเซสชัน แต่ถ้ามีต้องมีคอลัมน์ครบ
    if len(stints) > 0:
        for col in ["Stint", "Compound", "Laps", "AvgLapTime_s", "BestLap_s"]:
            assert col in stints.columns
        assert (stints["Laps"] >= 1).all()

def test_delta_vs_time_alignment(session):
    a, b = DRIVERS
    delta = build_delta_vs_time(session, a, b)
    # อนุโลมให้ว่างได้ถ้าลาพ์ไม่ทับกัน แต่โดยทั่วไปควรมีบ้าง
    assert "LapNumber" in delta.columns and "Delta_s" in delta.columns
    if len(delta) > 0:
        # ค่าต้องไม่เป็น NaN และมีเหตุผล
        assert not delta["Delta_s"].isna().any()
        # ป้องกันค่าเพี้ยน: |delta| ไม่ควรเกิน ~60s ในควอลิ
        assert (delta["Delta_s"].abs() < 60).all()
