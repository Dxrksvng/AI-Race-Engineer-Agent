import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from race_logger import log_event
import fastf1

def log_session(year=2024, gp="Bahrain", session_type="Q"):
    session = fastf1.get_session(year, gp, session_type)
    session.load()

    for _, lap_data in session.laps.iterlaps():
        log_event(
            "lap_completed",
            {
                "driver": lap_data["Driver"],
                "lap": int(lap_data["LapNumber"]),
                "lap_time": str(lap_data["LapTime"])
            }
        )

    print(f"✅ Logged {len(session.laps)} laps from {year} {gp} {session_type}")

if __name__ == "__main__":
    log_session()
