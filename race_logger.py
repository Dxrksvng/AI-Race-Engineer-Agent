# race_logger.py
from __future__ import annotations
import os
import json
import datetime

def log_event(event: dict, file_path: str = "race_log.json"):
    """บันทึก event ใหม่ลงไฟล์ JSON"""
    events = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                events = json.load(f)
            except json.JSONDecodeError:
                events = []

    # เพิ่ม timestamp
    event["timestamp"] = datetime.datetime.now().isoformat()
    events.append(event)

    with open(file_path, "w") as f:
        json.dump(events, f, indent=2)

def read_events(file_path: str = "race_log.json"):
    """อ่าน events ทั้งหมดจากไฟล์ JSON"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
