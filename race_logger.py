import os
import json
from datetime import datetime

# กำหนด path ของไฟล์ log
LOG_FILE = os.path.join(os.path.dirname(__file__), "race_log.json")

def log_event(event_type, data):
    """เขียน event ลงไฟล์ log"""
    event = {
        "event": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

    # ถ้ามีไฟล์อยู่แล้ว -> โหลด
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                events = json.load(f)
            except json.JSONDecodeError:
                events = []
    else:
        events = []

    # เพิ่ม event ใหม่
    events.append(event)

    # เขียนกลับลงไฟล์
    with open(LOG_FILE, "w") as f:
        json.dump(events, f, indent=2)

def read_events():
    """อ่าน event ทั้งหมดจากไฟล์ log"""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
