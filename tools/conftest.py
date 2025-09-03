# tests/conftest.py
from __future__ import annotations
import sys, os

# ชี้ให้ Python เห็นโฟลเดอร์ราก (ที่มี tools/, agents/, ui/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ตั้งค่า cache ให้ FastF1 (ป้องกัน path แปลก ๆ ตอนรันทดสอบ)
os.environ.setdefault("FASTF1_CACHE", os.path.join(PROJECT_ROOT, "data", "cache"))
os.makedirs(os.environ["FASTF1_CACHE"], exist_ok=True)
