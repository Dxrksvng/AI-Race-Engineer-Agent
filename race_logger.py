# ui/logger.py
from __future__ import annotations
import os, json, datetime
from typing import Optional, Dict, Any, List

LOG_DIR = os.getenv("LOG_DIR", "data/logs")
LOG_FILE = os.getenv("LOG_FILE", os.path.join(LOG_DIR, "chat.jsonl"))

def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

def log_event(event: Dict[str, Any], path: Optional[str] = None) -> str:
    """
    Append one JSON line to the log file.
    Auto-add 'ts' if missing. Returns file path used.
    """
    _ensure_log_dir()
    fp = path or LOG_FILE
    event = dict(event)
    event.setdefault("ts", datetime.datetime.now().isoformat(timespec="seconds"))
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return fp

def read_events(path: Optional[str] = None, max_lines: int = 1000) -> List[Dict[str, Any]]:
    """
    Read last up-to max_lines events (best-effort).
    """
    fp = path or LOG_FILE
    if not os.path.exists(fp):
        return []
    out: List[Dict[str, Any]] = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # skip broken lines
                pass
    return out[-max_lines:]
