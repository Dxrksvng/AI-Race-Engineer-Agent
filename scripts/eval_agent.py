# scripts/eval_agent.py
from __future__ import annotations
import sys, os, time, re
from pathlib import Path

# ให้สคริปต์เห็นแพ็กเกจในรากโปรเจกต์
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ใช้ FastF1 cache เดียวกันทุกที่
os.environ.setdefault("FASTF1_CACHE", "data/cache")

from tools.telemetry_tools import (
    load_session,
    lap_summary,
    stint_summary,
    build_delta_vs_time,
    evaluate_undercut_simple,
)

OUT_DIR = Path(os.getenv("EVAL_DIR", "data/eval"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_CSV = OUT_DIR / "report.csv"

YEAR, GP, SESSION = 2024, "Bahrain", "Q"
A, B = "VER", "LEC"

PROMPTS = [
    "lap summary VER",
    "stint summary VER",
    "VER vs LEC",
    "VER pit loss 20",
    "undercut VER vs LEC pit loss 20",
    "When is an undercut viable with pit loss 20s?",
]

def _ascii(s: str) -> str:
    """ทำความสะอาดตัวอักษรพิเศษเป็น ASCII เพื่อไม่ให้ mojibake ใน CSV/Excel"""
    if not isinstance(s, str):
        return s
    return (
        s.replace("Δ", "d")
         .replace("±", "+/-")
         .replace("≈", "~")
         .replace("−", "-")
         .replace("–", "-")
         .replace("—", "-")
    )

# ------------------------------ Router (เรียก tools ตรง ๆ) ------------------------------
def route_direct(session, q: str) -> str:
    q_stripped = q.strip()
    qlow = q_stripped.lower()

    # 1) lap summary AAA
    if qlow.startswith("lap summary"):
        drv = q_stripped.split()[-1].upper()
        df = lap_summary(session, drv)
        if df.empty:
            return f"(tool: telemetry_query) {drv}: no laps"
        best = df["LapTime_s"].min()
        avg  = df["LapTime_s"].mean()
        return f"(tool: telemetry_query) {drv}: rows={len(df)} best={best:.3f}s avg={avg:.3f}s"

    # 2) stint summary AAA
    if qlow.startswith("stint summary"):
        drv = q_stripped.split()[-1].upper()
        df = stint_summary(session, drv)
        if df.empty:
            return f"(tool: stint_summary) {drv}: no stints"
        head = df.head(3).to_string(index=False)
        return f"(tool: stint_summary)\n{head}"

    # 3) AAA vs BBB  (delta)
    if " vs " in qlow and "pit loss" not in qlow and "undercut" not in qlow:
        left, right = [p.strip().upper() for p in q_stripped.replace("VS", "vs").split(" vs ", 1)]
        df = build_delta_vs_time(session, left, right)
        if df.empty:
            return f"(tool: delta_compare) No aligned laps for {left} vs {right}"
        mean_delta = df["Delta_s"].mean()
        return f"(tool: delta_compare) {left}-{right} mean d={mean_delta:.3f}s samples={len(df)}"

    # 4) AAA pit loss X  (ยอมทุก spacing/case)
    if "pit loss" in qlow and "undercut" not in qlow:
        # รองรับ: "VER pit loss 20", "ver  pit   loss   19.5"
        m = re.search(r"\b([A-Z]{2,3})\b.*?\bpit\s*loss\b.*?([0-9]+(?:\.[0-9]+)?)", q_stripped, flags=re.I)
        if not m:
            return "(tool: strategy_simulator) Format: 'VER pit loss 20'"
        drv = m.group(1).upper()
        pit_loss = float(m.group(2))
        df = lap_summary(session, drv)
        if df.empty:
            return f"(tool: strategy_simulator) {drv}: no laps"
        best = df["LapTime_s"].min()
        return f"(tool: strategy_simulator) {drv}: pit_loss~{pit_loss:.1f}s, ref_best~{best:.3f}s"

    # 5) undercut AAA vs BBB pit loss X  (ยอมทุก spacing/case)
    if "undercut" in qlow and "vs" in qlow and "pit loss" in qlow:
        # รองรับ: "undercut VER vs LEC pit loss 20", "undercut   ver VS  lec  pit  loss  18.5"
        m = re.search(
            r"\bundercut\b.*?\b([A-Z]{2,3})\b.*?\bvs\b.*?\b([A-Z]{2,3})\b.*?\bpit\s*loss\b.*?([0-9]+(?:\.[0-9]+)?)",
            q_stripped,
            flags=re.I,
        )
        if not m:
            return "(tool: undercut_evaluator) Format: 'undercut VER vs LEC pit loss 20'"
        a, b, pit = m.group(1).upper(), m.group(2).upper(), float(m.group(3))
        res = evaluate_undercut_simple(session, a, b, pit_loss_s=pit)
        if res["viable"] is None:
            return f"(tool: undercut_evaluator) insufficient data: {res['reason']}"
        reason_ascii = _ascii(res["reason"])
        return (f"(tool: undercut_evaluator) {a} vs {b} | "
                f"viable={res['viable']} | gain~{res['expected_gain_s']:.2f}s vs pit_loss={res['pit_loss_s']:.1f}s | "
                f"{reason_ascii}")

    # 6) คำถามทฤษฎี (ตอบสั้น ๆ)
    if "undercut" in qlow and "pit loss" in qlow:
        return "(kb) Undercut is viable when time gained after pit > pit loss. Typical pit loss is ~18-23s."

    # default
    return "(eval) No route matched."
# -----------------------------------------------------------------------------------------

def main():
    print(f"Loading session {YEAR} {GP} {SESSION} ...")
    session = load_session(YEAR, GP, SESSION)

    # Warm cache
    _ = lap_summary(session, A)
    _ = lap_summary(session, B)

    truth = {
        "lap_best_VER": float(lap_summary(session, A)["LapTime_s"].min()),
        "stints_VER_rows": int(len(stint_summary(session, A))),
        "delta_rows": int(len(build_delta_vs_time(session, A, B))),
    }

    rows = []
    for q in PROMPTS:
        t0 = time.perf_counter()
        err = ""
        try:
            ans = route_direct(session, q)
            ok = 1
        except Exception as e:
            ans = ""
            ok = 0
            err = str(e)
        ms = round(1000 * (time.perf_counter() - t0), 2)
        rows.append({
            "question": q,
            "ok": ok,
            "latency_ms": ms,
            "answer_len": len(ans),
            "error": err,
            "answer": _ascii(ans)[:300].replace("\n", " "),
        })
        print(f"[{ok}] {q:40s} {ms:8.2f} ms  {err if err else ''}")

    # ----------------- Save report in multiple formats -----------------
    import pandas as pd

    df_out = pd.DataFrame(rows)
    for k, v in truth.items():
        df_out[k] = v

    # ทำให้ทุกคอลัมน์ที่เป็น string เป็น ASCII-friendly อีกรอบ
    for col in df_out.columns:
        if df_out[col].dtype == object:
            df_out[col] = df_out[col].map(_ascii)

    # 1) CSV (utf-8-sig + BOM สำหรับ Excel)
    df_out.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    # 2) TSV (แท็บคั่น) – เปิดได้สวยใน Excel/Numbers
    tsv_path = OUT_DIR / "report.tsv"
    df_out.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")

    # 3) XLSX (ดีที่สุดสำหรับ Excel) – ถ้าไม่มี openpyxl จะข้ามและแจ้งข้อความ
    xlsx_path = OUT_DIR / "report.xlsx"
    wrote_xlsx = True
    try:
        df_out.to_excel(xlsx_path, index=False)  # requires openpyxl
    except Exception as e:
        wrote_xlsx = False
        print(f"(skip XLSX) {e}  -> pip install openpyxl  ถ้าอยากได้ไฟล์ .xlsx")

    print(f"\nSaved report: {REPORT_CSV.resolve()}")
    print(f"Also wrote:   {tsv_path.resolve()}")
    if wrote_xlsx:
        print(f"             {xlsx_path.resolve()}")

if __name__ == "__main__":
    main()
def main():
    # ... โค้ดเดิมที่สร้าง report.csv ...
    print("Saved report: data/eval/report.csv")

# ---------------- เพิ่มฟังก์ชันใหม่ตรงนี้ ----------------
import pandas as pd

def export_eval_to_excel():
    # โหลดผลลัพธ์จาก report.csv
    df = pd.read_csv("data/eval/report.csv")

    # คำนวณ summary
    total = len(df)
    passed = df["ok"].sum()
    failed = total - passed
    accuracy = passed / total * 100
    avg_latency = df["latency_ms"].mean()
    max_latency = df["latency_ms"].max()
    min_latency = df["latency_ms"].min()

    summary_data = {
        "Metric": [
            "Total Questions",
            "Passed",
            "Failed",
            "Accuracy (%)",
            "Avg Latency (ms)",
            "Max Latency (ms)",
            "Min Latency (ms)"
        ],
        "Value": [
            total,
            passed,
            failed,
            round(accuracy, 2),
            round(avg_latency, 2),
            max_latency,
            min_latency
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # สร้างไฟล์ Excel ที่มี 2 sheet
    output_path = "data/eval/report_summary.xlsx"
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"✅ Exported evaluation summary to {output_path}")
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
    export_eval_to_excel()   # เรียกหลัง main()

# ---------------- เพิ่มโค้ดท้ายไฟล์ eval_agent.py ----------------
def export_raw_data(session, drivers):
    """Export lap-by-lap และ stint summary ของ driver ที่เลือก"""
    all_laps = []
    all_stints = []

    for drv in drivers:
        # lap summary (lap, laptime)
        laps_df = lap_summary(session, drv)
        if not laps_df.empty:
            laps_df["Driver"] = drv
            all_laps.append(laps_df)

        # stint summary (stint, avg pace)
        stints_df = stint_summary(session, drv)
        if not stints_df.empty:
            stints_df["Driver"] = drv
            all_stints.append(stints_df)

    # รวมทุก driver
    df_laps = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
    df_stints = pd.concat(all_stints, ignore_index=True) if all_stints else pd.DataFrame()

    # Export
    output_path = "data/eval/lap_times.xlsx"
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        if not df_laps.empty:
            df_laps.to_excel(writer, sheet_name="LapTimes", index=False)
        if not df_stints.empty:
            df_stints.to_excel(writer, sheet_name="StintSummary", index=False)

    print(f"✅ Exported lap & stint data to {output_path}")


if __name__ == "__main__":
    main()
    export_eval_to_excel()   # ของเดิม
    session = load_session(YEAR, GP, SESSION)
    export_raw_data(session, [A, B])  # export lap/stint ของ VER, LEC
