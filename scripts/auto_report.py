import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from race_logger import read_events

OUTPUT_DIR = "data/eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Data Prep ----------------
def load_lap_data():
    events = read_events()
    laps = [e["data"] for e in events if e["event"] == "lap_completed"]

    if not laps:
        return pd.DataFrame()

    df = pd.DataFrame(laps)

    # clean & convert
    df["lap_time"] = pd.to_timedelta(df["lap_time"], errors="coerce").dt.total_seconds()
    df = df.dropna(subset=["lap_time"])
    df = df[df["lap_time"] < 200]  # ตัด invalid / pit laps

    return df

# ---------------- Charts ----------------
def generate_charts(df):
    charts = []

    # Lap Time Trend
    fig, ax = plt.subplots(figsize=(6,4))
    for drv in df["driver"].unique():
        subset = df[df["driver"] == drv].sort_values("lap")
        ax.plot(subset["lap"], subset["lap_time"], marker="o", label=drv)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title("Lap Time Trend")
    ax.legend()
    trend_path = os.path.join(OUTPUT_DIR, "lap_time_trend.png")
    plt.savefig(trend_path, bbox_inches="tight")
    charts.append(trend_path)
    plt.close(fig)

    # Boxplot Consistency
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df, x="driver", y="lap_time", ax=ax)
    ax.set_title("Lap Time Consistency")
    ax.set_ylabel("Lap Time (s)")
    ax.set_xlabel("Driver")
    plt.xticks(rotation=90)
    box_path = os.path.join(OUTPUT_DIR, "lap_time_consistency.png")
    plt.savefig(box_path, bbox_inches="tight")
    charts.append(box_path)
    plt.close(fig)

    # Delta Chart (เลือก VER vs LEC ถ้ามี)
    if {"VER", "LEC"}.issubset(set(df["driver"].unique())):
        laps_a = df[df["driver"] == "VER"][["lap", "lap_time"]].set_index("lap")
        laps_b = df[df["driver"] == "LEC"][["lap", "lap_time"]].set_index("lap")
        delta = laps_a.join(laps_b, lsuffix="_VER", rsuffix="_LEC")
        delta["delta"] = delta["lap_time_VER"] - delta["lap_time_LEC"]

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(delta.index, delta["delta"], marker="o")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xlabel("Lap")
        ax.set_ylabel("Δ LapTime (VER - LEC) [s]")
        ax.set_title("Delta Chart: VER vs LEC")
        delta_path = os.path.join(OUTPUT_DIR, "delta_ver_lec.png")
        plt.savefig(delta_path, bbox_inches="tight")
        charts.append(delta_path)
        plt.close(fig)

    return charts

# ---------------- PDF ----------------
def generate_pdf(df, charts):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Race Engineer Auto Report", ln=True, align="C")

    # Summary
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    summary = df.groupby("driver")["lap_time"].agg(["mean", "min", "max"]).round(3)
    pdf.cell(0, 10, "Lap Time Summary (s):", ln=True)
    for drv, row in summary.iterrows():
        pdf.cell(0, 10,
                 f"Driver {drv}: Avg={row['mean']} | Best={row['min']} | Worst={row['max']}",
                 ln=True)

    # Charts
    for chart in charts:
        pdf.add_page()
        pdf.image(chart, x=10, y=20, w=180)

    out_file = os.path.join(OUTPUT_DIR, "auto_report.pdf")
    pdf.output(out_file)
    print(f"✅ PDF report saved: {out_file}")

# ---------------- Excel ----------------
def generate_excel(df):
    out_file = os.path.join(OUTPUT_DIR, "auto_report.xlsx")
    with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Lap Data")
        summary = df.groupby("driver")["lap_time"].agg(["mean", "min", "max"]).round(3)
        summary.to_excel(writer, sheet_name="Summary")
    print(f"✅ Excel report saved: {out_file}")

# ---------------- Main ----------------
def main():
    df = load_lap_data()
    if df.empty:
        print("⚠️ No lap data found in race_log.json")
        return

    charts = generate_charts(df)
    generate_pdf(df, charts)
    generate_excel(df)

if __name__ == "__main__":
    main()
