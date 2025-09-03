# scripts/auto_report.py
import pandas as pd
import os
from fpdf import FPDF

# ---------------- Excel Report ----------------
# โหลดข้อมูล
df_summary = pd.read_excel("data/eval/report_summary.xlsx", sheet_name="Summary")
df_strategy = pd.read_excel("data/eval/strategy_result.xlsx")

# Export Excel ที่รวม Summary + Strategy
output_xlsx = "data/eval/auto_report.xlsx"
with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    df_strategy.to_excel(writer, sheet_name="Strategy", index=False)

print(f"✅ Exported Excel report to {output_xlsx}")

# ---------------- PDF Report ----------------
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Race Engineer Auto Report", border=False, ln=1, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

pdf = PDF()
pdf.add_page()

# Summary Table
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Evaluation Summary", ln=1)
pdf.set_font("Arial", "", 10)

for i, row in df_summary.iterrows():
    pdf.cell(60, 8, str(row["Metric"]), 1)
    pdf.cell(40, 8, str(row["Value"]), 1, ln=1)

pdf.ln(10)

# Strategy Table
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Strategy Simulation", ln=1)
pdf.set_font("Arial", "", 10)

col_widths = [30, 40, 40, 40]
headers = ["Driver", "1-stop_time", "2-stop_time", "BestStrategy"]

# Header row
for h, w in zip(headers, col_widths):
    pdf.cell(w, 8, h, 1, 0, "C")
pdf.ln()

# Data rows
for _, row in df_strategy.iterrows():
    pdf.cell(col_widths[0], 8, str(row["Driver"]), 1)
    pdf.cell(col_widths[1], 8, str(round(row["1-stop_time"], 2)), 1)
    pdf.cell(col_widths[2], 8, str(round(row["2-stop_time"], 2)) if not pd.isna(row["2-stop_time"]) else "-", 1)
    pdf.cell(col_widths[3], 8, str(row["BestStrategy"]), 1)
    pdf.ln()

pdf.ln(10)

# Insert Plots if exist
plot_dir = "data/plots"
plots = [
    ("lap_time.png", "Lap Time Comparison"),
    ("stint_avg.png", "Average Lap Time per Stint"),
    ("consistency_boxplot.png", "Lap Time Consistency (Boxplot)"),
    ("delta_VER_vs_LEC.png", "Delta Chart: VER vs LEC"),
]

for fname, title in plots:
    path = os.path.join(plot_dir, fname)
    if os.path.exists(path):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, title, ln=1)
        pdf.image(path, w=170)
        pdf.ln(5)

# Save PDF
output_pdf = "data/eval/auto_report.pdf"
pdf.output(output_pdf)
print(f"✅ Exported PDF report to {output_pdf}")
