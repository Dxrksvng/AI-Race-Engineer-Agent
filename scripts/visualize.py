# scripts/visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# สร้างโฟลเดอร์เก็บรูปถ้าไม่มี
os.makedirs("data/plots", exist_ok=True)

# โหลดข้อมูล lap-by-lap
df = pd.read_excel("data/eval/lap_times.xlsx", sheet_name="LapTimes")

# ---------------- กราฟ Lap Time ----------------
plt.figure(figsize=(10,6))
for driver in df["Driver"].unique():
    subset = df[df["Driver"] == driver]
    plt.plot(subset["LapNumber"], subset["LapTime_s"], label=driver)

plt.title("Lap Time Comparison")
plt.xlabel("Lap")
plt.ylabel("Lap Time (s)")
plt.legend()
plt.grid(True)
plt.savefig("data/plots/lap_time.png")
plt.show()

# ---------------- กราฟ Stint Avg ----------------
stints = pd.read_excel("data/eval/lap_times.xlsx", sheet_name="StintSummary")

plt.figure(figsize=(8,6))
for driver in stints["Driver"].unique():
    subset = stints[stints["Driver"] == driver]
    plt.bar(subset["Stint"], subset["AvgLapTime_s"], label=driver, alpha=0.7)

plt.title("Average Lap Time per Stint")
plt.xlabel("Stint")
plt.ylabel("Avg Lap Time (s)")
plt.legend()
plt.savefig("data/plots/stint_avg.png")
plt.show()

# ---------------- Boxplot Consistency ----------------
plt.figure(figsize=(10,6))
df_clean = df[df["LapTime_s"] < 200]  # กรอง lap ผิดปกติ
df_clean.boxplot(column="LapTime_s", by="Driver")
plt.title("Lap Time Consistency per Driver")
plt.suptitle("")
plt.xlabel("Driver")
plt.ylabel("Lap Time (s)")
plt.grid(True)
plt.savefig("data/plots/consistency_boxplot.png")
plt.show()

# ---------------- Delta Chart (เลือก driver pair) ----------------
def plot_delta(driver_a, driver_b, df, save_path=None):
    laps_a = df[df["Driver"] == driver_a][["LapNumber", "LapTime_s"]].set_index("LapNumber")
    laps_b = df[df["Driver"] == driver_b][["LapNumber", "LapTime_s"]].set_index("LapNumber")

    delta = laps_a.join(laps_b, lsuffix=f"_{driver_a}", rsuffix=f"_{driver_b}")
    delta["Delta_s"] = delta[f"LapTime_s_{driver_a}"] - delta[f"LapTime_s_{driver_b}"]

    plt.figure(figsize=(10,6))
    plt.plot(delta.index, delta["Delta_s"], marker="o")
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"Delta Chart: {driver_a} vs {driver_b}")
    plt.xlabel("Lap")
    plt.ylabel(f"Δ LapTime ({driver_a} - {driver_b}) [s]")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ตัวอย่าง: VER vs LEC, VER vs HAM
plot_delta("VER", "LEC", df, "data/plots/delta_VER_vs_LEC.png")
plot_delta("VER", "HAM", df, "data/plots/delta_VER_vs_HAM.png")
