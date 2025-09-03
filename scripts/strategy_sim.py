import pandas as pd

# โหลดข้อมูลจาก report summary
df = pd.read_excel("data/eval/report_summary.xlsx")

# ฟังก์ชันจำลองเวลารวมของ strategy
def simulate_strategy(driver, stints, pit_loss=20):
    """
    stints = [(laps, avg_lap_time), ...]
    """
    total_time = 0
    for laps, avg_lap_time in stints:
        total_time += laps * avg_lap_time
    total_time += (len(stints)-1) * pit_loss  # add pit loss
    return total_time

# Example: VER 2-stop vs 1-stop
ver_stints_1stop = [(30, 92.5), (30, 93.1)]   # lap, avg lap time
ver_stints_2stop = [(20, 91.8), (20, 92.7), (20, 93.5)]

print("VER 1-stop:", simulate_strategy("VER", ver_stints_1stop))
print("VER 2-stop:", simulate_strategy("VER", ver_stints_2stop))

# scripts/strategy_sim.py
import pandas as pd

# ... (simulate_strategy และโค้ดคำนวณตามเดิม)

results = [
    {"Driver": "VER", "1-stop_time": 5588.0, "2-stop_time": 5600.0, "BestStrategy": "1-stop"}
    # ถ้ามีหลาย driver ก็ append ต่อไป
]

df_results = pd.DataFrame(results)

# Export Excel
output_path = "data/eval/strategy_result.xlsx"
df_results.to_excel(output_path, index=False)
print(f"✅ Exported strategy results to {output_path}")
