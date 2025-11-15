# 🏎️ AI Race Engineer Agent  
> **"Turning lap data into race-winning intelligence."**  
> A full-stack AI-driven dashboard that logs, analyzes, and explains race telemetry — simulating a real race engineer’s job with data, analytics, and conversational AI.

<p align="center">
  <img src="assets/cover-hero.png" alt="AI Race Engineer Dashboard" width="860" />
</p>

<p align="center">
  <a href="#">![version](https://img.shields.io/badge/version-1.0.0-black)</a>
  <a href="#">![build](https://img.shields.io/badge/build-passing-brightgreen)</a>
  <a href="#">![license](https://img.shields.io/badge/license-MIT-blue)</a>
  <a href="#">![stack](https://img.shields.io/badge/Stack-Streamlit%20·%20LangChain%20·%20Ollama%20·%20FastF1%20·%20Plotly-informational)</a>
</p>

---

## ✨ Project Highlights

* 🧠 **AI-Assisted Engineer:** วิเคราะห์ lap, stint, pit strategy และตอบคำถามเหมือน Race Engineer จริง ๆ  
* 🧾 **Real-time Event Logging:** บันทึกทุก lap, chat, stint ลง `race_log.json`  
* 📊 **Interactive Dashboard:** แสดง LapTime, Δ (A–B), Boxplot, และ Strategy Simulation  
* 💬 **Conversational Agent:** AI ที่เข้าใจคำถาม เช่น “VER vs LEC” หรือ “fastest lap summary”  
* 🧾 **Auto Reporting:** สร้าง PDF / Excel รายงานผลแบบอัตโนมัติ  
* 💡 **End-to-End Workflow:** FastF1 → JSON Log → Analytics Tools → Dashboard → AI Chat  

---

## 🏗 Architecture Overview
```mermaid
flowchart TD
  subgraph Data
    A[FastF1 Telemetry] --> B[race_logger.py]
    B --> C[(race_log.json)]
  end
  subgraph Processing
    C --> D[telemetry_tools.py]
    D --> E[auto_report.py]
    D --> F[eval_agent.py]
  end
  subgraph UI
    F --> G[Streamlit App]
    G --> H1[1_LapViewer.py]
    G --> H2[2_Dashboard.py]
    G --> H3[AI Chat Interface]
  end
  subgraph AI
    I[LangChain + Ollama]
    H3 <--> I
  end
````

---

## ⚙️ Core Components

### 🧾 race_logger.py

Custom event logger ที่เก็บทุกเหตุการณ์ของระบบลงใน `race_log.json`

```python
log_event("lap_completed", {"driver": "VER", "lap": 32, "lap_time": "92.3"})
```

### 📈 1_LapViewer.py

หน้าแสดงข้อมูล LapTime และ Δ (A–B) พร้อมกราฟแบบ interactive

* โหลดข้อมูลจาก FastF1
* แสดง Lap Chart, Delta Chart, Stint Summary
* เชื่อม AI Chat Agent เพื่อถามข้อมูลได้โดยตรง

### 📊 2_Dashboard.py

Dashboard สรุปข้อมูลทั้งหมด

* Fastest Lap, Average Lap Time
* Boxplot Consistency, Δ Comparison
* Strategy Simulation Table
* ปุ่ม Export CSV / Excel / PDF

### 🤖 agents/agent.py

โมดูล AI Agent เชื่อมกับ LangChain + Ollama

* ใช้ `initialize_agent()` และ tools เช่น

  * `lap_summary()`
  * `stint_summary()`
  * `delta_compare()`
  * `strategy_simulator()`
* ทำให้ AI เข้าใจคำถามผู้ใช้และเรียก tool ที่เหมาะสม

### 🧾 auto_report.py

สร้างรายงานสรุปผลอัตโนมัติ (PDF/Excel) พร้อมกราฟจาก `matplotlib`

### 🧮 strategy_sim.py

จำลองกลยุทธ์ 1-stop / 2-stop และ export เป็น Excel
ใช้สูตร:

```
total_time = Σ(laps * avg_lap_time) + (pit_loss × n_pit)
```

### 📊 visualize.py

สร้างกราฟ Lap Time / Stint Avg / Delta Chart
และบันทึกลงใน `data/plots/`

---

## 🧰 Tech Stack

| Layer           | Technologies                  |
| --------------- | ----------------------------- |
| Frontend        | Streamlit, Plotly, Matplotlib |
| Backend         | Python, FastF1                |
| AI Agent        | LangChain, Ollama (Llama3)    |
| Data            | JSON, Pandas, Excel           |
| Reporting       | FPDF2, XlsxWriter             |
| Deployment      | Streamlit Cloud               |
| Version Control | GitHub                        |

---

## 🚀 Getting Started

```bash
git clone https://github.com/Dxrksvng/AI-Race-Engineer-Agent.git
cd AI-Race-Engineer-Agent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui/app.py
```

เปิดหน้าเว็บที่: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
AI-Race-Engineer-Agent/
│
├── ui/
│   ├── app.py
│   └── pages/
│       ├── 1_LapViewer.py
│       ├── 2_Dashboard.py
│
├── agents/
│   └── agent.py
│
├── tools/
│   └── telemetry_tools.py
│
├── race_logger.py
├── scripts/
│   ├── auto_report.py
│   ├── eval_agent.py
│   ├── strategy_sim.py
│   └── visualize.py
├── requirements.txt
└── data/
    ├── eval/
    ├── logs/
    └── plots/
```

---

## 🧠 Key Learnings

* ออกแบบ Data Pipeline จากการเก็บ → วิเคราะห์ → แสดงผล → รายงาน
* เข้าใจการเชื่อม **LLM กับเครื่องมือวิเคราะห์ข้อมูลจริง**
* ออกแบบระบบ Dashboard + Chat AI ให้ทำงานร่วมกันใน workflow เดียว
* ใช้ FastF1 และ JSON log เป็น data source แบบ lightweight
* Deploy ระบบให้ HR / อาจารย์เข้าดูได้จริงผ่าน Streamlit Cloud

---

## 🌐 Live Demo

> 🚀 [**ai-race-engineer-agent.streamlit.app**](https://ai-race-engineer-agent-byjj.streamlit.app/)
> *(เปิดดูหน้า Lap Viewer, Dashboard, และ AI Chat ได้แบบ public)*

---

## 👩‍💻 Developer

**Nattakamon Jaimetha (เจ)**
> 🎓 Data Science & Business Analytics — KMITL (Data Engineering Track)
> 💡 สนใจด้าน AI, Data Engineering, Motorsport Analytics

> 🌐 GitHub: [github.com/Dxrksvng](https://github.com/Dxrksvng)
> 💼 LinkedIn: [Nattakamon Jaimetha](https://www.linkedin.com/in/nattakamon-jaimetha/)
> 📧 Email: [nattakamon.j@gmail.com](mailto:nattakamon0208@gmail.com)

---

> 🏁 *“Data never lies — but only the fastest can interpret it.”*
> — *AI Race Engineer Agent © 2025*
