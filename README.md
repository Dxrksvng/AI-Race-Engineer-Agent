# 🏎️ AI Race Engineer Dashboard
**Turn racing data into real-time strategy.**  
A modern end-to-end dashboard that ingests telemetry, predicts tyre & fuel windows, and recommends pit strategies — built for students, sim racers, and motorsport nerds who want pro-team tooling.
---
https://ai-race-engineer-agent-byjj.streamlit.app/
## 🚀 Overview
AI Race Engineer Dashboard คือระบบ Dashboard สำหรับวิเคราะห์ข้อมูลการแข่งขันรถ (Racing Telemetry) แบบเรียลไทม์ โดยใช้ Machine Learning ในการคาดการณ์กลยุทธ์ระหว่างแข่งขัน เช่น  
- การคาดการณ์รอบที่ควรเข้าพิต (Pit Stop Prediction)  
- การวิเคราะห์สภาพยาง (Tyre Degradation Model)  
- การบริหารน้ำมันเชื้อเพลิง (Fuel Window Estimation)  
- การเปรียบเทียบข้อมูลนักขับระหว่างทีม  

ระบบนี้ออกแบบมาสำหรับ **นักศึกษาด้านข้อมูล (Data Science), นักแข่ง Simulator, และทีม Motorsport Analyst** ที่ต้องการเครื่องมือระดับทีมแข่งมืออาชีพ

---

## ⚙️ Tech Stack
| Layer | Technology |
|-------|-------------|
| Frontend | Vue 3 + Vite + TailwindCSS + Chart.js |
| Backend | FastAPI + Python 3.10 |
| Database | PostgreSQL (Telemetry & Strategy Logs) |
| AI/ML | Scikit-learn / XGBoost / Prophet |
| Data Ingestion | Kafka (optional) / CSV / REST API |
| Visualization | Recharts / Plotly / D3.js |
| Deployment | Docker Compose + Nginx Reverse Proxy |

---

## 🧩 System Architecture
```
Driver → Telemetry API → FastAPI Backend → PostgreSQL
                                   ↓
                            ML Inference (Tyre/Fuel)
                                   ↓
                         Vue3 Dashboard Visualization
```

---

## 🧠 Core Features
- 📊 Real-time telemetry visualization (Speed, RPM, Gear, Fuel)
- ⚙️ Predictive tyre degradation model
- ⛽ Fuel strategy optimizer
- 🧮 Lap time delta analyzer
- 📈 Historical race comparison
- 🧠 AI-driven pit recommendation engine

---

## 🧰 Installation & Setup

### 1. Clone Repository  
```bash
git clone https://github.com/<your-username>/ai-race-engineer.git
cd ai-race-engineer
```

### 2. Setup Backend (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
> FastAPI backend will start at: `http://127.0.0.1:8000`

### 3. Setup Frontend (Vue3)
```bash
cd frontend
npm install
npm run dev
```
> Vue dashboard will run at: `http://localhost:5173`

### 4. Configure Database  
Create PostgreSQL database named `race_engineer_db`  
Update connection in `.env`  
```env
DATABASE_URL=postgresql://user:password@localhost:5432/race_engineer_db
```

---

## 🧪 Usage
1. เปิดหน้าเว็บ Dashboard  
2. Upload ไฟล์ telemetry (เช่น `.csv` หรือเชื่อม API จากเกมจำลอง เช่น F1 23, Assetto Corsa)  
3. ระบบจะประมวลผลผ่าน FastAPI และ AI Model  
4. Dashboard แสดงผล real-time (lap time, tyre wear, fuel level)  
5. ระบบเสนอ “Pit Strategy Suggestion” ตามสภาพการแข่งปัจจุบัน  

---
## 🖥️ Demo Website
---

## 📸 Screenshots
<p align="center">
  <img src="assets/dashboard-preview.png" width="800"/>
</p>

---

## 📚 Project Structure
```
ai-race-engineer/
│
├── backend/
│   ├── main.py
│   ├── models/
│   ├── routers/
│   └── ml/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   └── pages/
│   └── public/
│
├── assets/
│   └── cover-hero.png
│
├── docker-compose.yml
└── README.md
```

---

## 📄 License
License © 2025 [Nattakamon Jaimetha](https://github.com/Dxrksvng)

---

## 📬 Contact
- 💼 GitHub: [Dxrksvng](https://github.com/Dxrksvng)  
- 📧 Email: nattakamon0208@gmail.com  
- 🏫 KMITL | Data Science & Business Analytics (Data Engineering Track)

---

## 🧭 Next Milestones
- [ ] Integrate real F1 telemetry API (FastF1 / Ergast API)  
- [ ] Add live comparison mode between two drivers  
- [ ] Deploy to Render + Railway for free hosting  
- [ ] Publish model performance results in KMITL IT Journal  
