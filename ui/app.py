# ui/app.py
import sys, os
import streamlit as st

# บังคับให้มองเห็น root ของโปรเจกต์
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from race_logger import log_event, read_events


st.set_page_config(page_title="AI Race Engineer", layout="wide")

# ---------------- CSS ----------------
st.markdown(
    """
    <style>
    body {
        background-color: #0d0d0d;
        color: white;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .hero {
        text-align: center;
        padding: 80px 20px;
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                    url('https://media.contentapi.ea.com/content/dam/ea/f1/f1-23/common/articles/patch-note-v109/pj-f123-bel-w01-rus-unmarked.jpg.adapt.crop191x100.628p.jpg') no-repeat center;
        background-size: cover;
        border-radius: 16px;
        margin-bottom: 40px;
    }
    .hero h1 {
        font-size: 64px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #e63946;
    }
    .hero p {
        font-size: 22px;
        color: #f1f1f1;
    }
    .section {
        margin: 60px 0;
    }
    .section h2 {
        font-size: 32px;
        font-weight: bold;
        color: #e63946;
        margin-bottom: 20px;
    }
    .section p {
        font-size: 18px;
        line-height: 1.6;
        color: #ddd;
    }
    .footer {
        margin-top: 60px;
        padding: 20px;
        text-align: center;
        font-size: 14px;
        color: #aaa;
        border-top: 1px solid #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("📌 Project Overview")
st.write("""
**AI Race Engineer Agent** คือระบบต้นแบบ (prototype) ที่จำลองการทำงานของ **Race Engineer** 
ซึ่งเป็นบทบาทสำคัญในการแข่งขัน Motorsport (เช่น F1, Endurance Racing)  
โปรเจกต์นี้ถูกพัฒนาเพื่อแสดงทักษะด้าน **Data Engineering, Data Analytics และ AI for Decision Support**  
โดยใช้เทคโนโลยีและเครื่องมือที่เป็นมาตรฐานในอุตสาหกรรม
""")

st.header("🎯 Objective")
st.write("""
- สร้างระบบ **logging** สำหรับเก็บข้อมูลการแข่งขัน (Lap Time, Driver, Stint ฯลฯ)  
- สร้าง **Dashboard / Visualization** เพื่อวิเคราะห์ข้อมูล lap ของรถและนักขับ  
- จำลองการใช้งานจริงในการตัดสินใจ เช่น การเปรียบเทียบเวลา lap, วิเคราะห์ performance, และการวางแผน pit stop  
- ใช้เป็น **Portfolio Project** เพื่อแสดงความสามารถด้าน Data/AI + Motorsport Insight
""")

st.header("🛠️ Tools & Technologies")
st.markdown("""
- **Python** → ภาษาหลักในการพัฒนา  
- **Streamlit** → สร้าง Web UI ที่ใช้งานง่าย  
- **Pandas** → จัดการข้อมูลตารางและคำนวณสถิติ lap  
- **Plotly / Streamlit Charts** → สร้าง visualization เช่น line chart, bar chart  
- **race_logger (custom)** → โมดูลสำหรับบันทึกและอ่าน event (Lap Completed, Stint Start/End)  
- **JSON** → เก็บ log event แบบ structured data  
- **Excel/CSV Export (xlsxwriter)** → รองรับการ export ข้อมูลไปใช้ต่อ
""")

st.header("🚦 Features")
st.markdown("""
### 1. **Lap Logging**
- ระบบสามารถบันทึกข้อมูล lap ของนักขับ เช่น:
  - `Lap Number` → หมายเลขรอบ
  - `Lap Time` → เวลาในแต่ละรอบ (วัดประสิทธิภาพการขับ)
  - `Driver` → นักขับที่กำลังอยู่ในรถ
  - `Timestamp` → เวลาที่เกิดเหตุการณ์จริง
- ใช้ผ่าน `log_event("lap_completed", {...})`

### 2. **Lap Viewer (Page 1)**
- ดึงข้อมูล lap จาก log มาแสดงใน **ตาราง (Data Table)**  
- แสดงผลลัพธ์ lap time ด้วย **กราฟเส้น (Line Chart)**  
- ทำให้สามารถ track performance ต่อรอบได้อย่างง่ายดาย  

### 3. **Dashboard (Page 2)**
- แสดงผลสถิติภาพรวม:
  - Fastest Lap → รอบที่ใช้เวลาน้อยที่สุด
  - Average Lap Time → เวลาต่อรอบเฉลี่ย
  - Stint Analysis → การแบ่ง session ตามนักขับหรือเชื้อเพลิง
  - จำนวนรอบทั้งหมด
- Visualization แบบ bar chart, summary card  

### 4. **Event Logging**
- รองรับ event หลายประเภท เช่น:
  - `lap_completed`
  - `stint_started` / `stint_ended`
  - `pit_stop`
- ข้อมูลทั้งหมดเก็บไว้ในไฟล์ JSON

### 5. **Data Export**
- ผู้ใช้สามารถ export log ออกเป็น CSV / Excel ได้ เพื่อวิเคราะห์ต่อใน Excel, Power BI, หรือ ML pipeline
""")

st.header("📖 Motorsport Concepts (เพื่อให้ HR เข้าใจ)")
st.markdown("""
- **Lap Time** → เวลาที่นักขับใช้วิ่งครบหนึ่งรอบสนาม (เป็นตัวชี้วัดหลักของ performance)  
- **Stint** → ช่วงเวลาที่นักขับหนึ่งคนอยู่ในรถต่อเนื่อง (เช่น 20 lap ติดก่อนเปลี่ยนคนหรือเติมน้ำมัน)  
- **Pit Stop** → การหยุดเข้าพิทเพื่อเปลี่ยนยาง เติมน้ำมัน หรือสลับนักขับ  
- **Race Engineer** → บุคคลที่ดูแลข้อมูลการแข่งขันและสื่อสารกับนักขับเพื่อช่วยตัดสินใจในระหว่างแข่ง
""")

st.header("📊 Workflow / Steps")
st.markdown("""
1. **Log Data** → ระบบรับข้อมูล lap/event ผ่าน `log_event`  
2. **Store Data** → เก็บใน `race_log.json`  
3. **Read & Process** → ใช้ `read_events()` แปลงเป็น structured data  
4. **Visualize** → แสดงผลใน Streamlit (Lap Viewer, Dashboard)  
5. **Analyze** → ผู้ใช้สามารถสรุปผล performance และใช้เป็นข้อมูลในการตัดสินใจ
""")

st.header("💡 Value of the Project")
st.write("""
- แสดงให้เห็นความเข้าใจทั้งด้าน **Data Engineering** (logging, ETL pipeline เบื้องต้น)  
และ **Data Analytics** (summary stats, visualization)  
- ใช้บริบท Motorsport ที่เป็น **real-world high-pressure environment**  
เพื่อสื่อว่าผู้พัฒนาสามารถทำระบบวิเคราะห์ข้อมูลที่มีผลต่อการตัดสินใจได้จริง  
- สามารถต่อยอดเป็น **AI Decision Support System** ในอนาคต (เช่น แนะนำกลยุทธ์ pit stop อัตโนมัติ)  
- เป็นโปรเจกต์ Portfolio ที่เหมาะสำหรับตำแหน่งด้าน **Data Engineer, AI Engineer, Data Analyst**  
""")

st.header("🤖 AI Race Engineer Chat")
st.markdown("""
อีกหนึ่งหัวใจสำคัญของโปรเจกต์นี้คือ **AI Chat Interface**  
ผู้ใช้สามารถสนทนากับ **AI Race Engineer** ได้โดยตรงภายในแอป เช่น:
- ถามว่าใครทำ Fastest Lap
- ขอเปรียบเทียบ lap time ของนักขับ 2 คน
- ขอวิเคราะห์ performance ของ stint ล่าสุด
- ขอคำแนะนำกลยุทธ์ (เช่นควรเข้าพิทตอนไหน)

### วิธีทำงาน
- **LLM (Local AI Model)**: ใช้ LLM (เช่น Ollama / Llama 3) เพื่อประมวลผลภาษาธรรมชาติ  
- **LangChain**: ใช้เป็น Orchestration layer สำหรับเชื่อม AI กับระบบ log  
- **Integration**: AI สามารถเรียก `read_events()` เพื่อดึงข้อมูลจริงจาก log  
- **Streamlit Chat UI**: มีหน้า Chat ให้พิมพ์คำถาม-คำตอบกับ AI แบบ realtime  

### Value
- แสดงให้เห็นทักษะ **AI Integration** (เชื่อม LLM + Data Backend)  
- ทำให้โปรเจกต์ไม่ใช่แค่ Dashboard สถิติ แต่กลายเป็น **AI Decision Support System**  
- จำลองบทบาท Race Engineer ที่คุยกับทีมผ่านวิทยุ ให้ HR เข้าใจได้ชัดเจนว่า  
  ผู้สมัครสามารถใช้ AI ทำงานใน **real-world scenario** ได้จริง
""")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
    © 2025 AI Race Engineer — Built with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
