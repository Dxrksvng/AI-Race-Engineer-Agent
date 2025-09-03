# ui/app.py
import streamlit as st
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

# ---------------- HERO SECTION ----------------
st.markdown(
    """
    <div class="hero">
        <h1>🏎️ AI Race Engineer</h1>
        <p>Dashboards & AI Tools for Formula 1 Strategy Simulation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- ABOUT ----------------
st.markdown('<div class="section"><h2>🌟 เกี่ยวกับโปรเจกต์</h2>', unsafe_allow_html=True)
st.markdown(
    """
    <p>
    โปรเจกต์นี้ถูกสร้างขึ้นเพื่อเลียนแบบการทำงานของ <b>F1 Race Engineer</b>  
    โดยใช้ข้อมูลจริง (Telemetry, Lap Times, Stints) และประมวลผลด้วย AI + Data Tools  
    เพื่อวิเคราะห์ Performance และหา Strategy ที่ดีที่สุด  
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- HOW TO USE ----------------
st.markdown('<div class="section"><h2>⚙️ วิธีใช้งาน</h2>', unsafe_allow_html=True)
st.markdown(
    """
    <p>
    1️⃣ เลือก Session (ปี, สนาม, Q/Race) <br>
    2️⃣ เลือก Driver A และ Driver B (optional) <br>
    3️⃣ ดูกราฟ: Lap Time, Stint Avg, Consistency, Delta Chart <br>
    4️⃣ วิเคราะห์กลยุทธ์ Pit Stop (1-stop vs 2-stop) <br>
    5️⃣ Export Report → PDF/Excel  
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- TOOLS ----------------
st.markdown('<div class="section"><h2>🛠️ Tools ที่ใช้</h2>', unsafe_allow_html=True)
st.markdown(
    """
    <p>
    - <b>FastF1</b> → ดึงและ cache ข้อมูลจาก F1 <br>
    - <b>Pandas/Numpy</b> → จัดการข้อมูลและคำนวณ <br>
    - <b>Matplotlib/Plotly</b> → วาดกราฟ Lap, Stint, Delta <br>
    - <b>Streamlit</b> → พัฒนา Dashboard interactive <br>
    - <b>FPDF2 / XlsxWriter</b> → สร้างรายงาน PDF และ Excel  
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- PURPOSE ----------------
st.markdown('<div class="section"><h2>🎯 ทำไมถึงทำโปรเจกต์นี้</h2>', unsafe_allow_html=True)
st.markdown(
    """
    <p>
    - ฝึกการประยุกต์ AI & Data Engineering กับ Motorsport <br>
    - ทดลองสร้าง Dashboard ที่ใกล้เคียงกับของจริงที่ทีม F1 ใช้ <br>
    - สร้าง Insight ที่ช่วยวิเคราะห์กลยุทธ์การแข่งขัน  
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
    © 2025 AI Race Engineer — Built with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
