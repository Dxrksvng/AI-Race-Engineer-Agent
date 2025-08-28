# 🏎️ AI Race Engineer Agent

AI-powered Race Engineer for F1 / SimRacing — built with:
- FastF1 (telemetry, laps, tyres)
- Streamlit + Plotly (dashboard UI)
- LangChain + Ollama (local LLM agents)
- Chroma (RAG knowledge base)

## Quickstart
```bash
git clone https://github.com/<USERNAME>/ai-race-engineer.git
cd ai-race-engineer
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# run
python -m streamlit run ui/app.py
