# agents/agent.py
from __future__ import annotations
import os
from typing import Optional
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from tools.telemetry_tools import (
    lap_summary, stint_summary, build_delta_vs_time, suggest_pit_lap_simple, evaluate_undercut_simple
)


DEFAULT_LLM = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
KB_DIR = os.getenv("KB_DIR", "kb")

def _build_retriever() -> Optional[object]:
    try:
        loader = DirectoryLoader(KB_DIR, glob="**/*.md", loader_cls=TextLoader, show_progress=False)
        docs = loader.load()
    except Exception:
        docs = []
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    return vectordb.as_retriever(search_kwargs={"k": 3})

def build_agent(session=None):
    llm = ChatOllama(model=DEFAULT_LLM, base_url=OLLAMA_BASE_URL, temperature=0)

    retriever = _build_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=False
    ) if retriever else None

    # --- Tools ---
    def _tool_lap_summary(q: str) -> str:
        drv = q.strip().split()[-1].upper()
        df = lap_summary(session, drv)
        if df.empty:
            return f"No laps for {drv}"
        best = df["LapTime_s"].min()
        avg = df["LapTime_s"].mean()
        return f"Driver {drv}: best={best:.3f}s avg={avg:.3f}s"

    def _tool_stint_summary(q: str) -> str:
        drv = q.strip().split()[-1].upper()
        df = stint_summary(session, drv)
        return df.to_string(index=False) if not df.empty else f"No stints for {drv}"

    def _tool_delta(q: str) -> str:
        txt = q.replace("VS", "vs")
        parts = txt.split("vs")
        if len(parts) != 2:
            return "Format: 'AAA vs BBB'"
        a, b = parts[0].strip().upper(), parts[1].strip().upper()
        df = build_delta_vs_time(session, a, b)
        if df.empty:
            return f"No aligned laps for {a} vs {b}"
        mean_delta = df["Delta_s"].mean()
        return f"Δ(A-B) mean={mean_delta:.3f}s, samples={len(df)}"

    def _tool_plan_pit(q: str) -> str:
        parts = q.strip().split()
        drv = parts[0].upper() if parts else "VER"
        pit_loss = 20.0
        try:
            pit_loss = float(parts[-1])
        except:
            pass
        rec = suggest_pit_lap_simple(session, drv, pit_loss_s=pit_loss)
        return f"Recommend pit on lap≈{rec['recommend_lap']} | {rec['reason']}"

    def _tool_kb(q: str) -> str:
        if not qa_chain:
            return "KB empty. Put .md files in kb/."
        ans = qa_chain.invoke(q)
        return ans["result"] if isinstance(ans, dict) else str(ans)

    tools = [
        Tool("telemetry_query", _tool_lap_summary, "สรุป lap เช่น 'lap summary VER'"),
        Tool("stint_summary", _tool_stint_summary, "สรุป stint เช่น 'stint summary VER'"),
        Tool("delta_compare", _tool_delta, "เปรียบเทียบ เช่น 'VER vs LEC'"),
        Tool("strategy_simulator", _tool_plan_pit, "จำลอง pit เช่น 'VER pit loss 20'"),
        Tool("kb_ask", _tool_kb, "ถามความรู้ทั่วไป เช่น 'When do we use soft tyres?'"),
        Tool(
            name="undercut_evaluator",
            func=lambda q: (lambda a,b,p: str(evaluate_undercut_simple(session,a,b,p)))(
            q.split()[0].upper(), q.split()[2].upper(),
            float([t for t in q.split() if t.replace('.','',1).isdigit()][-1]) if any(ch.isdigit() for ch in q) else 20.0
            ), description="ประเมิน undercut แบบเร็ว: รูปแบบ 'VER vs LEC pit loss 20'")

    ]

    prefix = (
        "You are an AI Race Engineer.\n"
        "RULES:\n"
        "1) For lap/sector/stint/delta/pit questions → use the right tool.\n"
        "2) For theory/undercut/tyres/weather → use kb_ask.\n"
        "3) Never answer directly without calling a tool first.\n"
        "4) Mention which tool you used.\n"
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={"prefix": prefix},
        max_iterations=4,
    )
    return agent
