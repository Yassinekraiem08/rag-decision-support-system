"""
RAG Decision Support System — Interactive Demo

Usage:
    streamlit run demo/chat_demo.py
    (requires FastAPI backend: uvicorn app.main:app --reload)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
import time
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Decision Support System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }

    .chat-msg-user {
        background: #1e3a5f;
        border-left: 3px solid #4a9eff;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e8f0fe;
    }
    .chat-msg-assistant {
        background: #1a1f2e;
        border-left: 3px solid #00c853;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e8f0fe;
    }
    .chat-msg-refused {
        background: #2a1a1a;
        border-left: 3px solid #ff5252;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #ffcdd2;
    }
    .confidence-high   { background:#1b5e20; color:#a5d6a7; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:600; }
    .confidence-medium { background:#e65100; color:#ffe0b2; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:600; }
    .confidence-low    { background:#b71c1c; color:#ffcdd2; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:600; }
    .verdict-supported   { background:#1b5e20; color:#a5d6a7; padding:2px 8px; border-radius:8px; font-size:12px; }
    .verdict-partial     { background:#e65100; color:#ffe0b2; padding:2px 8px; border-radius:8px; font-size:12px; }
    .verdict-unsupported { background:#b71c1c; color:#ffcdd2; padding:2px 8px; border-radius:8px; font-size:12px; }
    .source-chip { background:#1a2744; border:1px solid #2a3f6f; color:#90caf9; padding:2px 10px; border-radius:12px; font-size:12px; margin:2px; display:inline-block; }
    .metric-card { background:#1a1f2e; border:1px solid #2a3044; border-radius:10px; padding:16px; text-align:center; }
    .metric-num  { font-size:28px; font-weight:700; color:#4a9eff; }
    .metric-label { font-size:12px; color:#8892b0; margin-top:4px; }
    .refused-badge { background:#b71c1c; color:#ffcdd2; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .example-btn { cursor:pointer; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def query_api(question: str) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the FastAPI server is running: uvicorn app.main:app --reload"}
    except Exception as e:
        return {"error": str(e)}


def confidence_badge(score: float) -> str:
    if score >= 0.7:
        return f'<span class="confidence-high">Confidence: {score:.0%}</span>'
    elif score >= 0.4:
        return f'<span class="confidence-medium">Confidence: {score:.0%}</span>'
    else:
        return f'<span class="confidence-low">Confidence: {score:.0%}</span>'


def verdict_badge(verdict: str) -> str:
    v = verdict.upper()
    if v == "SUPPORTED":
        return f'<span class="verdict-supported">SUPPORTED</span>'
    elif v == "PARTIALLY_SUPPORTED":
        return f'<span class="verdict-partial">PARTIAL</span>'
    else:
        return f'<span class="verdict-unsupported">UNSUPPORTED</span>'


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 AI Decision Support")
    st.markdown("*Production RAG with evaluation*")
    st.divider()

    # API status
    api_ok = check_api_health()
    if api_ok:
        st.success("API Connected", icon="✅")
    else:
        st.error("API Offline", icon="🔴")
        st.caption("Run: `uvicorn app.main:app --reload`")

    st.divider()
    st.markdown("### Try these questions")

    example_questions = [
        "What is embodied AI?",
        "How does embodied AI affect manufacturing supply chains?",
        "What role do tactile sensors play in robot dexterity?",
        "Compare approaches to human-robot collaboration across the papers.",
        "What do all papers say about generalization in AI systems?",
        "How does embodied AI differ from screen-based AI in education?",
    ]

    example_unanswerable = [
        "What is the current price of NVIDIA stock?",
        "What is two plus two?",
        "Who won the 2025 Nobel Prize in Physics?",
    ]

    st.markdown("**On-topic (will answer):**")
    for i, q in enumerate(example_questions):
        if st.button(q[:50] + ("..." if len(q) > 50 else ""), key=f"ex_{i}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("**Off-topic (will refuse):**")
    for i, q in enumerate(example_unanswerable):
        if st.button(q, key=f"un_{i}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.divider()
    st.markdown("### System Stats")
    st.markdown("""
    - **18 documents** indexed
    - **1,673 chunks** in pgvector
    - **Hybrid retrieval** + reranking
    - **Confidence threshold**: 0.43
    - **Parallel reranking**: -76% latency
    """)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.stats = {"total": 0, "answered": 0, "refused": 0, "latency_sum": 0}
        st.rerun()


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "answered": 0, "refused": 0, "latency_sum": 0}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 AI Decision Support System")
st.markdown("*Production RAG — citation-grounded answers with confidence scoring and hallucination prevention*")

# Stats row
s = st.session_state.stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-num">{s["total"]}</div><div class="metric-label">Queries asked</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-num">{s["answered"]}</div><div class="metric-label">Answered</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-num">{s["refused"]}</div><div class="metric-label">Refused (out-of-corpus)</div></div>', unsafe_allow_html=True)
with col4:
    avg_latency = s["latency_sum"] / s["total"] if s["total"] > 0 else 0
    st.markdown(f'<div class="metric-card"><div class="metric-num">{avg_latency:.1f}s</div><div class="metric-label">Avg latency</div></div>', unsafe_allow_html=True)

st.divider()


# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-msg-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        data = msg["data"]
        refused = data.get("confidence", 1.0) == 0.0

        if refused:
            st.markdown(f'''
            <div class="chat-msg-refused">
                <span class="refused-badge">REFUSED — Out of corpus</span><br><br>
                🤖 {data["answer"]}
            </div>''', unsafe_allow_html=True)
        else:
            verdict = data.get("verification", {}).get("verdict", "UNSUPPORTED")
            conf = data.get("confidence", 0.0)
            sources = [c["filename"] for c in data.get("retrieved_chunks", [])]
            unique_sources = list(dict.fromkeys(sources))
            latency = msg.get("latency", 0)

            source_chips = " ".join([f'<span class="source-chip">📄 {s}</span>' for s in unique_sources])

            st.markdown(f'''
            <div class="chat-msg-assistant">
                <div style="margin-bottom:8px;">
                    {confidence_badge(conf)} &nbsp; {verdict_badge(verdict)} &nbsp;
                    <span style="color:#8892b0; font-size:12px;">{latency:.1f}s</span>
                </div>
                🤖 {data["answer"]}
                <div style="margin-top:10px; color:#8892b0; font-size:12px;">Sources: {source_chips}</div>
            </div>''', unsafe_allow_html=True)

            # Show sources expandable
            if data.get("retrieved_chunks"):
                with st.expander(f"View {len(data['retrieved_chunks'])} retrieved chunks"):
                    for i, chunk in enumerate(data["retrieved_chunks"], 1):
                        st.markdown(f"**[{i}] {chunk['filename']}** — score: `{chunk['score']:.3f}`")
                        st.caption(chunk["content"][:300] + "...")
                        if i < len(data["retrieved_chunks"]):
                            st.divider()


# ── Input ──────────────────────────────────────────────────────────────────────
# Handle sidebar button clicks
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    question = None

user_input = st.chat_input("Ask anything about the document corpus...")
if user_input:
    question = user_input

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Retrieving and generating..."):
        t0 = time.time()
        result = query_api(question)
        latency = time.time() - t0

    if result and "error" not in result:
        st.session_state.messages.append({
            "role": "assistant",
            "data": result,
            "latency": latency
        })

        # Update stats
        st.session_state.stats["total"] += 1
        st.session_state.stats["latency_sum"] += latency
        if result.get("confidence", 0) == 0.0:
            st.session_state.stats["refused"] += 1
        else:
            st.session_state.stats["answered"] += 1
    else:
        error_msg = result.get("error", "Unknown error") if result else "No response"
        st.error(f"Error: {error_msg}")

    st.rerun()


# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#8892b0;">
        <h2 style="color:#4a9eff;">Ask a question to get started</h2>
        <p>Try one of the example questions in the sidebar, or type your own.</p>
        <p style="font-size:13px; margin-top:20px;">
            The system will answer questions grounded in 18 technical AI/robotics papers,<br>
            and refuse questions outside the corpus scope.
        </p>
    </div>
    """, unsafe_allow_html=True)
