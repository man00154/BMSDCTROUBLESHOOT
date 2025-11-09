"""
app.py
----------------------------------
Data Centre BMS Troubleshooting Chatbot
Implements:
 - RAG (Retrieval-Augmented Generation)
 - LLM Agent tools for diagnostics
 - Local free LLM fallback (no OpenAI key needed)
 - Compatible with LangChain >= 0.3
----------------------------------
Run:
    streamlit run app.py
"""

import os
import time
import json
import logging
from typing import List, Any
import streamlit as st

# -------------------------------------------------------------
# LangChain Imports (Modern + Backward Compatibility)
# -------------------------------------------------------------
try:
    # New LangChain (>=0.3) structure
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.chains import ConversationalRetrievalChain
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
except ImportError:
    # Fallback for older LangChain (<0.3)
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.chains import ConversationalRetrievalChain
    from langchain.agents import initialize_agent, AgentType
    from langchain.agents import Tool

# -------------------------------------------------------------
# Optional LangGraph (Orchestration Layer)
# -------------------------------------------------------------
try:
    import langgraph as lg
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# -------------------------------------------------------------
# Fallback for Local Embeddings (no OpenAI key)
# -------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# -------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Directories
# -------------------------------------------------------------
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# -------------------------------------------------------------
# Streamlit UI Setup
# -------------------------------------------------------------
st.set_page_config(page_title="BMS Troubleshooter", layout="wide")
st.sidebar.title("Configuration")
use_langgraph = st.sidebar.checkbox("Enable LangGraph", value=False and LANGGRAPH_AVAILABLE)
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.05)

# -------------------------------------------------------------
# Embedding Factory
# -------------------------------------------------------------
def get_embedding_fn():
    """Select embedding provider (OpenAI or local)."""
    if os.getenv("OPENAI_API_KEY"):
        emb = OpenAIEmbeddings()
        def f(texts: List[str]) -> List[List[float]]:
            return emb.embed_documents(texts)
        return f
    else:
        if not FAISS_AVAILABLE:
            raise RuntimeError("Install sentence-transformers & faiss for local embeddings.")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def f(texts: List[str]) -> List[List[float]]:
            return model.encode(texts, show_progress_bar=False).tolist()
        return f

# -------------------------------------------------------------
# Vectorstore Builder
# -------------------------------------------------------------
def build_vectorstore(docs: List[dict]):
    """Create Chroma (OpenAI) or FAISS (local) vectorstore."""
    texts = [d["page_content"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]

    if os.getenv("OPENAI_API_KEY"):
        emb = OpenAIEmbeddings()
        vs = Chroma.from_texts(
            texts=texts, embedding=emb, metadatas=metadatas,
            persist_directory=VECTORSTORE_DIR
        )
        vs.persist()
        return vs
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(texts)
        d = embs.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.array(embs).astype("float32"))
        faiss.write_index(index, os.path.join(VECTORSTORE_DIR, "faiss.index"))
        with open(os.path.join(VECTORSTORE_DIR, "docs.json"), "w") as f:
            json.dump([{"text": t, "metadata": m} for t, m in zip(texts, metadatas)], f)

        class LocalRetriever:
            def __init__(self, index, texts, metadatas, model):
                self.index = index
                self.texts = texts
                self.metadatas = metadatas
                self.model = model
            def get_relevant_documents(self, query, k=4):
                q_emb = self.model.encode([query]).astype("float32")
                D, I = self.index.search(q_emb, k)
                docs = []
                for idx in I[0]:
                    docs.append(Document(page_content=self.texts[idx],
                                         metadata=self.metadatas[idx]))
                return docs

        with open(os.path.join(VECTORSTORE_DIR, "docs.json")) as f:
            doc_list = json.load(f)
        texts_local = [d["text"] for d in doc_list]
        metadatas_local = [d["metadata"] for d in doc_list]
        return LocalRetriever(index, texts_local, metadatas_local, model)

# -------------------------------------------------------------
# Document Ingestion
# -------------------------------------------------------------
def ingest_documents(file_objs: List[Any]):
    """Convert uploaded files to text chunks."""
    docs = []
    for f in file_objs:
        name = getattr(f, "name", "uploaded")
        raw = f.read()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except:
                raw = str(raw)
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            docs.append({"page_content": p,
                         "metadata": {"source": name, "chunk": i}})
    st.success(f"Ingested {len(docs)} chunks from {len(file_objs)} files.")
    return docs

# -------------------------------------------------------------
# Tools (Simulated Hardware)
# -------------------------------------------------------------
def tool_check_chiller_status(args: str) -> str:
    mocked = {
        "chiller1": {"temp": 6.1, "rpm": 1450, "status": "OK"},
        "chiller2": {"temp": 12.4, "rpm": 1300, "status": "ALARM: low flow"},
        "chiller3": {"temp": 7.8, "rpm": 1420, "status": "OK"},
    }
    if args.strip().lower() in ("all", ""):
        return json.dumps(mocked, indent=2)
    return json.dumps({args: mocked.get(args, "unknown chiller")}, indent=2)

def tool_run_flow_test(args: str) -> str:
    time.sleep(1)
    return f"Flow test completed for circuit '{args}'. Measured flow: 64 L/min (simulated)."

def tool_search_docs(query: str, retriever) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        [f"SOURCE: {d.metadata.get('source','?')} — {d.page_content[:300]}" for d in docs]
    )

# -------------------------------------------------------------
# Local Dummy LLM (Free Mode)
# -------------------------------------------------------------
class LocalLLM:
    def __init__(self, temperature=0.0): self.temperature = temperature
    def __call__(self, prompt): return "LOCAL_LLM_RESPONSE: " + prompt[:1000]

def get_llm():
    return LocalLLM(temperature)

# -------------------------------------------------------------
# Agent Builder
# -------------------------------------------------------------
def build_agent(llm, retriever):
    """Create agent tools and initialize LangChain Agent."""
    tools = [
        Tool.from_function(
            func=lambda q: tool_check_chiller_status(q),
            name="chiller_status",
            description="Get current chiller telemetry (input: ID or 'all')."
        ),
        Tool.from_function(
            func=lambda q: tool_run_flow_test(q),
            name="flow_test",
            description="Run simulated flow test (input: circuit name)."
        ),
        Tool.from_function(
            func=lambda q: tool_search_docs(q, retriever),
            name="doc_search",
            description="Search uploaded BMS documents."
        ),
    ]
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.title("🏭 MANISH SINGH — Data Centre BMS Troubleshooter Chatbot")
st.markdown("Upload **BMS manuals or logs** and ask troubleshooting queries using RAG + AI Agents.")

uploaded_files = st.file_uploader("📁 Upload BMS Manuals or Logs (TXT)", type=["txt"], accept_multiple_files=True)
if uploaded_files:
    docs = ingest_documents(uploaded_files)
    retriever = build_vectorstore(docs)
else:
    retriever = None

llm = get_llm()
if retriever:
    agent = build_agent(llm, retriever)

st.subheader("💬 Ask your BMS troubleshooting question")
user_input = st.text_input("Example: 'Check status of all chillers' or 'Search for low flow alarm procedure'")

if st.button("Submit") and user_input:
    if retriever:
        with st.spinner("Agent reasoning..."):
            response = agent.run(user_input)
        st.text_area("Response", value=response, height=300)
    else:
        st.warning("Please upload at least one document to initialize RAG.")
