"""
app.py
BMS Troubleshooting Chatbot (RAG + Agents + Local LLM)
------------------------------------------------------
Features:
 - Document ingestion & RAG (Chroma or FAISS)
 - Local LLM fallback (no API key needed)
 - LangChain Tools & Agents for diagnostics
 - Compatible with LangChain v0.2 and v0.3+
Run:
  streamlit run app.py
"""

import os
import time
import json
import logging
from typing import List, Any
import streamlit as st

# ----------------------- LangChain Imports -----------------------
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.chains import ConversationalRetrievalChain
    from langchain.agents import initialize_agent, AgentType
except Exception:
    OpenAIEmbeddings = None
    Chroma = None
    Document = None
    ConversationalRetrievalChain = None
    initialize_agent = None
    AgentType = None

# ✅ Tool Import Compatibility Fix (new vs old LangChain)
try:
    from langchain.tools import Tool
    def make_tool(func, name, description):
        return Tool.from_function(func=func, name=name, description=description)
except ImportError:
    from langchain.agents import Tool
    def make_tool(func, name, description):
        return Tool(name=name, func=func, description=description)

# ----------------------- Optional LangGraph -----------------------
try:
    import langgraph as lg
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# ----------------------- Fallback Embeddings -----------------------
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# ----------------------- Logging -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------- Config -----------------------
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

st.set_page_config(page_title="BMS Troubleshooter", layout="wide")
st.sidebar.title("Configuration")

use_langgraph = st.sidebar.checkbox("Use LangGraph (optional)", value=False and LANGGRAPH_AVAILABLE)
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.05)

# ----------------------- Embeddings -----------------------
def get_embedding_fn():
    """Select embedding provider (OpenAI or local SentenceTransformer)."""
    if os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings:
        emb = OpenAIEmbeddings()
        def f(texts: List[str]) -> List[List[float]]:
            return emb.embed_documents(texts)
        return f
    else:
        if not FAISS_AVAILABLE:
            raise RuntimeError("Install sentence-transformers & faiss for local embedding.")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def f(texts: List[str]) -> List[List[float]]:
            return model.encode(texts, show_progress_bar=False).tolist()
        return f

# ----------------------- Vectorstore -----------------------
def build_vectorstore(docs: List[dict]):
    """Build Chroma or FAISS vectorstore from docs."""
    texts = [d["page_content"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]

    if os.getenv("OPENAI_API_KEY") and Chroma:
        emb = OpenAIEmbeddings()
        vs = Chroma.from_texts(texts=texts, embedding=emb,
                               metadatas=metadatas, persist_directory=VECTORSTORE_DIR)
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

# ----------------------- Document Ingestion -----------------------
def ingest_documents(file_objs: List[Any]):
    """Split uploaded files into small document chunks."""
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

# ----------------------- Tools -----------------------
def tool_check_chiller_status(args: str) -> str:
    """Simulate chiller status fetch."""
    mocked = {
        "chiller1": {"temp": 6.1, "rpm": 1450, "status": "OK"},
        "chiller2": {"temp": 12.4, "rpm": 1300, "status": "ALARM: low flow"},
        "chiller3": {"temp": 7.8, "rpm": 1420, "status": "OK"},
    }
    if args.strip().lower() in ("all", ""):
        return json.dumps(mocked, indent=2)
    return json.dumps({args: mocked.get(args, "unknown chiller")}, indent=2)

def tool_run_flow_test(args: str) -> str:
    """Simulate flow test execution."""
    time.sleep(1)
    return f"Flow test completed for circuit '{args}'. Measured flow: 64 L/min (simulated)."

def tool_search_docs(query: str, retriever) -> str:
    """Search docs (for the agent)."""
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant docs found."
    out = [f"SOURCE: {d.metadata.get('source','?')} — {d.page_content[:300]}" for d in docs]
    return "\n\n".join(out)

# ----------------------- Dummy Local LLM -----------------------
class LocalLLM:
    def __init__(self, temperature=0.0): self.temperature = temperature
    def __call__(self, prompt): return "LOCAL_LLM_ECHO: " + prompt[:1000]

def get_llm():
    """Return a dummy LLM if OpenAI not configured."""
    return LocalLLM(temperature)

# ----------------------- RAG & Agent -----------------------
def build_rag_chain(retriever, llm):
    if ConversationalRetrievalChain is None:
        raise RuntimeError("LangChain ConversationalRetrievalChain not available.")
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True
    )

def build_agent(llm, retriever):
    """Create tools and agent safely (cross-version compatible)."""
    tools = [
        make_tool(lambda q: tool_check_chiller_status(q),
                  "chiller_status", "Get chiller telemetry (input: chiller ID or 'all')."),
        make_tool(lambda q: tool_run_flow_test(q),
                  "flow_test", "Run simulated flow test (input: circuit name)."),
        make_tool(lambda q: tool_search_docs(q, retriever),
                  "doc_search", "Search uploaded BMS documents.")
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    return agent

# ----------------------- Streamlit UI -----------------------
st.title("🏭 MANISH SINGH — Data Centre BMS Troubleshooter Chatbot")
st.markdown("Upload **BMS logs/manuals** and ask the chatbot for diagnostics, flow test simulation, or chiller analysis.")

uploaded_files = st.file_uploader("Upload BMS logs/manuals (TXT only)", type=["txt"], accept_multiple_files=True)
if uploaded_files:
    docs = ingest_documents(uploaded_files)
    retriever = build_vectorstore(docs)
else:
    retriever = None

llm = get_llm()

if retriever:
    agent = build_agent(llm, retriever)

st.subheader("💬 Ask your question or run tools")
user_input = st.text_input("Example: 'Why is chiller2 in alarm low flow?'")

if st.button("Submit") and user_input:
    if retriever:
        with st.spinner("Running Agent..."):
            response = agent.run(user_input)
        st.text_area("Response", value=response, height=300)
    else:
        st.warning("Upload documents first to enable RAG/Agent features.")
