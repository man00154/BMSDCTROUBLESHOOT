"""
app.py
Data Center BMS Troubleshooting Chatbot (Streamlit)
Implements:
 - Document ingestion & RAG (Chroma vectorstore)
 - Embedding fallback (OpenAI or sentence-transformers)
 - LLM interface (OpenAI or local HF/transformer model)
 - LangChain agent tools for diagnostics
 - Optional LangGraph orchestration (if installed)
 - Streamlit UI for chat, troubleshooting flows and "run tool" buttons

Run:
  streamlit run app.py

Required env vars:
  OPENAI_API_KEY (optional)
  HUGGINGFACE_API_TOKEN (optional)
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Callable, Optional

import streamlit as st

# NLP & RAG imports
try:
    # LangChain components (recommended)
    from langchain.llms import OpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.chains import ConversationalRetrievalChain
    from langchain.agents import Tool, initialize_agent, AgentType, load_tools
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
except Exception as e:
    st.warning("Some langchain packages missing (will attempt to import on-demand). "
               "Install requirements or review requirements.txt.")
    # We will import lazily later if needed
    OpenAI = None
    OpenAIEmbeddings = None
    Chroma = None
    Document = None
    ConversationalRetrievalChain = None
    Tool = None
    initialize_agent = None
    AgentType = None
    ChatOpenAI = None
    PromptTemplate = None

# Optional LangGraph usage (agent orchestration)
try:
    import langgraph as lg
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# Fallback embedding (sentence-transformers) and vectorstore (FAISS) if OpenAI not present
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# Simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Configuration / Helpers
# -----------------------
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Streamlit sidebar - config
st.set_page_config(page_title="BMS Troubleshooter (RAG + Agents)", layout="wide")

st.sidebar.title("Config")
use_langgraph = st.sidebar.checkbox("Use LangGraph for orchestration (optional)", value=False and LANGGRAPH_AVAILABLE)
st.sidebar.markdown("API keys should be set in environment variables or via `.env` (see `.env.example`).")

# -----------------------
# Embedding factory
# -----------------------
def get_embedding_fn():
    """
    Returns a function that given a list of texts -> list of embeddings (list of float arrays).
    Prefers OpenAI embeddings if OPENAI_API_KEY is present, otherwise falls back to sentence-transformers.
    """
    if os.getenv("OPENAI_API_KEY"):
        # Use LangChain's OpenAIEmbeddings (if available)
        if OpenAIEmbeddings is None:
            raise RuntimeError("OpenAIEmbeddings not available in environment. Install langchain.")
        emb = OpenAIEmbeddings()
        def f(texts: List[str]) -> List[List[float]]:
            return emb.embed_documents(texts)
        return f
    else:
        if not FAISS_AVAILABLE:
            raise RuntimeError("No embedding provider available (set OPENAI_API_KEY or install sentence-transformers & faiss).")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def f(texts: List[str]) -> List[List[float]]:
            arr = model.encode(texts, show_progress_bar=False)
            return arr.tolist()
        return f

# -----------------------
# Vectorstore helpers
# -----------------------
def build_chroma_from_documents(docs: List[Dict[str,Any]], persist_directory=VECTORSTORE_DIR):
    """
    Accepts list of dicts {'page_content': str, 'metadata': {...}}.
    Builds or updates a Chroma vectorstore. Uses OpenAI embeddings if available, else uses local embeddings + FAISS.
    """
    # Prepare text list
    texts = [d["page_content"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]

    if os.getenv("OPENAI_API_KEY") and Chroma is not None:
        # Use Chroma via LangChain + OpenAI embeddings
        emb = OpenAIEmbeddings()
        vs = Chroma.from_texts(texts=texts, embedding=emb, metadatas=metadatas, persist_directory=persist_directory)
        vs.persist()
        return vs
    else:
        # Use local FAISS-backed "Chroma-like" store (lightweight)
        if not FAISS_AVAILABLE:
            raise RuntimeError("No vectorstore available. Install faiss & sentence-transformers.")
        # Create numpy embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(texts)
        d = embs.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.array(embs).astype("float32"))
        # Save index + texts + metadata
        faiss.write_index(index, os.path.join(persist_directory, "faiss.index"))
        with open(os.path.join(persist_directory, "docs.json"), "w", encoding="utf-8") as f:
            json.dump([{"text": t, "metadata": m} for t,m in zip(texts, metadatas)], f, indent=2)
        st.info("Built local FAISS vectorstore (docs.json + faiss.index).")
        # wrap a minimal retriever
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
                    docs.append(Document(page_content=self.texts[idx], metadata=self.metadatas[idx]))
                return docs
        with open(os.path.join(persist_directory, "docs.json"), "r", encoding="utf-8") as f:
            doc_list = json.load(f)
        texts_local = [d["text"] for d in doc_list]
        metadatas_local = [d["metadata"] for d in doc_list]
        return LocalRetriever(index, texts_local, metadatas_local, model)

# -----------------------
# Document ingestion
# -----------------------
def ingest_documents_from_files(file_objs: List[Any]) -> List[Dict[str,Any]]:
    """
    Read uploaded files (txt, json, markdown, csv) and produce documents list.
    """
    docs = []
    for f in file_objs:
        name = getattr(f, "name", "uploaded")
        raw = f.read()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except Exception:
                raw = str(raw)
        # Basic split for large docs
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            docs.append({"page_content": p, "metadata": {"source": name, "chunk": i}})
    st.success(f"Ingested {len(docs)} text chunks from {len(file_objs)} files.")
    return docs

# -----------------------
# Tool implementations (agent tools)
# -----------------------
def tool_check_chiller_status(args: str) -> str:
    """
    Placeholder tool: in a real deployment this would query OPC-UA/BACnet/MQTT to fetch chiller telemetry.
    Here we simulate: args can be 'chiller1' or 'all'.
    """
    logger.info("Running chiller status tool with args: %s", args)
    # Simulated telemetry
    mocked = {
        "chiller1": {"temp": 6.1, "rpm": 1450, "status": "OK"},
        "chiller2": {"temp": 12.4, "rpm": 1300, "status": "ALARM: low flow"},
        "chiller3": {"temp": 7.8, "rpm": 1420, "status": "OK"},
    }
    if args.strip().lower() in ("all", ""):
        return json.dumps(mocked, indent=2)
    else:
        return json.dumps({args: mocked.get(args, "unknown chiller")}, indent=2)

def tool_run_flow_test(args: str) -> str:
    """
    Placeholder: run a simulated flow test sequence (would call PLC/BMS writes normally).
    """
    logger.info("Running flow test tool with args: %s", args)
    time.sleep(1)  # simulate time
    return f"Flow test completed for circuit '{args}'. Measured flow: 64 L/min (simulated)."

def tool_search_docs(query: str, retriever) -> str:
    """
    Tool to search docs and return extracted snippets (for the agent).
    """
    docs = retriever.get_relevant_documents(query)
    out = []
    for d in docs:
        out.append(f"SOURCE: {d.metadata.get('source', '?')} - {d.page_content[:400]}")
    return "\n\n".join(out) if out else "No relevant docs found."

# -----------------------
# Build conversational RAG chain
# -----------------------
def build_rag_chain(retriever, llm):
    """
    Build a conversational retrieval chain that RAGs: retrieves docs -> passes as context to LLM.
    """
    if ConversationalRetrievalChain is None:
        raise RuntimeError("LangChain ConversationalRetrievalChain not available. Install langchain.")
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
    return chain

# -----------------------
# Agent initialization
# -----------------------
def build_agent(llm, retriever):
    """
    Create an agent that can call tools (diagnostics, doc search, run tests).
    We use LangChain agent interface; if LangGraph is enabled, we show how it could orchestrate the tools.
    """
    tools = [
        Tool(
            name="chiller_status",
            func=lambda q: tool_check_chiller_status(q),
            description="Get chiller telemetry and status. Input: chiller ID or 'all'.",
        ),
        Tool(
            name="flow_test",
            func=lambda q: tool_run_flow_test(q),
            description="Run a flow test on a circuit. Input: circuit name.",
        ),
    ]

    # Add a retriever tool that uses the vectorstore
    def doc_search_tool(q: str) -> str:
        try:
            return tool_search_docs(q, retriever)
        except Exception as e:
            return f"doc search error: {e}"

    tools.append(Tool(name="doc_search", func=doc_search_tool, description="Search uploaded BMS docs and manuals."))

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    return agent

# -----------------------
# UI: Streamlit App Layout
# -----------------------
st.title("🔧 MANISH SINGH - Data Center BMS Troubleshooter — RAG + Agents")
st.markdown("Upload BMS manuals, logs, SOPs and ask the agent. Use tool buttons to run diagnostics (simulated).")

# File upload for docs
uploaded_files = st.file_uploader("Upload BMS docs (txt, md, csv, json) — they become the knowledge base (RAG)", accept_multiple_files=True)
if uploaded_files:
    with st.spinner("Ingesting files..."):
        docs = ingest_documents_from_files(uploaded_files)
        vs = build_chroma_from_documents(docs, persist_directory=VECTORSTORE_DIR)
        st.session_state["vectorstore_built"] = True
        st.session_state["vectorstore"] = vs
else:
    if "vectorstore_built" not in st.session_state:
        st.session_state["vectorstore_built"] = False

# LLM selection / initialization
st.sidebar.header("LLM settings")
llm_provider = st.sidebar.selectbox("LLM provider", options=["openai", "local-chat"], index=0)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

def get_llm():
    if llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not found in env — switch to 'local-chat' or set key.")
            raise RuntimeError("No OPENAI_API_KEY")
        # Use ChatOpenAI wrapper if available
        if ChatOpenAI is None:
            raise RuntimeError("ChatOpenAI not available (install langchain).")
        return ChatOpenAI(temperature=temperature, model="gpt-4o-mini")  # adapt to available model
    else:
        # Local placeholder LLM -> echo-style small local model. In production, load a HF pipeline.
        class DummyLLM:
            def __init__(self, temp=0.0):
                self.temperature = temp
            def __call__(self, prompt):
                return "LOCAL_LLM_ECHO: " + (prompt[:1000])
            def generate(self, *args, **kwargs):
                return "LOCAL_LLM_GENERATION"
        return DummyLLM(temperature)

# Build LLM + retriever when asked
if st.button("Initialize Agent & RAG"):
    try:
        llm = get_llm()
        if not st.session_state["vectorstore_built"]:
            st.warning("No vectorstore: upload docs first to make RAG useful. Continuing with empty retriever.")
            retriever = None
        else:
            vs = st.session_state["vectorstore"]
            if hasattr(vs, "as_retriever"):
                retriever = vs.as_retriever(search_kwargs={"k":4})
            else:
                # local wrapper implements get_relevant_documents
                retriever = vs
        # Build RAG chain and agent
        if retriever:
            rag_chain = build_rag_chain(retriever, llm)
            st.session_state["rag_chain"] = rag_chain
        else:
            st.session_state["rag_chain"] = None
        agent = build_agent(llm, retriever)
        st.session_state["agent"] = agent
        st.success("Agent & RAG initialized.")
        if use_langgraph and LANGGRAPH_AVAILABLE:
            st.info("LangGraph is available and may be used for orchestration in advanced flows.")
    except Exception as e:
        st.error(f"Initialization error: {e}")
        logger.exception(e)

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

col1, col2 = st.columns([3,1])
with col1:
    user_input = st.text_area("Ask the BMS agent a question or type a troubleshooting request", height=120)
    if st.button("Send"):
        if not user_input.strip():
            st.warning("Type something first.")
        else:
            st.session_state["chat_history"].append({"role":"user","content":user_input})
            # If rag_chain exists, use it
            if st.session_state.get("rag_chain"):
                chain = st.session_state["rag_chain"]
                with st.spinner("Querying RAG chain..."):
                    result = chain({"question": user_input, "chat_history": []})
                    answer = result["answer"]
                    sources = result.get("source_documents", [])
                st.session_state["chat_history"].append({"role":"assistant","content":answer})
                st.success("Answer generated (RAG).")
                # show sources
                if sources:
                    st.markdown("**Sources used:**")
                    for s in sources:
                        st.write(f"- {s.metadata.get('source', 'unknown')}: {s.page_content[:300]}...")
            else:
                # Fall back to agent-only
                agent = st.session_state.get("agent")
                if not agent:
                    st.warning("Agent not initialized. Click **Initialize Agent & RAG**.")
                else:
                    with st.spinner("Agent thinking..."):
                        res = agent.run(user_input)
                        st.session_state["chat_history"].append({"role":"assistant","content":res})

with col2:
    st.markdown("### Tools (simulate)")
    if st.button("Get all chiller status"):
        agent = st.session_state.get("agent")
        if agent:
            out = tool_check_chiller_status("all")
            st.code(out)
            st.session_state["chat_history"].append({"role":"assistant","content":out})
        else:
            st.warning("Initialize agent first.")
    if st.button("Run flow test (simulated)"):
        out = tool_run_flow_test("main-circuit")
        st.code(out)
        st.session_state["chat_history"].append({"role":"assistant","content":out})

# Chat history display
st.markdown("---")
st.header("Chat history")
for msg in st.session_state["chat_history"]:
    if msg["role"]=="user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Agent:** {msg['content']}")

# -----------------------
# Optional LangGraph example graph creation (demonstrative only)
# -----------------------
if use_langgraph:
    if not LANGGRAPH_AVAILABLE:
        st.warning("LangGraph not installed in this environment. See README to install.")
    else:
        st.markdown("## LangGraph orchestration (example)")
        st.markdown("This example builds a simple LangGraph workflow that: 1) retrieves docs, 2) runs an agent tool, 3) aggregates results.")
        if st.button("Create example LangGraph workflow"):
            # Pseudocode using LangGraph; real code depends on installed LangGraph API
            try:
                graph = lg.Graph()
                # State stores query and results
                s = lg.State({"query": "check chiller2 flow", "doc_results": None, "tool_output": None})
                graph.add_state("s", s)
                # Node: doc retrieval
                def node_retrieve(state):
                    q = state["query"]
                    if st.session_state.get("vectorstore_built"):
                        retr = st.session_state["vectorstore"]
                        docs = retr.get_relevant_documents(q, k=3)
                        state["doc_results"] = [d.page_content for d in docs]
                    else:
                        state["doc_results"] = []
                graph.add_node("retrieve", node_retrieve, inputs=["s"], outputs=["s"])
                # Node: run tool
                def node_tool(state):
                    state["tool_output"] = tool_check_chiller_status("chiller2")
                graph.add_node("diagnose", node_tool, inputs=["s"], outputs=["s"])
                # Execute graph (synchronously)
                graph.run("retrieve")
                graph.run("diagnose")
                st.success("LangGraph example run complete. You can inspect state in logs.")
            except Exception as e:
                st.error(f"LangGraph run error: {e}")

# Footer / credits
st.markdown("---")
st.caption("This app demonstrates a RAG + agentic architecture. Replace simulated tools with real BMS integrations (OPC-UA, BACnet, MQTT) for production.")

