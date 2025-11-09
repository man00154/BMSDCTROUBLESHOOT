"""
app.py
Data Centre BMS Troubleshooter (minimal, LangChain-free)
 - Uses sentence-transformers + FAISS for RAG-style retrieval
 - Simple tool runner (chiller_status, flow_test, doc_search)
 - Local LLM fallback (no OpenAI key required)
Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import time
import json
import logging
from typing import List, Any, Dict

import streamlit as st

# Embedding / vector store libs (local)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    ST_AVAILABLE = True
except Exception as e:
    ST_AVAILABLE = False
    # We'll show a helpful message later if missing

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Streamlit config
st.set_page_config(page_title="BMS Troubleshooter (minimal)", layout="wide")
st.title("🏭 BMS Troubleshooter — MANISH SINGH")

st.sidebar.header("Config")
st.sidebar.write("This minimal app uses local embeddings (sentence-transformers) and FAISS. No OpenAI or LangChain imports required.")
temperature = st.sidebar.slider("Temperature (affects dummy LLM randomness)", 0.0, 1.0, 0.0, 0.05)

# ---------------------------
# Utilities: local retriever
# ---------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = None

def ensure_embedding_model():
    if not ST_AVAILABLE:
        st.error("Missing dependencies: please install sentence-transformers and faiss (see requirements).")
        raise RuntimeError("Missing sentence-transformers / faiss")
    global embed_model, EMBED_DIM
    if "embed_model" not in globals():
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        EMBED_DIM = embed_model.get_sentence_embedding_dimension()
    return embed_model

class SimpleFAISSIndex:
    """
    Simple FAISS backed index that stores texts+metadata in docs.json and faiss.index.
    Methods:
      - build(texts, metadatas)
      - load()
      - search(query, k)
    """
    def __init__(self, index_path=os.path.join(VECTORSTORE_DIR, "faiss.index"),
                 docs_path=os.path.join(VECTORSTORE_DIR, "docs.json")):
        self.index_path = index_path
        self.docs_path = docs_path
        self.index = None
        self.texts = []
        self.metadatas = []
        self.model = None

    def build(self, texts: List[str], metadatas: List[Dict[str,Any]]):
        model = ensure_embedding_model()
        embs = model.encode(texts)
        d = embs.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.array(embs).astype("float32"))
        faiss.write_index(index, self.index_path)
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump([{"text": t, "metadata": m} for t,m in zip(texts, metadatas)], f, indent=2)
        # in-memory
        self.index = index
        self.texts = texts
        self.metadatas = metadatas
        self.model = model
        st.success("Built local FAISS index.")

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docs_path):
            return False
        idx = faiss.read_index(self.index_path)
        with open(self.docs_path, "r", encoding="utf-8") as f:
            doc_list = json.load(f)
        self.index = idx
        self.texts = [d["text"] for d in doc_list]
        self.metadatas = [d["metadata"] for d in doc_list]
        self.model = ensure_embedding_model()
        return True

    def search(self, query: str, k: int = 4):
        if self.index is None:
            raise RuntimeError("Index not built/loaded.")
        q_emb = self.model.encode([query]).astype("float32")
        D, I = self.index.search(q_emb, k)
        results = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "distance": float(D[0][list(I[0]).index(idx)]) if D.size else None
            })
        return results

# instantiate index instance
faiss_index = SimpleFAISSIndex()

# ---------------------------
# Document ingestion helpers
# ---------------------------
def ingest_documents(file_objs: List[Any]):
    docs = []
    for f in file_objs:
        name = getattr(f, "name", "uploaded")
        raw = f.read()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except:
                raw = str(raw)
        # split by blank lines to create chunks
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            docs.append({"page_content": p, "metadata": {"source": name, "chunk": i}})
    st.success(f"Ingested {len(docs)} chunks from {len(file_objs)} files.")
    return docs

def build_local_vectorstore(docs: List[dict]):
    texts = [d["page_content"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]
    faiss_index.build(texts, metadatas)

# ---------------------------
# Tools (simulated)
# ---------------------------
def tool_check_chiller_status(args: str) -> str:
    mocked = {
        "chiller1": {"temp": 6.1, "rpm": 1450, "status": "OK"},
        "chiller2": {"temp": 12.4, "rpm": 1300, "status": "ALARM_LOW_FLOW"},
        "chiller3": {"temp": 7.8, "rpm": 1420, "status": "OK"},
    }
    key = (args or "all").strip().lower()
    if key in ("all", ""):
        return json.dumps(mocked, indent=2)
    return json.dumps({args: mocked.get(args, "unknown chiller")}, indent=2)

def tool_run_flow_test(args: str) -> str:
    time.sleep(1.0)
    return f"Flow test completed for circuit '{args}'. Measured flow: 64 L/min (simulated)."

def tool_search_docs(query: str) -> str:
    try:
        results = faiss_index.search(query, k=4)
    except Exception as e:
        return f"Doc search error: {e}"
    if not results:
        return "No relevant docs found."
    out = []
    for r in results:
        src = r["metadata"].get("source", "?")
        out.append(f"SOURCE: {src} — {r['text'][:400]}")
    return "\n\n".join(out)

# ---------------------------
# Simple "agent" router
# ---------------------------
def run_agent(user_input: str) -> str:
    """
    Very small agent/router:
     - If user asks a tool-like command (contains keywords), execute tool
     - Else: do retrieval (doc_search) and generate a simple answer by concatenating snippets
    """
    ui = user_input.strip().lower()

    # Tool triggers heuristics
    if ui.startswith("get chiller") or "chiller status" in ui or ui.startswith("check chiller"):
        # try to extract id e.g. 'chiller2' else all
        parts = user_input.split()
        ch_id = ""
        for p in parts:
            if p.lower().startswith("chiller"):
                ch_id = p
                break
        return tool_check_chiller_status(ch_id or "all")

    if ui.startswith("run flow test") or "flow test" in ui:
        # extract circuit name naive
        # e.g., "run flow test on condenser-circuit"
        tokens = user_input.split()
        # find token after 'test' or last token
        arg = ""
        if "on" in tokens:
            arg = tokens[tokens.index("on")+1] if tokens.index("on")+1 < len(tokens) else ""
        return tool_run_flow_test(arg or "main-circuit")

    if ui.startswith("search") or ui.startswith("find") or "search" in ui or "doc" in ui:
        # e.g., "search low flow alarm procedure"
        query = user_input.replace("search", "").replace("find", "").strip()
        return tool_search_docs(query or user_input)

    # Default: retrieval + simple synthesis
    try:
        snippets = faiss_index.search(user_input, k=4)
    except Exception as e:
        return f"Retrieval error (index not built?): {e}\n\nTip: Upload docs and click Build Index."

    if not snippets:
        return "No relevant documents found. Try a different query or upload BMS manuals/logs."

    # Simple answer: present snippets then an autogenerated suggestion (dummy LLM)
    aggregated = "\n\n".join([f"SOURCE: {s['metadata'].get('source','?')}\n{s['text']}" for s in snippets])
    # "Generate" an answer using simple heuristics + echo (local LLM fallback)
    answer = "Retrieved information (top snippets):\n\n" + aggregated + "\n\nSuggested next steps (automated):\n"
    # Basic rules: detect 'low flow' or 'flow' or 'temp' keywords in snippets
    top_text = " ".join([s["text"].lower() for s in snippets])
    if "flow" in top_text and "low" in top_text:
        answer += "- Low flow detected in logs. Check pump impeller, filter clogging and differential pressure across heat exchanger.\n"
    if "temp" in top_text or "temperature" in top_text:
        answer += "- High temperature in zones detected. Check AHU fan speeds and chiller setpoints.\n"
    if "alarm" in top_text:
        answer += "- Review alarm logs and follow SOP for alarm code.\n"
    if answer.endswith(":\n"):
        answer += "- Investigate sensors and controllers.\n"
    return answer

# ---------------------------
# Streamlit UI: upload, build index, chat
# ---------------------------
st.markdown("**Upload BMS manuals or logs (.txt)**. After upload, click Build Index, then ask questions or run simulated tools.")

uploaded_files = st.file_uploader("Upload TXT files (BMS logs/manuals)", type=["txt"], accept_multiple_files=True)
if uploaded_files:
    docs = ingest_documents(uploaded_files)
    st.session_state["last_ingested_docs"] = docs
else:
    docs = st.session_state.get("last_ingested_docs", None)

if st.button("Build Index") or (faiss_index.load() and False):
    if not docs:
        st.warning("Please upload at least one TXT file before building the index.")
    else:
        with st.spinner("Building local FAISS index..."):
            try:
                build_local_vectorstore(docs)
            except Exception as e:
                st.error(f"Error building index: {e}")
                logger.exception(e)

st.markdown("---")
st.subheader("Ask or run tools")
user_input = st.text_input("Example: 'Why is chiller2 in low flow alarm?' or 'Get chiller status'")

if st.button("Submit") and user_input:
    if not faiss_index.index and not os.path.exists(faiss_index.index_path):
        st.warning("Index not present. Please upload docs and click Build Index.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Ensure index loaded if not in memory
                if faiss_index.index is None:
                    faiss_index.load()
                response = run_agent(user_input)
            except Exception as e:
                response = f"Agent error: {e}"
                logger.exception(e)
        st.text_area("Response", value=response, height=300)

st.markdown("---")
st.caption("This minimized app avoids heavy LangChain imports and uses local embeddings + FAISS. For full LangChain agents integration, install and align langchain versions carefully.")
