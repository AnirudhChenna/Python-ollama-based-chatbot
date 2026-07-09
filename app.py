"""
Local RAG Chatbot — powered by Ollama (100% free, no API costs)
==================================================================

What this app does
-------------------
1. You upload documents (PDF / DOCX / TXT / MD).
2. The app chunks them, embeds them with a local sentence-transformers
   model, and stores the vectors in a persistent local ChromaDB.
3. When you ask a question, it:
      a. Embeds your question
      b. Retrieves the most relevant chunks (vector search)
      c. Re-ranks them with a cross-encoder for higher precision
      d. Feeds the best chunks + your question into a local LLM
         served by Ollama (Llama 3.1, Mistral, Qwen2.5, Phi-4, etc.)
      e. Streams back a grounded, cited answer

Everything runs on your machine. No Anthropic/OpenAI API key, no
per-token billing, no internet dependency once models are pulled.

Setup (see README.md for full details)
---------------------------------------
1. Install Ollama:            https://ollama.com/download
2. Pull a chat model:         ollama pull llama3.1
3. Pull an embedding model:   ollama pull nomic-embed-text   (optional, we default
                               to a bundled sentence-transformers model so this
                               isn't strictly required)
4. pip install -r requirements.txt
5. streamlit run app.py
"""

import os
import io
import uuid
import hashlib
from datetime import datetime

import streamlit as st

# ---- Optional heavy imports are wrapped so the app still boots and shows
# ---- a friendly setup message if a dependency hasn't been installed yet.
MISSING = []

try:
    import ollama
except ImportError:
    MISSING.append("ollama")

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    MISSING.append("chromadb")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    MISSING.append("sentence-transformers")

try:
    from pypdf import PdfReader
except ImportError:
    MISSING.append("pypdf")

try:
    import docx  # python-docx
except ImportError:
    MISSING.append("python-docx")


# =========================================================================
# PAGE CONFIG & STYLING
# =========================================================================
st.set_page_config(
    page_title="Local RAG Chatbot (Ollama)",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #F0F4FF, #E8E4FF);
    }
    .chat-message {
        padding: 1.2rem 1.5rem;
        border-radius: 1rem;
        margin-bottom: 0.9rem;
        line-height: 1.55;
    }
    .user-message {
        background-color: #4F46E5;
        color: white;
        margin-left: 14%;
    }
    .assistant-message {
        background-color: white;
        color: #1F2937;
        margin-right: 14%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .source-chip {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        border-radius: 0.5rem;
        padding: 0.15rem 0.6rem;
        font-size: 0.78rem;
        margin: 0.15rem 0.25rem 0 0;
        border: 1px solid #C7D2FE;
    }
    </style>
""", unsafe_allow_html=True)


if MISSING:
    st.error(
        "Missing required packages: **" + ", ".join(MISSING) + "**\n\n"
        "Run this in your terminal, then restart the app:\n\n"
        f"```\npip install {' '.join(MISSING)}\n```"
    )
    st.stop()


# =========================================================================
# CONSTANTS
# =========================================================================
DB_DIR = os.path.join(os.getcwd(), "chroma_store")
COLLECTION_NAME = "rag_knowledge_base"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"       # strong, free, local embedding model
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local cross-encoder reranker
CHUNK_SIZE = 900          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between chunks
TOP_K_RETRIEVE = 8        # candidates pulled from vector search
TOP_K_FINAL = 4           # kept after re-ranking, sent to the LLM


# =========================================================================
# CACHED RESOURCES (loaded once per session, not per message)
# =========================================================================
@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~400MB)...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Loading re-ranking model...")
def get_reranker():
    return CrossEncoder(RERANK_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    os.makedirs(DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=DB_DIR)


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def list_ollama_models():
    """Return the list of chat models the user has already pulled locally."""
    try:
        resp = ollama.list()
        models = resp.get("models", []) if isinstance(resp, dict) else resp.models
        names = []
        for m in models:
            name = m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                names.append(name)
        return sorted(set(names))
    except Exception:
        return []


# =========================================================================
# DOCUMENT PARSING
# =========================================================================
def extract_text(file) -> str:
    name = file.name.lower()
    data = file.read()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")

    return ""


def chunk_text(text: str, source: str):
    """Simple recursive-ish character chunker with overlap."""
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        # try to break on a sentence boundary near the end
        boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + 200:
            end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        if end >= n:
            break
        start = end - CHUNK_OVERLAP
    return chunks


def add_documents_to_kb(files):
    embedder = get_embedder()
    collection = get_collection()

    total_chunks = 0
    for file in files:
        text = extract_text(file)
        if not text.strip():
            st.warning(f"Couldn't extract any text from **{file.name}** — skipping.")
            continue

        chunks = chunk_text(text, source=file.name)
        if not chunks:
            continue

        embeddings = embedder.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = [hashlib.sha256(f"{file.name}-{i}-{uuid.uuid4()}".encode()).hexdigest() for i in range(len(chunks))]
        metadatas = [{"source": c["source"], "chunk_index": i} for i, c in enumerate(chunks)]
        documents = [c["text"] for c in chunks]

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        total_chunks += len(chunks)

    return total_chunks


def retrieve_context(query: str):
    embedder = get_embedder()
    reranker = get_reranker()
    collection = get_collection()

    if collection.count() == 0:
        return [], []

    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=min(TOP_K_RETRIEVE, collection.count()),
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return [], []

    # Re-rank with the cross-encoder for much higher precision
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    top = ranked[:TOP_K_FINAL]

    context_chunks = [t[0] for t in top]
    sources = [t[1]["source"] for t in top]
    return context_chunks, sources


def build_prompt(query: str, context_chunks):
    if context_chunks:
        context_block = "\n\n---\n\n".join(context_chunks)
        system = (
            "You are a precise, helpful research assistant. Answer the user's "
            "question using ONLY the context provided below when it is relevant. "
            "If the context does not contain the answer, say so honestly and then "
            "answer from general knowledge, clearly noting that it isn't from the "
            "provided documents. Be concise, accurate, and well-structured. Cite "
            "which source(s) you used when you rely on the context.\n\n"
            f"CONTEXT:\n{context_block}"
        )
    else:
        system = (
            "You are a precise, helpful assistant. No documents have been loaded "
            "into the knowledge base yet, so answer from general knowledge."
        )
    return system


# =========================================================================
# SESSION STATE
# =========================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload some documents in the sidebar (optional) and ask me anything — I'll ground my answers in them when possible. Everything runs locally via Ollama, so there's no API cost.", "sources": []}
    ]

if "kb_chunks" not in st.session_state:
    try:
        st.session_state.kb_chunks = get_collection().count()
    except Exception:
        st.session_state.kb_chunks = 0


# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.header("⚙️ Setup")

    available_models = list_ollama_models()
    if not available_models:
        st.warning(
            "No local Ollama models found. Open a terminal and run, e.g.:\n\n"
            "```\nollama pull llama3.1\n```\n"
            "then refresh this page."
        )
        chat_model = st.text_input("Model name (once pulled)", value="llama3.1")
    else:
        chat_model = st.selectbox("Chat model (local, free)", available_models, index=0)

    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.05)
    use_rag = st.checkbox("Use document knowledge base (RAG)", value=True)

    st.markdown("---")
    st.header("📚 Knowledge Base")
    st.caption(f"Currently indexed: **{st.session_state.kb_chunks}** chunks")

    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / TXT / MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("📥 Process & Index Documents", use_container_width=True, disabled=not uploaded_files):
        with st.spinner("Chunking, embedding, and indexing..."):
            added = add_documents_to_kb(uploaded_files)
            st.session_state.kb_chunks = get_collection().count()
        st.success(f"Indexed {added} new chunks from {len(uploaded_files)} file(s).")

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        client = get_chroma_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        st.session_state.kb_chunks = 0
        st.success("Knowledge base cleared.")
        st.rerun()

    st.markdown("---")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. What would you like to know?", "sources": []}
        ]
        st.rerun()

    st.markdown("---")
    st.caption("**Stack:** Ollama (LLM) · bge-base-en-v1.5 (embeddings) · "
               "ms-marco cross-encoder (re-ranking) · ChromaDB (vector store)")
    st.metric("Messages", len(st.session_state.messages))


# =========================================================================
# HEADER
# =========================================================================
st.title("🧠 Local RAG Chatbot")
st.caption("Runs entirely on your machine via Ollama — no API keys, no per-token cost.")
st.markdown("---")


# =========================================================================
# CHAT HISTORY
# =========================================================================
for message in st.session_state.messages:
    css_class = "user-message" if message["role"] == "user" else "assistant-message"
    label = "👤 You" if message["role"] == "user" else "🤖 Assistant"
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{label}</strong>
        <p style="margin-top: 0.5rem; white-space: pre-wrap;">{message["content"]}</p>
    </div>
    """, unsafe_allow_html=True)

    if message.get("sources"):
        chips = "".join(f'<span class="source-chip">📄 {s}</span>' for s in sorted(set(message["sources"])))
        st.markdown(chips, unsafe_allow_html=True)


# =========================================================================
# CHAT INPUT
# =========================================================================
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
    st.markdown(f"""
    <div class="chat-message user-message">
        <strong>👤 You</strong>
        <p style="margin-top: 0.5rem;">{user_input}</p>
    </div>
    """, unsafe_allow_html=True)

    context_chunks, sources = ([], [])
    if use_rag:
        with st.spinner("🔎 Searching knowledge base..."):
            try:
                context_chunks, sources = retrieve_context(user_input)
            except Exception as e:
                st.warning(f"Retrieval skipped due to an error: {e}")

    system_prompt = build_prompt(user_input, context_chunks)

    ollama_messages = [{"role": "system", "content": system_prompt}]
    # include recent conversation turns for continuity (last 10 messages)
    for m in st.session_state.messages[-10:]:
        ollama_messages.append({"role": m["role"], "content": m["content"]})

    answer_placeholder = st.empty()
    full_answer = ""

    try:
        stream = ollama.chat(
            model=chat_model,
            messages=ollama_messages,
            options={"temperature": temperature},
            stream=True,
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "") if isinstance(chunk, dict) else chunk.message.content
            full_answer += token
            answer_placeholder.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant</strong>
                <p style="margin-top: 0.5rem; white-space: pre-wrap;">{full_answer}▌</p>
            </div>
            """, unsafe_allow_html=True)

        answer_placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": full_answer, "sources": sources})

    except Exception as e:
        st.error(
            f"❌ Couldn't reach Ollama: {e}\n\n"
            "Make sure the Ollama app/service is running (`ollama serve`) and that "
            f"the model **{chat_model}** has been pulled (`ollama pull {chat_model}`)."
        )

    st.rerun()
