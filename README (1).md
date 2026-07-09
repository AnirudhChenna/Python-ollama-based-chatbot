# 🧠 Local RAG Chatbot (Ollama-powered, zero API cost)

A fully local, RAG-based chatbot. It answers questions grounded in your own
documents (PDF/DOCX/TXT/MD), using:

| Layer | Tool | Why |
|---|---|---|
| LLM (generation) | **Ollama** (Llama 3.1, Mistral, Qwen2.5, Phi-4, Gemma2, ...) | Runs 100% on your machine — no API key, no per-token billing |
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers) | Strong open-source retrieval embedding model, runs locally |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-scores retrieved chunks for much higher answer precision |
| Vector store | **ChromaDB** (persistent, local) | Stores embeddings on disk, survives restarts |
| UI | **Streamlit** | Chat interface, file upload, model picker |

---

## 1. Install Ollama

- **macOS / Windows:** download from https://ollama.com/download and install like any app.
- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Start the Ollama service (it usually starts automatically after install; if not):
```bash
ollama serve
```

## 2. Pull a chat model

Pick any model you like based on your RAM. Good starting points:

```bash
ollama pull llama3.1        # 8B, great all-rounder, needs ~8GB RAM
ollama pull mistral         # 7B, fast, needs ~8GB RAM
ollama pull qwen2.5:7b      # strong reasoning + multilingual
ollama pull phi4            # small, very capable for its size
```

You can pull more than one and switch between them from the sidebar dropdown.

## 3. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run will also auto-download the embedding model (~400MB) and the
cross-encoder re-ranker (~90MB) from Hugging Face — this happens once and is
cached locally afterward.

## 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 5. Use it

1. In the sidebar, pick the Ollama model you pulled.
2. (Optional) Upload documents and click **Process & Index Documents**.
3. Ask questions in the chat box — answers will be grounded in your documents
   when relevant, with source file names shown as chips under each answer.
4. Toggle **"Use document knowledge base (RAG)"** off if you just want a
   plain local chatbot with no retrieval.

---

## How the RAG pipeline works

1. **Chunking** — documents are split into ~900-character chunks with 150
   characters of overlap, breaking on sentence boundaries where possible.
2. **Embedding** — each chunk is embedded with `bge-base-en-v1.5` and stored
   in ChromaDB along with its source filename.
3. **Retrieval** — on a query, the top 8 nearest chunks are pulled by cosine
   similarity.
4. **Re-ranking** — a cross-encoder re-scores all 8 candidates *jointly*
   against the query (much more accurate than embedding similarity alone),
   and the top 4 are kept.
5. **Generation** — the top chunks are inserted into a system prompt instructing
   the model to answer from the context when possible, and to say so honestly
   when the answer isn't in the documents, and then answer with the Ollama
   model, streamed token-by-token back to the UI.

## Troubleshooting

- **"Couldn't reach Ollama"** → make sure `ollama serve` is running and the
  model name in the sidebar matches exactly what `ollama list` shows.
- **Slow first response** → the first call loads the model into memory; it's
  fast afterward. Larger models (13B+) need more RAM/VRAM.
- **Nothing found in documents** → check the "Currently indexed" chunk count
  in the sidebar; if it's 0, click "Process & Index Documents" again.
- **Out of memory** → use a smaller model, e.g. `ollama pull llama3.2:3b` or
  `ollama pull phi4-mini`.

## Swapping in different models

- Change `EMBED_MODEL_NAME` in `app.py` to any sentence-transformers model
  (e.g. `BAAI/bge-large-en-v1.5` for higher accuracy at the cost of speed).
- Change `RERANK_MODEL_NAME` to a different cross-encoder if desired.
- Any model you `ollama pull` will automatically show up in the sidebar
  dropdown — no code changes needed to switch LLMs.
