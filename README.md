# LangChain RAG demo

Minimal RAG sample: remote CS229 PDFs, local embeddings, Chroma, chat completions via OpenRouter.

## Stack

| Layer | Choice |
|-------|--------|
| Documents | PyPDF, HTTP URLs |
| Chunks | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace sentence-transformers (local) |
| Store | Chroma on disk |
| LLM | OpenRouter (OpenAI-compatible API) |
| Optional retrieval | Contextual compression (`EmbeddingsFilter` + same HF embeddings) |

## Layout

| File | Role |
|------|------|
| `main.py` | Entry, demo queries |
| `config.py` | Settings from env |
| `documents.py` | Load and split PDFs |
| `vectorstore.py` | Embeddings and Chroma |
| `llm.py` | ChatOpenAI pointed at OpenRouter |
| `chains.py` | RetrievalQA; optional `ContextualCompressionRetriever` + `EmbeddingsFilter` |
| `compression.py` | Streamlit-only: compression preview chart and table |
| `streamlit_app.py` | Web UI |
| `.streamlit/config.toml` | `fileWatcherType = none` (quieter console on Windows) |
| `.env.example` | Copy to `.env` |
| `notebook.ipynb` | Course-style walkthrough (optional) |
| `requirements.txt` | Dependencies (includes `torchvision` for Streamlit imports) |

## Setup

1. Python 3.11 or newer recommended.
2. Create the project virtual environment: `python -m venv .venv` (folder `.venv` is expected in this repo).
3. Activate `.venv`:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - Windows cmd: `.venv\Scripts\activate.bat`
   - macOS or Linux: `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

## Run

CLI (prints sample retrieval and one answer):

`python main.py`

Web UI:

`streamlit run streamlit_app.py`

`.streamlit/config.toml` turns off Streamlit’s file watcher so the console is not flooded when `transformers` is imported (you lose auto-reload on save; restart the process after code changes).

First run downloads embedding weights and PDFs; Chroma lives under `CHROMA_PERSIST_DIR` (default `docs/chroma`). The UI reuses that folder until you click **Rebuild index**. Sidebar: **Contextual compression** (and `.env` `USE_CONTEXTUAL_COMPRESSION`) toggles filtering; when on, a compression section shows fetched vs kept chunks after each answer.

## Environment

| Variable | Required | Default |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | yes | |
| `OPENROUTER_BASE_URL` | no | `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | no | `openai/gpt-4o-mini` |
| `OPENROUTER_HTTP_REFERER` | no | `http://localhost` |
| `OPENROUTER_APP_NAME` | no | `langchain-rag` |
| `EMBEDDING_MODEL` | no | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | no | `docs/chroma` |
| `PDF_BASE_URL` | no | GitHub raw base for cs229 lectures |
| `USE_CONTEXTUAL_COMPRESSION` | no | `false`: plain retriever; `true`: EmbeddingsFilter on fetched chunks |
| `COMPRESSION_FETCH_K` | no | Chunks to retrieve before filtering (default `max(2*k, 6)` when compression on) |

OpenRouter does not supply embeddings in this project; vectors use the local model above.
