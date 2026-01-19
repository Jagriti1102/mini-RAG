# mini-RAG

A minimal, hands-on Retrieval-Augmented Generation (RAG) demo I built. It contains a small FastAPI backend and a static frontend to exercise a retrieval → rerank → generate flow using Qdrant as the vector store and OpenAI / Cohere for embeddings/LLM calls.

What’s in this repo
- backend/ — FastAPI backend, simple env checks and Qdrant test scripts
- frontend/ — static UI (frontend/index.html)
- myvenv/ — local virtualenv (ignored for production)

Files worth checking
- backend/check_env.py — quick environment variable checks
- backend/test_qdrant_connection.py — simple script to verify Qdrant connectivity
- backend/requirements.txt — Python dependencies used by the backend
- frontend/index.html — minimal frontend to submit queries

Architecture (high-level)
Documents -> Ingest/Extract -> Chunker (token-based) -> Embeddings -> Qdrant (vector store) -> Retriever -> (Optional) Reranker -> LLM -> Answer

Mermaid (optional rendering):

```mermaid
flowchart LR
  Docs[Files (pdf / md / txt)] --> Ingest[Ingest / Extract]
  Ingest --> Chunk[Chunker (token-based)]
  Chunk --> Emb[Embeddings (OpenAI / Cohere)]
  Emb --> Qdr[Qdrant Vector Store]
  Qdr --> Ret[Retriever (ANN)]
  Ret --> Rer[Reranker (optional)]
  Rer --> LLM[LLM (generation)]
  LLM --> UI[User output (frontend)]
```

Quick start (what I used)
1. Clone
   git clone https://github.com/Jagriti1102/mini-RAG.git
   cd mini-RAG

2. Backend environment
   - Create a Python venv and install dependencies:
     python -m venv .venv
     source .venv/bin/activate
     pip install -r backend/requirements.txt

   - Create a backend/.env (or export env vars). The backend expects the following environment variables (used in this repo):
     - OPENAI_API_KEY
     - COHERE_API_KEY
     - QDRANT_URL (e.g., http://localhost:6333)
     - QDRANT_API_KEY (if your Qdrant instance requires it)
     - EMBEDDING_MODEL (optional; e.g., openai/text-embedding-3-large)
     - VECTOR_STORE=qdrant

   - Quick environment check:
     python backend/check_env.py

   - Verify Qdrant connection:
     python backend/test_qdrant_connection.py

3. Run the backend
   I use Uvicorn to serve the FastAPI app. From the repo root:
   uvicorn backend.app.main:app --reload --port 8000
   (If your main module path differs, replace `backend.app.main:app` with the correct import path.)

4. Frontend
   - Open frontend/index.html in your browser, or serve it with a simple server:
     cd frontend
     python -m http.server 3000
     Visit http://localhost:3000

Key defaults / tuning I used
- Chunking (token-based):
  - chunk_size_tokens: 800
  - chunk_overlap_tokens: 200
  These values balance context locality and recall for the models I used. I rely on a tiktoken-like tokenizer for token counts.

- Retriever / Reranker:
  - top_k_retriever: 50 (candidate pool from Qdrant)
  - top_k_reranker: 5 (final passages to prompt the LLM)
  - Vector metric: cosine

Providers & libraries used in this repo
- Qdrant (vector store) — qdrant-client
- OpenAI — embeddings / LLM (openai)
- Cohere — optional embeddings (cohere)
- tiktoken — token counting
- FastAPI + Uvicorn — backend API

Notes / tips
- Keep embeddings cached for static documents to avoid repeated API costs.
- If you need higher precision, add a cross-encoder reranker before passing passages to the LLM.
- Adjust chunk sizes if you use an LLM with a larger/smaller context window.

Contributing / license
- This is a small demo. If you want changes (add a proper ingest script, switch the vector store, or wire a different LLM), open an issue or a PR.
- License: MIT (add LICENSE file if you want to keep it)

If you want, I can:
- Wire exact run commands to match files under backend/app if you want me to inspect those files and update the README to be fully specific.
- Replace the Mermaid diagram with an exported image or a short diagram that matches your deployment.
