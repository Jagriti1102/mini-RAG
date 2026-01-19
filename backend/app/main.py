from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.ask_core import answer
from app.index_text import index_pasted_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IndexReq(BaseModel):
    text: str

class AskReq(BaseModel):
    query: str
    doc_id: str
    k: int = 8

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/index")
def index(req: IndexReq):
    return index_pasted_text(req.text)

@app.post("/ask")
def ask(req: AskReq):
    return answer(req.query, doc_id=req.doc_id, k_retrieve=req.k, k_rerank=5, min_score=0.55)
