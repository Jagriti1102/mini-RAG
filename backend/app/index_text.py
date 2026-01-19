from dotenv import load_dotenv
load_dotenv()

import os, uuid, zlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from app.embeddings import embed_texts

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "mini_rag_docs"

def simple_chunk(text, max_chars=1200, overlap=150):
    text = text.strip()
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += max_chars - overlap
    return chunks

def _base_id_from_doc_id(doc_id: str) -> int:
    # stable 32-bit int based on doc_id
    return zlib.crc32(doc_id.encode("utf-8")) * 100000

def index_pasted_text(text, doc_id=None):
    if not doc_id:
        doc_id = str(uuid.uuid4())

    chunks = simple_chunk(text)
    vecs = embed_texts(chunks)

    base = _base_id_from_doc_id(doc_id)

    points = []
    for i, (ch, v) in enumerate(zip(chunks, vecs)):
        points.append(
            PointStruct(
                id=base + i,   
                vector=v,
                payload={
                    "text": ch,
                    "doc_id": doc_id,
                    "chunk_id": i
                }
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    return {"doc_id": doc_id, "chunks": len(chunks)}
