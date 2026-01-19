from dotenv import load_dotenv
load_dotenv()

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.embeddings import embed_texts

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "mini_rag_docs"

def search(query, k=8, doc_id=None):
    qvec = embed_texts([query])[0]

    flt = None
    if doc_id:
        flt = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

    res = client.query_points(
        collection_name=collection_name,
        query=qvec,
        limit=k,
        with_payload=True,
        query_filter=flt
    )

    return res.points

if __name__ == "__main__":
    q = input("query: ")
    did = input("doc_id (optional): ").strip() or None
    res = search(q, k=5, doc_id=did)
    for i, h in enumerate(res, 1):
        p = h.payload or {}
        print(f"\n#{i} score={h.score}")
        print("text:", (p.get("text") or "")[:400])
