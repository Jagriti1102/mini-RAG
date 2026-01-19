from qdrant_client import QdrantClient
import uuid
import os

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def upsert_chunk(vector, payload):
    client.upsert(
        collection_name="mini_rag_docs",
        points=[{
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": payload
        }]
    )
