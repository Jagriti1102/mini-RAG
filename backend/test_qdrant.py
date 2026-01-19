from dotenv import load_dotenv
load_dotenv()

import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

print("QDRANT_URL =", os.getenv("QDRANT_URL"))
print("QDRANT_API_KEY =", os.getenv("QDRANT_API_KEY"))

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

if not client.collection_exists("test_vectors"):
    client.create_collection(
        collection_name="test_vectors",
        vectors_config=VectorParams(
            size=3,
            distance=Distance.COSINE
        )
    )

print("Test collection ready")
print("Connected to Qdrant")
