from dotenv import load_dotenv
load_dotenv()

import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from embeddings import embed_texts

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "mini_rag_docs"

texts = [
    "Artificial Intelligence is a field of computer science.",
    "Machine learning is a subset of AI.",
    "Large Language Models are trained on vast amounts of data."
]

vectors = embed_texts(texts)

points = [
    PointStruct(
        id=i,
        vector=vectors[i],
        payload={"text": texts[i]}
    )
    for i in range(len(texts))
]

client.upsert(
    collection_name=collection_name,
    points=points
)

print("Vectors inserted successfully")
