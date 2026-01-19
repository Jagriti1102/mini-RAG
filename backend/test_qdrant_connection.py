from dotenv import load_dotenv
from pathlib import Path
import os
from qdrant_client import QdrantClient

load_dotenv(Path(__file__).parent / ".env")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collections = client.get_collections()
print("Connected. Collections:", collections)
