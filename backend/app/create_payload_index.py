from dotenv import load_dotenv
load_dotenv()

import os
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "mini_rag_docs"

client.create_payload_index(
    collection_name=collection_name,
    field_name="doc_id",
    field_schema=PayloadSchemaType.KEYWORD
)

print("payload index created for doc_id")
