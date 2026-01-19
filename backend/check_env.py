from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

print("ENV PATH:", env_path)
print("QDRANT_URL:", os.getenv("QDRANT_URL"))
print("QDRANT_API_KEY:", os.getenv("QDRANT_API_KEY"))
print("GEMINI_API_KEY:",os.getenv("GEMINI_API_KEY"))