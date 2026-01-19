from dotenv import load_dotenv
load_dotenv()

import os, time
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed_texts(texts, batch_size=8, tries=3):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        last = None

        for t in range(tries):
            try:
                res = client.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=1536
                    )
                )
                out.extend([e.values for e in res.embeddings])
                last = None
                break
            except Exception as e:
                last = e
                time.sleep(1.5 * (t + 1))

        if last is not None:
            raise last

    return out
