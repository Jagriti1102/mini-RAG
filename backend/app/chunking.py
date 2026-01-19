def chunk_text(text, size=1000, overlap=150):
    chunks = []
    start = 0
    pos = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]

        chunks.append({
            "text": chunk,
            "position": pos
        })

        pos += 1
        start = end - overlap

    return chunks
