from dotenv import load_dotenv
load_dotenv()

import os, time, json
import google.generativeai as genai
from app.retrieval import search

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def _compact_sources(hits, max_chars_each=900):
    src = []
    for i, h in enumerate(hits, 1):
        t = (h.payload or {}).get("text", "") or ""
        t = t.strip().replace("\n", " ")
        if len(t) > max_chars_each:
            t = t[:max_chars_each] + "..."
        src.append({"idx": i, "score": float(getattr(h, "score", 0.0)), "text": t})
    return src

def rerank_with_gemini(query, hits, keep=5):
    sources = _compact_sources(hits)

    prompt = (
        "You are a reranker for RAG.\n"
        "Given a user query and candidate chunks, return the most relevant chunk indices.\n"
        "Return ONLY valid JSON like: {\"order\":[3,1,2],\"reason\":\"...\"}\n\n"
        f"Query: {query}\n\n"
        f"Chunks:\n{json.dumps(sources, ensure_ascii=False)}\n"
    )

    resp = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )

    text = (resp.text or "").strip()

    try:
        data = json.loads(text)
        order = data.get("order", [])
        order = [int(x) for x in order if str(x).isdigit()]
    except Exception:
        order = []

    if not order:
        return hits[:keep]

    idx_to_hit = {i + 1: h for i, h in enumerate(hits)}
    ranked = [idx_to_hit[i] for i in order if i in idx_to_hit]

    seen = set(id(r) for r in ranked)
    for h in hits:
        if id(h) not in seen:
            ranked.append(h)

    return ranked[:keep]

def build_context_with_labels(hits, max_chars_total=12000):
    labeled = []
    total = 0
    for i, h in enumerate(hits, 1):
        t = (h.payload or {}).get("text", "") or ""
        t = t.strip()
        if not t:
            continue
        block = f"[{i}] {t}"
        if total + len(block) > max_chars_total:
            break
        labeled.append(block)
        total += len(block)
    return "\n\n".join(labeled)

def rough_token_estimate(s):
    return max(1, int(len(s) / 4))

# ✅ UPDATED: accepts doc_id and uses it in search()
def answer(query, doc_id=None, k_retrieve=8, k_rerank=5, min_score=0.55):
    t0 = time.time()

    hits = search(query, k=k_retrieve, doc_id=doc_id)

    top_score = float(getattr(hits[0], "score", 0.0)) if hits else 0.0
    if not hits or top_score < min_score:
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "answer": "I don't know.",
            "sources": [],
            "timing_ms": elapsed_ms,
            "token_estimate": rough_token_estimate(query),
            "top_score": top_score
        }

    reranked = rerank_with_gemini(query, hits, keep=k_rerank)

    ctx = build_context_with_labels(reranked)

    prompt = f"""You are a helpful assistant.
Use ONLY the context below to answer.
When you use a fact from a chunk, add an inline citation like [1] or [2].
If the answer is not in the context, say "I don't know."

Context:
{ctx}

Question: {query}
Answer:"""

    resp = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )

    ans = (resp.text or "").strip()

    elapsed_ms = int((time.time() - t0) * 1000)
    token_est = rough_token_estimate(query + "\n" + ctx + "\n" + ans)

    sources_out = []
    for i, h in enumerate(reranked, 1):
        p = h.payload or {}
        sources_out.append({
            "ref": i,
            "score": float(getattr(h, "score", 0.0)),
            "text": (p.get("text") or "")[:500]
        })

    return {
        "answer": ans,
        "sources": sources_out,
        "timing_ms": elapsed_ms,
        "token_estimate": token_est,
        "top_score": top_score
    }

if __name__ == "__main__":
    q = input("query: ")
    did = input("doc_id (optional): ").strip() or None
    out = answer(q, doc_id=did, k_retrieve=8, k_rerank=5)
    print("\nanswer:\n", out["answer"])
    print("\nmeta:", {"timing_ms": out["timing_ms"], "token_estimate": out["token_estimate"], "top_score": out["top_score"]})
    for s in out["sources"]:
        print(f"\n[{s['ref']}] score={s['score']:.3f}\n{s['text']}")
