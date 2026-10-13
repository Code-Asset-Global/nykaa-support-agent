"""grounded.py - Task 4: sirf retrieved context se jawab, warna fallback."""
import os
import json
from rag import retrieve
from queries import DEMO_QUERIES, OUT_OF_SCOPE_DEMO

COLLECTION = "kb_fixed"
MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"
FALLBACK = "I don't know based on the available policy documents."


def load_threshold(collection):
    with open("threshold.json") as f:
        return json.load(f)[collection]


def generate_answer(query, context_chunks):
    """MOCK_LLM: deterministic, sirf context ke text se jawab banata hai (koi API nahi)."""
    if not MOCK_LLM:
        raise NotImplementedError("Real LLM yahan jodna ho to env flag ke peeche lagao.")
    seen, parts = set(), []
    for c in context_chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            parts.append(c["text"])
    return "According to our policy: " + " ".join(parts)


def answer(query, collection=COLLECTION, k=3):
    chunks = retrieve(query, collection, k)
    thr = load_threshold(collection)
    top = chunks[0]["similarity"]
    if top < thr:
        return {"answer": FALLBACK, "grounded": False, "top_similarity": top,
                "threshold": thr, "sources": []}
    context = [c for c in chunks if c["similarity"] >= thr]
    return {"answer": generate_answer(query, context), "grounded": True,
            "top_similarity": top, "threshold": thr,
            "sources": sorted({c["doc_id"] for c in context})}


if __name__ == "__main__":
    print("Collection:", COLLECTION, "| MOCK_LLM:", MOCK_LLM)
    for q, _ in DEMO_QUERIES:
        r = answer(q)
        print(f"\nQ: {q}\n  grounded={r['grounded']} top_sim={r['top_similarity']} "
              f"thr={r['threshold']} sources={r['sources']}\n  A: {r['answer']}")
    r = answer(OUT_OF_SCOPE_DEMO)
    print(f"\nQ (out-of-scope): {OUT_OF_SCOPE_DEMO}\n  grounded={r['grounded']} "
          f"top_sim={r['top_similarity']} thr={r['threshold']}\n  A: {r['answer']}")