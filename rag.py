"""rag.py - Task 3: do chunking strategies + do alag ChromaDB collections."""
import re
import chromadb
from sentence_transformers import SentenceTransformer
from kb import KB_DOCUMENTS

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_db"
FIXED_WORDS = 35      # fixed-size chunk: 35 words
OVERLAP_WORDS = 8     # agle chunk se 8 words overlap

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def fixed_chunks(text, size=FIXED_WORDS, overlap=OVERLAP_WORDS):
    """Fixed-size (words) chunks with overlap."""
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def sentence_chunks(text):
    """Har sentence ek chunk."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


STRATEGIES = {"kb_fixed": fixed_chunks, "kb_sentence": sentence_chunks}


def build_index():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    model = get_model()
    for name, chunk_fn in STRATEGIES.items():
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        ids, texts, metas = [], [], []
        for d in KB_DOCUMENTS:
            for i, ch in enumerate(chunk_fn(d["text"])):
                ids.append(f"{d['doc_id']}::{i}")
                texts.append(ch)
                metas.append({"doc_id": d["doc_id"], "chunk_index": i})
        embs = model.encode(texts, normalize_embeddings=True).tolist()
        col.add(ids=ids, documents=texts, embeddings=embs, metadatas=metas)
        print(f"{name}: {len(texts)} chunks indexed")


def retrieve(query, collection_name="kb_sentence", k=3):
    """Top-k chunks wapas deta hai, cosine similarity ke saath."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(collection_name)
    emb = get_model().encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=k)
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"doc_id": meta["doc_id"], "text": doc, "similarity": round(1 - dist, 4)})
    return out


if __name__ == "__main__":
    build_index()
    q = "How many days do I have to return a beauty product?"
    for name in STRATEGIES:
        print(f"\n=== {name} | query: {q}")
        for r in retrieve(q, name, k=3):
            print(f"  sim={r['similarity']}  doc={r['doc_id']}  | {r['text'][:90]}...")