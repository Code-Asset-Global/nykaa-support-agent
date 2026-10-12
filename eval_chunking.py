"""eval_chunking.py - Task 5: Precision@3 / Recall@3 (document level), dono collections ke liye."""
from rag import retrieve
from queries import DEMO_QUERIES

K = 3


def dedup_keep_order(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


summary = {}
for coll in ["kb_fixed", "kb_sentence"]:
    print(f"\n================ {coll} ================")
    precisions, recalls = [], []
    for q, relevant in DEMO_QUERIES:
        chunks = retrieve(q, coll, k=K)
        chunk_docs = [c["doc_id"] for c in chunks]
        unique_docs = dedup_keep_order(chunk_docs)          # chunks ko parent doc se map karke dedup
        hits = [d for d in unique_docs if d in relevant]
        p = len(hits) / len(unique_docs)
        r = len(hits) / len(relevant)
        precisions.append(p)
        recalls.append(r)
        print(f"\nQ: {q}")
        print(f"  relevant docs     : {relevant}")
        print(f"  top-3 chunk docs  : {chunk_docs}")
        print(f"  unique docs       : {unique_docs}")
        print(f"  Precision@3 = {len(hits)}/{len(unique_docs)} = {p:.3f}")
        print(f"  Recall@3    = {len(hits)}/{len(relevant)} = {r:.3f}")
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    summary[coll] = (avg_p, avg_r)
    print(f"\n>>> {coll}: mean Precision@3 = {avg_p:.3f}, mean Recall@3 = {avg_r:.3f}")

print("\n================ SUMMARY ================")
for coll, (p, r) in summary.items():
    print(f"{coll:12} mean P@3 = {p:.3f} | mean R@3 = {r:.3f}")