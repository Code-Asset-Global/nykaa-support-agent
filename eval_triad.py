"""eval_triad.py - Task 13: RAG triad (MOCK_LLM judge) on 15 queries."""
import os
import re
import numpy as np

from agent import run_agent
from grounded import COLLECTION, FALLBACK
from rag import get_model, retrieve

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

# Asli LLM ke liye ye prompt bheja jata. MOCK_LLM mein neeche ka deterministic judge chalta hai.
JUDGE_PROMPT = """You are an impartial judge for a customer-support RAG system.
Score each item from 0.0 to 1.0 and reply with JSON only.
QUESTION: {query}
RETRIEVED CONTEXT: {context}
ANSWER: {answer}
1. context_relevance: how relevant is the retrieved context to the question?
2. groundedness: is every claim in the answer supported by the context? (a refusal makes no claims)
3. answer_relevance: does the answer address the question?
Reply as: {{"context_relevance": x, "groundedness": y, "answer_relevance": z}}"""

# 12 required topics (1 query each) + 1 edge case + 2 out-of-scope = 15
TEST_SET = [
    ("return window", "How many days do I have to return a lipstick after delivery?"),
    ("COD refund", "When will I get my money back for a cash on delivery return?"),
    ("delivery SLA", "How many days does standard delivery take in metro cities?"),
    ("reverse pickup", "Will the courier come to my home to collect a return?"),
    ("warranty", "How long is the warranty on electronics I bought?"),
    ("cancellation", "Can I cancel my order after it has shipped?"),
    ("loyalty points", "How many loyalty points are needed to redeem?"),
    ("payment failure", "My payment failed but money was deducted, what happens?"),
    ("size exchange", "Can I swap my sneakers for a bigger size?"),
    ("damaged item", "My parcel arrived broken, how soon must I report it?"),
    ("international shipping", "Can I ship perfume sprays to the USA?"),
    ("escalation matrix", "Who handles my complaint if the first support agent does not reply within a day?"),
    ("EDGE CASE", "Can I return a perfume after 20 days?"),
    ("OUT OF SCOPE", "Who won the IPL in 2023?"),
    ("OUT OF SCOPE", "Can you recommend a good movie to watch tonight?"),
]


def _cos(a, b):
    ea, eb = get_model().encode([a, b], normalize_embeddings=True)
    return float(np.dot(ea, eb))


def _clip(x):
    return max(0.0, min(1.0, x))


def mock_judge(query, context_chunks, answer):
    """Deterministic judge (koi API nahi). Teeno scores 0 se 1."""
    if not MOCK_LLM:
        raise NotImplementedError("Real judge: JUDGE_PROMPT ko LLM ko bhejo (env flag ke peeche).")
    ctx_text = " ".join(context_chunks)
    # 1. context relevance = query aur retrieved chunks ki mean cosine similarity
    ctx_rel = np.mean([_cos(query, c) for c in context_chunks]) if context_chunks else 0.0
    # 2. groundedness = answer ke kitne sentences retrieved context mein milte hain (refusal = koi claim nahi = 1.0)
    if answer == FALLBACK:
        grounded = 1.0
    else:
        body = answer.replace("According to our policy:", "").strip()
        sents = [s for s in re.split(r"(?<=[.!?])\s+", body) if s]
        grounded = sum(s in ctx_text for s in sents) / len(sents) if sents else 0.0
    # 3. answer relevance = query aur answer ki cosine similarity
    body = answer.replace("According to our policy:", "").strip()
    ans_rel = _cos(query, body)
    return {"context_relevance": round(_clip(ctx_rel), 3),
            "groundedness": round(_clip(grounded), 3),
            "answer_relevance": round(_clip(ans_rel), 3)}


if __name__ == "__main__":
    print(f"Collection: {COLLECTION} | MOCK_LLM: {MOCK_LLM}")
    print("\n--- Judge prompt template ---")
    print(JUDGE_PROMPT)
    rows = []
    print("\n--- Per-query scores ---")
    print(f"{'#':>2} {'topic':22} {'route':6} {'CtxRel':>7} {'Ground':>7} {'AnsRel':>7}  query")
    for i, (topic, q) in enumerate(TEST_SET, 1):
        r = run_agent(q, f"eval_{i}")
        ctx = [c["text"] for c in retrieve(r["masked_query"], COLLECTION, 3)[:2]]
        s = mock_judge(q, ctx, r["answer"])
        rows.append(s)
        print(f"{i:>2} {topic:22} {r['route']:6} {s['context_relevance']:>7} "
              f"{s['groundedness']:>7} {s['answer_relevance']:>7}  {q}")
        if not r["grounded"]:
            print(f"     -> agent refused: {r['answer']}")
    print("\n--- Averages over", len(rows), "queries ---")
    for k in ("context_relevance", "groundedness", "answer_relevance"):
        print(f"mean {k:18} = {sum(x[k] for x in rows) / len(rows):.3f}")