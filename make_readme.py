"""make_readme.py - transcripts se README.md banata hai (numbers files se aate hain)."""
import json
import os
import re

import dataset
import tools

FENCE = "`" * 3


def read(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return f"(missing: {path} - pehle wo step chalao)"


def block(text):
    return f"{FENCE}text\n{text}\n{FENCE}"


def after(text, marker):
    return text.split(marker, 1)[1].strip() if marker in text else text


# ---------- numbers ----------
with open("threshold.json") as f:
    thr = json.load(f)

t5 = read("transcripts/task5_eval.txt")
found = re.findall(r"(kb_fixed|kb_sentence)\s+mean P@3 = ([\d.]+) \| mean R@3 = ([\d.]+)", t5)
scores = {c: (float(p), float(r)) for c, p, r in found}
best = None
if len(scores) == 2:
    best = max(scores, key=lambda c: (scores[c][0] + scores[c][1], scores[c][1]))
    recommendation = (
        f"kb_fixed scored mean P@3 = {scores['kb_fixed'][0]:.3f} and mean R@3 = {scores['kb_fixed'][1]:.3f}, "
        f"while kb_sentence scored mean P@3 = {scores['kb_sentence'][0]:.3f} and mean R@3 = {scores['kb_sentence'][1]:.3f} "
        f"on the same 5 queries. I would deploy {best} because it has the higher combined precision and recall. "
        f"EDIT-ME: apni ek-do line ki wajah yahan likho (jaise chunk size ka asar)."
    )
else:
    recommendation = "(Task 5 numbers missing - pehle eval_chunking.py chalao)"

with open("grounded.py") as f:
    m = re.search(r'COLLECTION\s*=\s*"(\w+)"', f.read())
current_collection = m.group(1) if m else "?"

design = f"""- Seed used: {dataset.SEED_USED} (generator starts at {dataset.START_SEED} and increments the seed until delayed_shipment is between 10% and 30%; no record is hand-edited)
- Category counts (fixed): {dataset.CATEGORIES}
- Status counts (fixed): {dataset.STATUSES}
- Order value range (INR) per category: {dataset.PRICE_RANGE_INR}
- Probability an order is delayed: {dataset.DELAY_PROB}
- Price-range reasoning: Beauty orders are low-ticket (300-3000) while Electronics can reach 60000, which mirrors real online-order values."""

task6 = read("transcripts/task6_tool.txt")
formula = (f"escalation_score = {tools.W_DELAY} * delayed_shipment + {tools.W_AGE} * (days_since_created / {tools.MAX_DAYS})")

README = f"""# Nykaa Domain Support Agent (LangGraph)

**Track completed: Nykaa (E-commerce & Retail)**

A support agent that answers Nykaa policy questions from a knowledge base (RAG) and checks a specific
order's status (tool). Built with LangGraph, ChromaDB, SentenceTransformers, FastAPI and fastmcp.

**Runs fully under `MOCK_LLM` with zero API keys and zero network access** (after the one-time download of the
`all-MiniLM-L6-v2` embedding model). `MOCK_LLM=1` is the default.
All policies and data are fictional and written for this assignment. No real customer data is used.

## How to run
{FENCE}bash
python3 -m pip install -r requirements.txt
python3 dataset.py            # Task 1
python3 kb.py                 # Task 2
python3 rag.py                # Task 3 (builds chroma_db with 2 collections)
python3 calibrate.py          # Task 4 (writes threshold.json)
python3 grounded.py           # Task 4 demo
python3 eval_chunking.py      # Task 5
python3 tools.py              # Task 6
python3 demo_agent.py         # Tasks 7-10
python3 demo_api.py           # Tasks 11-12
python3 eval_triad.py         # Task 13
# Task 14: server in background, then the separate client
mkdir -p logs
nohup python3 mcp_server.py > logs/mcp_server.log 2>&1 &
sleep 6 && python3 mcp_client.py
pkill -f mcp_server.py
python3 checkpoint_demo.py    # Task 15
python3 resilience_demo.py    # Task 16
{FENCE}
Optional FastAPI server: `python3 -m uvicorn api:app --port 8001` (docs at `/docs`).
Every task's real output is saved in `transcripts/`.

## Part 1 - Dataset and RAG core

### Task 1 - Dataset design choices (reproducible)
{design}

Output (`transcripts/task1_dataset.txt`):
{block(read("transcripts/task1_dataset.txt"))}

### Task 2 - Knowledge base
12 documents (4 sentences each) in `kb.py` (also exported to `kb/`) covering all required topics.
Output: `transcripts/task2_kb.txt`.

### Task 3 - Two chunking strategies, two ChromaDB collections
- `kb_fixed`: 35-word chunks with 8-word overlap
- `kb_sentence`: one sentence per chunk
- Embeddings: `all-MiniLM-L6-v2`, cosine space, one collection per strategy.
Output: `transcripts/task3_index.txt`.

### Task 4 - Grounded generation and measured threshold
Top-1 cosine similarity was measured on 6 in-scope and 4 out-of-scope queries, and the threshold was set at the
midpoint of the gap (no preset such as 0.5 was used).

| Collection | threshold (midpoint of measured clusters) |
|---|---|
| kb_fixed | {thr['kb_fixed']} |
| kb_sentence | {thr['kb_sentence']} |

Full measurements (`transcripts/task4_calibration.txt`):
{block(read("transcripts/task4_calibration.txt"))}

Demo on 5 in-scope queries plus 1 out-of-scope query that triggers the fallback: `transcripts/task4_grounded.txt`.

### Task 5 - Precision@3 / Recall@3 (document level, chunks mapped to parent docs and de-duplicated)
{block(t5)}

**Recommendation:** {recommendation}

## Part 2 - LangGraph agent

### Task 6 - `check_order_status` and designed escalation score
- Formula: `{formula}`
- Both terms are in [0, 1], so the score always stays in [0, 1]. Delayed and older orders are more urgent.
- Escalation threshold: 80th percentile of the scores in my generated dataset (`{tools.ESCALATION_THRESHOLD}`).
{block(task6)}

### Tasks 7-10 - Graph, memory, schema, guardrails (`agent.py`, `guardrails.py`)
- Graph nodes: `input_guard` -> (conditional edge) -> `rag_node` | `order_node` | `blocked` -> `output_guard` -> `finalize`.
  The conditional edge routes by query intent (order ID / status keywords vs policy question).
- Memory: conversation history and the last order ID are persisted in `memory.json` (only masked text is stored).
- Structured output: every response is validated against `RESPONSE_SCHEMA` (jsonschema) in `finalize`.
- Guardrails: PII masking (phone number and card last-4), prompt-injection detection, and an output-side
  groundedness check that refuses when retrieved context does not support the question.
Transcript with both routes, multi-turn and fresh conversation, schema validation, and each guardrail firing:
`transcripts/task7_10_agent.txt`.

## Part 3 - API, logging, evaluation

### Tasks 11-12 - FastAPI and JSON-Lines logging (`api.py`)
Endpoints: `POST /ask`, `POST /add-document`, `GET /health` (Pydantic request/response models).
Every request writes one JSON line with a trace ID and latency to `logs/requests.jsonl`; the logged query is the
masked text, so raw phone/card values never reach disk. Transcript: `transcripts/task11_12_api.txt`.

### Task 13 - RAG triad on 15 queries (MOCK_LLM judge, `eval_triad.py`)
12 required topics + 1 edge case + 2 out-of-scope queries.
{block(after(read("transcripts/task13_triad.txt"), "--- Per-query scores ---"))}

## Part 4 - Resilience and interoperability
- Task 14 - MCP: `mcp_server.py` (fastmcp, HTTP at `http://127.0.0.1:8000/mcp`) and a separate `mcp_client.py`.
  Transcript: `transcripts/task14_mcp.txt`.
- Task 15 - SQLite checkpointing (`checkpoints.sqlite`, thread ID): run stops before `output_guard`, then resumes
  from the checkpoint; earlier nodes are not re-executed. Transcript: `transcripts/task15_checkpoint.txt`.
- Task 16 - Retry policy (max_attempts=4, initial_interval=0.2s, backoff_factor=2.0, max_interval=1.0s, jitter=True),
  per-node timeout (1s), global timeout (5s). Transcript: `transcripts/task16_resilience.txt`.

## Limitations (honest notes)
- `MOCK_LLM` builds answers by copying the top retrieved sentences, so groundedness scores are naturally high.
- The mock judge uses embedding cosine similarity as a proxy; it is not a real LLM judge.
- PII masking covers only fixed-format fields (phone number, card last-4). Customer name and delivery address are
  free text and are out of scope for masking.
- Timeouts and retries are demonstrated with simulated slow/failing calls.
"""

with open("README.md", "w") as f:
    f.write(README)
print("README.md written.\n")

# ---------- self-check ----------
checks = [
    ("Task 1", "transcripts/task1_dataset.txt", "Total records"),
    ("Task 2", "transcripts/task2_kb.txt", "Total documents: 12"),
    ("Task 3", "transcripts/task3_index.txt", "chunks indexed"),
    ("Task 4 calibration", "transcripts/task4_calibration.txt", "threshold"),
    ("Task 4 grounded", "transcripts/task4_grounded.txt", "grounded=False"),
    ("Task 5", "transcripts/task5_eval.txt", "SUMMARY"),
    ("Task 6", "transcripts/task6_tool.txt", "Chosen threshold"),
    ("Tasks 7-10", "transcripts/task7_10_agent.txt", "schema validation: PASSED"),
    ("Tasks 11-12", "transcripts/task11_12_api.txt", "PII check"),
    ("Task 13", "transcripts/task13_triad.txt", "mean groundedness"),
    ("Task 14", "transcripts/task14_mcp.txt", "MCP content"),
    ("Task 15", "transcripts/task15_checkpoint.txt", "PASSED"),
    ("Task 16", "transcripts/task16_resilience.txt", "GLOBAL timeout"),
]
print("SELF-CHECK")
for name, path, marker in checks:
    ok = os.path.exists(path) and marker in open(path).read()
    print(f"  {'OK     ' if ok else 'MISSING'} {name}  ({path})")
print(f"\nRecommended collection from Task 5 numbers: {best}")
print(f"COLLECTION currently set in grounded.py   : {current_collection}")
if best and best != current_collection:
    print("WARNING: mismatch. Tell me and we will switch the collection and rerun transcripts.")