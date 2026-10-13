# Nykaa Domain Support Agent (LangGraph)

**Track completed: Nykaa (E-commerce & Retail)**

A support agent that answers Nykaa policy questions from a knowledge base (RAG) and checks a specific
order's status (tool). Built with LangGraph, ChromaDB, SentenceTransformers, FastAPI and fastmcp.

**Runs fully under `MOCK_LLM` with zero API keys and zero network access** (after the one-time download of the
`all-MiniLM-L6-v2` embedding model). `MOCK_LLM=1` is the default.
All policies and data are fictional and written for this assignment. No real customer data is used.

## How to run
```bash
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
```
Optional FastAPI server: `python3 -m uvicorn api:app --port 8001` (docs at `/docs`).
Every task's real output is saved in `transcripts/`.

## Part 1 - Dataset and RAG core

### Task 1 - Dataset design choices (reproducible)
- Seed used: 42 (generator starts at 42 and increments the seed until delayed_shipment is between 10% and 30%; no record is hand-edited)
- Category counts (fixed): {'Apparel': 12, 'Beauty': 12, 'Footwear': 10, 'Home': 8, 'Electronics': 8}
- Status counts (fixed): {'Placed': 8, 'Shipped': 10, 'Delivered': 20, 'Returned': 7, 'Refunded': 5}
- Order value range (INR) per category: {'Beauty': (300, 3000), 'Apparel': (500, 5000), 'Footwear': (800, 6000), 'Home': (700, 15000), 'Electronics': (1500, 60000)}
- Probability an order is delayed: 0.2
- Price-range reasoning: Beauty orders are low-ticket (300-3000) while Electronics can reach 60000, which mirrors real online-order values.

Output (`transcripts/task1_dataset.txt`):
```text
Total records: 50
Seed used: 42
Count per category: {'Footwear': 10, 'Beauty': 12, 'Apparel': 12, 'Electronics': 8, 'Home': 8}
Count per status: {'Returned': 7, 'Shipped': 10, 'Delivered': 20, 'Placed': 8, 'Refunded': 5}
delayed_shipment True: 14.0%
Sample: {'record_id': 'NYK-001', 'category': 'Footwear', 'status': 'Returned', 'order_value_inr': 2528, 'days_since_created': 29, 'delayed_shipment': False}
```

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
| kb_fixed | 0.308 |
| kb_sentence | 0.343 |

Full measurements (`transcripts/task4_calibration.txt`):
```text
===== kb_fixed =====
IN-SCOPE (top-1 cosine similarity):
  0.4662  | What is the return period for lipstick?
  0.6507  | How long does a bank refund take for cash on delivery?
  0.7908  | Can I cancel my order after it has shipped?
  0.6872  | How many loyalty points do I need to redeem?
  0.5291  | What is the warranty on a laptop?
  0.5756  | What should I do if my parcel arrives broken?
OUT-OF-SCOPE:
  0.1498  | Who won the cricket world cup in 2011?
  0.1031  | How do I bake a chocolate cake?
  0.1046  | What is the capital of France?
  0.0393  | Explain how a car engine works.
min in-scope = 0.4662, max out-of-scope = 0.1498
Clusters alag hain -> threshold = midpoint = 0.308

===== kb_sentence =====
IN-SCOPE (top-1 cosine similarity):
  0.5495  | What is the return period for lipstick?
  0.7153  | How long does a bank refund take for cash on delivery?
  0.8288  | Can I cancel my order after it has shipped?
  0.6314  | How many loyalty points do I need to redeem?
  0.5956  | What is the warranty on a laptop?
  0.5868  | What should I do if my parcel arrives broken?
OUT-OF-SCOPE:
  0.1368  | Who won the cricket world cup in 2011?
  0.0937  | How do I bake a chocolate cake?
  0.1308  | What is the capital of France?
  0.0906  | Explain how a car engine works.
min in-scope = 0.5495, max out-of-scope = 0.1368
Clusters alag hain -> threshold = midpoint = 0.343

Saved threshold.json: {'kb_fixed': 0.308, 'kb_sentence': 0.343}
```

Demo on 5 in-scope queries plus 1 out-of-scope query that triggers the fallback: `transcripts/task4_grounded.txt`.

### Task 5 - Precision@3 / Recall@3 (document level, chunks mapped to parent docs and de-duplicated)
```text
================ kb_fixed ================

Q: How many days do I have to return a beauty product?
  relevant docs     : ['return_window']
  top-3 chunk docs  : ['return_window', 'return_window', 'size_exchange']
  unique docs       : ['return_window', 'size_exchange']
  Precision@3 = 1/2 = 0.500
  Recall@3    = 1/1 = 1.000

Q: When will I get my refund for a COD order?
  relevant docs     : ['cod_refund']
  top-3 chunk docs  : ['order_cancellation', 'order_cancellation', 'cod_refund']
  unique docs       : ['order_cancellation', 'cod_refund']
  Precision@3 = 1/2 = 0.500
  Recall@3    = 1/1 = 1.000

Q: Can I exchange shoes for a different size?
  relevant docs     : ['size_exchange']
  top-3 chunk docs  : ['size_exchange', 'size_exchange', 'size_exchange']
  unique docs       : ['size_exchange']
  Precision@3 = 1/1 = 1.000
  Recall@3    = 1/1 = 1.000

Q: What should I do if the item I received is damaged?
  relevant docs     : ['damaged_item_claim']
  top-3 chunk docs  : ['damaged_item_claim', 'reverse_pickup', 'damaged_item_claim']
  unique docs       : ['damaged_item_claim', 'reverse_pickup']
  Precision@3 = 1/2 = 0.500
  Recall@3    = 1/1 = 1.000

Q: Who do I contact if my shipment is delayed by more than a week?
  relevant docs     : ['escalation_matrix', 'delivery_sla']
  top-3 chunk docs  : ['delivery_sla', 'damaged_item_claim', 'reverse_pickup']
  unique docs       : ['delivery_sla', 'damaged_item_claim', 'reverse_pickup']
  Precision@3 = 1/3 = 0.333
  Recall@3    = 1/2 = 0.500

>>> kb_fixed: mean Precision@3 = 0.567, mean Recall@3 = 0.900

================ kb_sentence ================

Q: How many days do I have to return a beauty product?
  relevant docs     : ['return_window']
  top-3 chunk docs  : ['return_window', 'return_window', 'size_exchange']
  unique docs       : ['return_window', 'size_exchange']
  Precision@3 = 1/2 = 0.500
  Recall@3    = 1/1 = 1.000

Q: When will I get my refund for a COD order?
  relevant docs     : ['cod_refund']
  top-3 chunk docs  : ['order_cancellation', 'cod_refund', 'cod_refund']
  unique docs       : ['order_cancellation', 'cod_refund']
  Precision@3 = 1/2 = 0.500
  Recall@3    = 1/1 = 1.000

Q: Can I exchange shoes for a different size?
  relevant docs     : ['size_exchange']
  top-3 chunk docs  : ['size_exchange', 'size_exchange', 'size_exchange']
  unique docs       : ['size_exchange']
  Precision@3 = 1/1 = 1.000
  Recall@3    = 1/1 = 1.000

Q: What should I do if the item I received is damaged?
  relevant docs     : ['damaged_item_claim']
  top-3 chunk docs  : ['damaged_item_claim', 'warranty_terms', 'return_window']
  unique docs       : ['damaged_item_claim', 'warranty_terms', 'return_window']
  Precision@3 = 1/3 = 0.333
  Recall@3    = 1/1 = 1.000

Q: Who do I contact if my shipment is delayed by more than a week?
  relevant docs     : ['escalation_matrix', 'delivery_sla']
  top-3 chunk docs  : ['delivery_sla', 'damaged_item_claim', 'order_cancellation']
  unique docs       : ['delivery_sla', 'damaged_item_claim', 'order_cancellation']
  Precision@3 = 1/3 = 0.333
  Recall@3    = 1/2 = 0.500

>>> kb_sentence: mean Precision@3 = 0.533, mean Recall@3 = 0.900

================ SUMMARY ================
kb_fixed     mean P@3 = 0.567 | mean R@3 = 0.900
kb_sentence  mean P@3 = 0.533 | mean R@3 = 0.900
```

**Recommendation:** kb_fixed scored mean P@3 = 0.567 and mean R@3 = 0.900, while kb_sentence scored mean P@3 = 0.533 and mean R@3 = 0.900 on the same 5 queries. I would deploy kb_fixed because it has the higher combined precision and recall.Recall is identical for both strategies (0.900), so the difference comes entirely from precision; because this is measured on only 5 queries, the gap of 0.034 is a small, marginal preference rather than a strong result.

## Part 2 - LangGraph agent

### Task 6 - `check_order_status` and designed escalation score
- Formula: `escalation_score = 0.6 * delayed_shipment + 0.4 * (days_since_created / 30)`
- Both terms are in [0, 1], so the score always stays in [0, 1]. Delayed and older orders are more urgent.
- Escalation threshold: 80th percentile of the scores in my generated dataset (`0.376`).
```text
Formula: escalation_score = 0.6*delayed_shipment + 0.4*(days_since_created/30)
Score min/max in dataset: 0.0 1.0
50th percentile of scores: 0.247
80th percentile of scores: 0.376
90th percentile of scores: 0.740
Chosen threshold (80th percentile) = 0.376
Orders at/above threshold: 10 of 50
NYK-001 -> {'found': True, 'record_id': 'NYK-001', 'status': 'Returned', 'order_value_inr': 2528, 'escalation_score': 0.3867, 'escalation_recommended': True}
NYK-010 -> {'found': True, 'record_id': 'NYK-010', 'status': 'Placed', 'order_value_inr': 2870, 'escalation_score': 0.0667, 'escalation_recommended': False}
NYK-999 -> {'found': False, 'record_id': 'NYK-999', 'error': 'Order not found'}
```

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
```text
# topic                  route   CtxRel  Ground  AnsRel  query
 1 return window          rag      0.469     1.0   0.535  How many days do I have to return a lipstick after delivery?
 2 COD refund             rag      0.576     1.0   0.667  When will I get my money back for a cash on delivery return?
 3 delivery SLA           rag      0.739     1.0   0.788  How many days does standard delivery take in metro cities?
 4 reverse pickup         rag      0.557     1.0   0.597  Will the courier come to my home to collect a return?
 5 warranty               rag      0.506     1.0   0.664  How long is the warranty on electronics I bought?
 6 cancellation           rag      0.666     1.0   0.662  Can I cancel my order after it has shipped?
 7 loyalty points         rag       0.65     1.0   0.709  How many loyalty points are needed to redeem?
 8 payment failure        rag      0.534     1.0   0.557  My payment failed but money was deducted, what happens?
 9 size exchange          rag      0.334     1.0   0.444  Can I swap my sneakers for a bigger size?
10 damaged item           rag      0.606     1.0   0.756  My parcel arrived broken, how soon must I report it?
11 international shipping rag      0.486     1.0   0.577  Can I ship perfume sprays to the USA?
12 escalation matrix      rag      0.509     1.0    0.57  Who handles my complaint if the first support agent does not reply within a day?
13 EDGE CASE              rag      0.454     1.0   0.502  Can I return a perfume after 20 days?
14 OUT OF SCOPE           rag      0.123     1.0   0.079  Who won the IPL in 2023?
     -> agent refused: I don't know based on the available policy documents.
15 OUT OF SCOPE           rag       0.07     1.0     0.0  Can you recommend a good movie to watch tonight?
     -> agent refused: I don't know based on the available policy documents.

--- Averages over 15 queries ---
mean context_relevance  = 0.485
mean groundedness       = 1.000
mean answer_relevance   = 0.540
```

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
