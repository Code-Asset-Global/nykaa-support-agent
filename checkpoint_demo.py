"""checkpoint_demo.py - Task 15: SQLite checkpoint, interruption, resume (same thread_id)."""
import os
import sqlite3
from collections import Counter

from langgraph.checkpoint.sqlite import SqliteSaver

import agent

DB = "checkpoints.sqlite"
for f in (DB, "memory.json"):
    if os.path.exists(f):
        os.remove(f)

COUNTS = Counter()


def counted(name, fn):
    def wrapper(state):
        COUNTS[name] += 1
        print(f"    >>> EXECUTING node '{name}' (execution #{COUNTS[name]})")
        return fn(state)
    return wrapper


# agent ke asli nodes ko counter ke saath wrap karo (graph banne se pehle)
for n in ("input_guard", "rag_node", "output_guard", "finalize"):
    setattr(agent, n, counted(n, getattr(agent, n)))

THREAD = {"configurable": {"thread_id": "ckpt-demo-1"}}
INPUT = {"conversation_id": "ckpt_demo",
         "raw_query": "How many days do I have to return a beauty product?",
         "history": [], "last_order_id": None}

print("=== (a) RUN 1: 'output_guard' se pehle rokna ===")
conn1 = sqlite3.connect(DB, check_same_thread=False)
g1 = agent.build_graph(SqliteSaver(conn1))
g1.invoke(INPUT, THREAD, interrupt_before=["output_guard"])
snap = g1.get_state(THREAD)
print("\n=== (b) EXECUTION STOPPED ===")
print("Next node waiting :", snap.next)
print("Saved in checkpoint: masked_query =", snap.values.get("masked_query"),
      "| top_similarity =", snap.values.get("top_similarity"))
print("Execution counts  :", dict(COUNTS))
conn1.close()

print("\n=== (c) RESUME: nayi connection + naya graph object, SAME thread_id ===")
conn2 = sqlite3.connect(DB, check_same_thread=False)
g2 = agent.build_graph(SqliteSaver(conn2))
snap2 = g2.get_state(THREAD)
print("Loaded from checkpoints.sqlite -> next node:", snap2.next)
print("Loaded state has masked_query:", snap2.values.get("masked_query"))
before = dict(COUNTS)
out = g2.invoke(None, THREAD)
print("\nFinal answer:", out["response"]["answer"][:120], "...")
print("Counts before resume:", before)
print("Counts after resume :", dict(COUNTS))
assert COUNTS["input_guard"] == 1 and COUNTS["rag_node"] == 1
assert COUNTS["output_guard"] == 1 and COUNTS["finalize"] == 1
print("PASSED: input_guard aur rag_node dobara execute NAHI hue "
      "(unka result checkpoint se load hua); sirf output_guard aur finalize resume ke baad chale.")
conn2.close()