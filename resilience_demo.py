"""resilience_demo.py - Task 16: exponential-backoff retry, per-node timeout, global timeout."""
import asyncio
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.types import RetryPolicy
except ImportError:
    from langgraph.pregel import RetryPolicy

from tools import check_order_status


class TransientError(Exception):
    """Simulated temporary failure (jaise upstream service down)."""


class NodeTimeoutError(Exception):
    """Ek node apne per-node timeout se zyada chala."""


# ---- Settings ----
RETRY = RetryPolicy(initial_interval=0.2, backoff_factor=2.0, max_interval=1.0,
                    max_attempts=4, jitter=True, retry_on=TransientError)
NODE_TIMEOUT_S = 1.0
GLOBAL_TIMEOUT_S = 5.0


class State(TypedDict, total=False):
    record_id: str
    fail_first_n: int
    slow_step_delay: float
    enrich_delay: float
    result: dict
    trace: list


_calls = {"flaky": 0}


def node_timeout(seconds):
    """Per-node timeout: node seconds se zyada chale to clean NodeTimeoutError."""
    def deco(fn):
        async def wrapper(state):
            try:
                return await asyncio.wait_for(fn(state), timeout=seconds)
            except asyncio.TimeoutError:
                raise NodeTimeoutError(f"node '{fn.__name__}' exceeded its {seconds}s timeout") from None
        return wrapper
    return deco


async def receive(state):
    return {"trace": ["receive"]}


async def flaky_lookup(state):
    """Pehli fail_first_n calls fail, phir real tool (check_order_status) chalta hai."""
    _calls["flaky"] += 1
    n = _calls["flaky"]
    if n <= state.get("fail_first_n", 0):
        print(f"    flaky_lookup: call #{n} -> TransientError (simulated)")
        raise TransientError(f"simulated transient failure on call {n}")
    print(f"    flaky_lookup: call #{n} -> SUCCESS")
    return {"result": check_order_status(state["record_id"]),
            "trace": state["trace"] + [f"flaky_lookup(succeeded on call {n})"]}


async def enrich(state):
    """Koi per-node timeout nahi; sirf global timeout ise rok sakta hai."""
    try:
        await asyncio.sleep(state.get("enrich_delay", 0))
    except asyncio.CancelledError:
        print("    enrich: CANCELLED by the global timeout")
        raise
    return {"trace": state["trace"] + ["enrich"]}


@node_timeout(NODE_TIMEOUT_S)
async def slow_step(state):
    await asyncio.sleep(state.get("slow_step_delay", 0))
    return {"trace": state["trace"] + ["slow_step"]}


async def respond(state):
    return {"trace": state["trace"] + ["respond"]}


def build():
    g = StateGraph(State)
    g.add_node("receive", receive)
    try:
        g.add_node("flaky_lookup", flaky_lookup, retry_policy=RETRY)
    except TypeError:                      # purane langgraph version ka naam
        g.add_node("flaky_lookup", flaky_lookup, retry=RETRY)
    g.add_node("enrich", enrich)
    g.add_node("slow_step", slow_step)
    g.add_node("respond", respond)
    g.add_edge(START, "receive")
    g.add_edge("receive", "flaky_lookup")
    g.add_edge("flaky_lookup", "enrich")
    g.add_edge("enrich", "slow_step")
    g.add_edge("slow_step", "respond")
    g.add_edge("respond", END)
    return g.compile()


APP = build()


async def run(label, state):
    _calls["flaky"] = 0
    print(f"\n=== {label} ===")
    t0 = time.perf_counter()
    try:
        out = await asyncio.wait_for(APP.ainvoke(state), timeout=GLOBAL_TIMEOUT_S)
        print("  RESULT: completed ->", out["trace"], "| order status:", out["result"]["status"])
    except NodeTimeoutError as e:
        print("  RESULT: clean per-node timeout error ->", e)
    except asyncio.TimeoutError:
        print(f"  RESULT: GLOBAL timeout after {GLOBAL_TIMEOUT_S}s -> whole run cancelled")
    print(f"  elapsed: {time.perf_counter() - t0:.2f}s")


async def main():
    print("Retry policy : max_attempts=4, initial_interval=0.2s, backoff_factor=2.0, "
          "max_interval=1.0s, jitter=True")
    print(f"Per-node timeout (slow_step): {NODE_TIMEOUT_S}s | Global timeout: {GLOBAL_TIMEOUT_S}s")
    await run("(a) Retry recovers (fails first 2 calls, 3rd succeeds)",
              {"record_id": "NYK-005", "fail_first_n": 2})
    await run("(b) Per-node timeout (slow_step simulates a 3s call)",
              {"record_id": "NYK-005", "slow_step_delay": 3.0})
    await run("(c) Global timeout (enrich simulates an 8s overrun)",
              {"record_id": "NYK-005", "enrich_delay": 8.0})


asyncio.run(main())