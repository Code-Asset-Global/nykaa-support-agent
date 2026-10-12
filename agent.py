"""agent.py - Tasks 7, 8, 9: LangGraph agent, JSON memory, structured output schema."""
import json
import os
import re
from typing import TypedDict

from jsonschema import validate
from langgraph.graph import StateGraph, START, END

from grounded import COLLECTION, FALLBACK, generate_answer, load_threshold
from guardrails import check_groundedness, detect_injection, mask_pii
from rag import retrieve
from tools import ESCALATION_THRESHOLD, check_order_status

MEMORY_FILE = "memory.json"

ORDER_ID_RE = re.compile(r"\bNYK-\d{3}\b", re.I)
FOLLOWUP_RE = re.compile(r"\b(it|its|that order|this order|same order)\b", re.I)
ORDER_INTENT_RE = re.compile(r"\b(status|track|tracking|where is|escalat\w*)\b", re.I)

# ---------- Task 9: structured output schema ----------
RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["conversation_id", "turn", "route", "masked_query", "answer",
                 "grounded", "sources", "tool_result", "guardrails"],
    "additionalProperties": False,
    "properties": {
        "conversation_id": {"type": "string"},
        "turn": {"type": "integer", "minimum": 1},
        "route": {"enum": ["rag", "order_tool", "needs_order_id", "blocked"]},
        "masked_query": {"type": "string"},
        "answer": {"type": "string", "minLength": 1},
        "grounded": {"type": "boolean"},
        "sources": {"type": "array", "items": {"type": "string"}},
        "tool_result": {"type": ["object", "null"]},
        "guardrails": {
            "type": "object",
            "required": ["pii_masked", "injection_detected", "groundedness_refused"],
            "properties": {
                "pii_masked": {"type": "array", "items": {"type": "string"}},
                "injection_detected": {"type": "boolean"},
                "groundedness_refused": {"type": "boolean"},
            },
        },
    },
}


# ---------- Task 8: JSON file memory ----------
def load_conversation(cid):
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            data = json.load(f)
    else:
        data = {}
    return data.get(cid, {"history": [], "last_order_id": None})


def save_conversation(cid, conv):
    data = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            data = json.load(f)
    data[cid] = conv
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


class AgentState(TypedDict, total=False):
    conversation_id: str
    raw_query: str
    masked_query: str
    pii_found: list
    injection: bool
    intent: str            # rag | order | blocked
    order_id: str
    last_order_id: str
    history: list
    chunks: list
    top_similarity: float
    threshold: float
    draft_answer: str
    final_answer: str
    grounded: bool
    sources: list
    tool_result: dict
    route_label: str
    groundedness_refused: bool
    response: dict


# ---------- Nodes ----------
def input_guard(state):
    """Node 1: PII masking + injection detection + intent classification."""
    masked, found = mask_pii(state["raw_query"])
    injected, _ = detect_injection(masked)
    m = ORDER_ID_RE.search(masked)
    order_id = m.group(0).upper() if m else None
    last = state.get("last_order_id")
    if injected:
        intent = "blocked"
    elif order_id:
        intent = "order"
    elif last and FOLLOWUP_RE.search(masked):
        intent, order_id = "order", last      # memory se pichla order ID
    elif ORDER_INTENT_RE.search(masked):
        intent = "order"                      # order ka sawaal, par ID nahi mila
    else:
        intent = "rag"
    return {"masked_query": masked, "pii_found": found, "injection": injected,
            "intent": intent, "order_id": order_id}


def route_after_input(state):
    """Conditional edge: intent ke hisaab se agla node."""
    return {"blocked": "blocked", "order": "order_node", "rag": "rag_node"}[state["intent"]]


def rag_node(state):
    """Node 2a: knowledge base se retrieval + draft answer."""
    chunks = retrieve(state["masked_query"], COLLECTION, 3)
    return {"chunks": chunks,
            "top_similarity": chunks[0]["similarity"],
            "threshold": load_threshold(COLLECTION),
            "draft_answer": generate_answer(state["masked_query"], chunks[:2]),
            "route_label": "rag"}


def order_node(state):
    """Node 2b: check_order_status tool."""
    oid = state.get("order_id")
    if not oid:
        return {"final_answer": "Please share your order ID (for example NYK-001) so I can check it.",
                "grounded": False, "sources": [], "tool_result": None,
                "route_label": "needs_order_id"}
    res = check_order_status(oid)
    if not res["found"]:
        return {"final_answer": f"I could not find an order with ID {oid}.",
                "grounded": False, "sources": [], "tool_result": res,
                "route_label": "order_tool"}
    rec = "yes" if res["escalation_recommended"] else "no"
    ans = (f"Order {res['record_id']} is {res['status']}. Order value: INR {res['order_value_inr']}. "
           f"Escalation score: {res['escalation_score']} (threshold {ESCALATION_THRESHOLD}). "
           f"Escalation recommended: {rec}.")
    return {"final_answer": ans, "grounded": True, "sources": [], "tool_result": res,
            "route_label": "order_tool", "last_order_id": res["record_id"]}


def blocked_node(state):
    """Node 2c: injection par refusal."""
    return {"final_answer": "I can't help with that request because it looks like an attempt to "
                            "override my instructions. I can answer Nykaa policy questions or check an order status.",
            "grounded": False, "sources": [], "tool_result": None, "route_label": "blocked"}


def output_guard(state):
    """Node 3: output-side groundedness check (sirf RAG answers par)."""
    if state["intent"] != "rag":
        return {"groundedness_refused": False}
    ok, _ = check_groundedness(state["top_similarity"], state["threshold"],
                               state["draft_answer"], state["chunks"][:2])
    if ok:
        return {"final_answer": state["draft_answer"], "grounded": True,
                "sources": sorted({c["doc_id"] for c in state["chunks"][:2]}),
                "tool_result": None, "groundedness_refused": False}
    return {"final_answer": FALLBACK, "grounded": False, "sources": [],
            "tool_result": None, "groundedness_refused": True}


def finalize(state):
    """Node 4: structured response banao, schema se validate karo, memory save karo."""
    history = list(state.get("history", []))
    response = {
        "conversation_id": state["conversation_id"],
        "turn": len(history) + 1,
        "route": state["route_label"],
        "masked_query": state["masked_query"],
        "answer": state["final_answer"],
        "grounded": state["grounded"],
        "sources": state["sources"],
        "tool_result": state.get("tool_result"),
        "guardrails": {"pii_masked": state["pii_found"],
                       "injection_detected": state["injection"],
                       "groundedness_refused": state.get("groundedness_refused", False)},
    }
    validate(instance=response, schema=RESPONSE_SCHEMA)      # fail hua to exception
    history.append({"user": state["masked_query"], "assistant": response["answer"],
                    "route": response["route"]})
    save_conversation(state["conversation_id"],
                      {"history": history, "last_order_id": state.get("last_order_id")})
    return {"response": response}


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("input_guard", input_guard)
    g.add_node("rag_node", rag_node)
    g.add_node("order_node", order_node)
    g.add_node("blocked", blocked_node)
    g.add_node("output_guard", output_guard)
    g.add_node("finalize", finalize)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input,
                            {"blocked": "blocked", "order_node": "order_node", "rag_node": "rag_node"})
    g.add_edge("rag_node", "output_guard")
    g.add_edge("order_node", "output_guard")
    g.add_edge("blocked", "finalize")
    g.add_edge("output_guard", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


_GRAPH = None


def run_agent(query, conversation_id):
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    conv = load_conversation(conversation_id)
    out = _GRAPH.invoke({"conversation_id": conversation_id, "raw_query": query,
                         "history": conv["history"], "last_order_id": conv["last_order_id"]})
    return out["response"]