"""api.py - Tasks 11-12: FastAPI backend + JSON-Lines structured logging."""
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import run_agent
from guardrails import mask_pii
from rag import CHROMA_PATH, STRATEGIES, get_model

LOG_FILE = "logs/requests.jsonl"
os.makedirs("logs", exist_ok=True)

app = FastAPI(title="Nykaa Support Agent")


# ---------- Pydantic models ----------
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    conversation_id: str = Field("default", min_length=1, max_length=64)


class GuardrailInfo(BaseModel):
    pii_masked: list[str]
    injection_detected: bool
    groundedness_refused: bool


class AskResponse(BaseModel):
    trace_id: str
    conversation_id: str
    turn: int
    route: str
    masked_query: str
    answer: str
    grounded: bool
    sources: list[str]
    tool_result: Optional[dict] = None
    guardrails: GuardrailInfo


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., pattern=r"^[a-z0-9_]{3,40}$")
    topic: str = Field(..., min_length=3, max_length=100)
    text: str = Field(..., min_length=20, max_length=3000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_added: dict


# ---------- Logging (JSON Lines, ek request = ek line) ----------
def write_log(entry):
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    trace_id = uuid.uuid4().hex
    # body log nahi karte (usme raw PII ho sakta hai)
    write_log({"trace_id": trace_id, "endpoint": request.url.path,
               "status": "validation_error", "latency_ms": 0})
    return JSONResponse(status_code=422,
                        content={"trace_id": trace_id, "detail": "Invalid request body"})


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    trace_id = uuid.uuid4().hex
    start = time.perf_counter()
    safe_query, _ = mask_pii(req.query)      # wahi masking jo model ko dikhti hai
    try:
        result = run_agent(req.query, req.conversation_id)
    except Exception as e:
        write_log({"trace_id": trace_id, "endpoint": "/ask", "status": "error",
                   "error_type": type(e).__name__, "query": safe_query,
                   "latency_ms": _ms(start)})
        raise HTTPException(status_code=500, detail=f"Internal error. trace_id={trace_id}")
    write_log({"trace_id": trace_id, "endpoint": "/ask", "status": "ok",
               "conversation_id": result["conversation_id"],
               "query": result["masked_query"],       # sirf masked text disk par jata hai
               "route": result["route"], "grounded": result["grounded"],
               "pii_masked": result["guardrails"]["pii_masked"],
               "injection_detected": result["guardrails"]["injection_detected"],
               "latency_ms": _ms(start)})
    return {"trace_id": trace_id, **result}


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest):
    trace_id = uuid.uuid4().hex
    start = time.perf_counter()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    cols = {name: client.get_collection(name) for name in STRATEGIES}
    for col in cols.values():
        if col.get(ids=[f"{req.doc_id}::0"])["ids"]:
            write_log({"trace_id": trace_id, "endpoint": "/add-document", "status": "conflict",
                       "doc_id": req.doc_id, "latency_ms": _ms(start)})
            raise HTTPException(status_code=409, detail="doc_id already exists")
    model = get_model()
    added = {}
    for name, chunk_fn in STRATEGIES.items():
        chunks = chunk_fn(req.text)
        embs = model.encode(chunks, normalize_embeddings=True).tolist()
        cols[name].add(ids=[f"{req.doc_id}::{i}" for i in range(len(chunks))],
                       documents=chunks, embeddings=embs,
                       metadatas=[{"doc_id": req.doc_id, "chunk_index": i} for i in range(len(chunks))])
        added[name] = len(chunks)
    write_log({"trace_id": trace_id, "endpoint": "/add-document", "status": "ok",
               "doc_id": req.doc_id, "text_length": len(req.text),
               "chunks_added": added, "latency_ms": _ms(start)})
    return {"trace_id": trace_id, "doc_id": req.doc_id, "chunks_added": added}