"""demo_api.py - Tasks 11-12 ka transcript (FastAPI TestClient, server ki zaroorat nahi)."""
import os
import chromadb
from fastapi.testclient import TestClient

import api
from rag import CHROMA_PATH

if os.path.exists(api.LOG_FILE):
    os.remove(api.LOG_FILE)

client = TestClient(api.app)
RAW_PHONE = "9876543210"

print("=== GET /health ===")
print(client.get("/health").json())

print("\n=== POST /ask (policy question) ===")
r = client.post("/ask", json={"query": "How many days do I have to return a beauty product?",
                              "conversation_id": "api1"})
print(r.status_code, r.json()["route"], "| trace_id:", r.json()["trace_id"])
print("answer:", r.json()["answer"])

print("\n=== POST /ask (PII wala query) ===")
r = client.post("/ask", json={"query": f"My phone is {RAW_PHONE} and card ending 4321, where is my order NYK-020?",
                              "conversation_id": "api1"})
print(r.status_code, r.json()["route"], "| masked_query:", r.json()["masked_query"])

print("\n=== POST /add-document ===")
r = client.post("/add-document", json={
    "doc_id": "gift_wrap", "topic": "Gift wrapping",
    "text": "Nykaa offers gift wrapping for 50 rupees per item. "
            "Gift wrap can be added at checkout before payment. "
            "Wrapped items can still be returned within the normal return window."})
print(r.status_code, r.json())

print("\n=== POST /ask (nayi document par sawaal) ===")
r = client.post("/ask", json={"query": "How much does gift wrapping cost?", "conversation_id": "api2"})
print(r.status_code, r.json()["route"], "| sources:", r.json()["sources"])
print("answer:", r.json()["answer"])

print("\n=== POST /ask (invalid body -> 422) ===")
r = client.post("/ask", json={"query": ""})
print(r.status_code, r.json())

print("\n=== logs/requests.jsonl (har request = ek line) ===")
with open(api.LOG_FILE) as f:
    content = f.read()
print(content)

assert RAW_PHONE not in content and "card ending 4321" not in content
print("PII check: raw phone number aur raw card digits log mein nahi hain -> PASSED")
print("Total log lines:", len(content.strip().splitlines()))

# demo wali document ko hata do, taki KB saaf rahe
cl = chromadb.PersistentClient(path=CHROMA_PATH)
for name in ("kb_fixed", "kb_sentence"):
    cl.get_collection(name).delete(where={"doc_id": "gift_wrap"})
print("Cleanup: demo document removed")