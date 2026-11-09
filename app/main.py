"""RAG API: /ask (citation soho), /search, /ingest (runtime e notun document)"""
import json
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from rag.generate import generate
from rag.retrieval import KnowledgeBase

kb = KnowledgeBase()
app = FastAPI(title="Knowledge Base RAG API")


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=3, ge=1, le=10)


class Doc(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9_\-]{2,50}$")
    text: str = Field(min_length=20, max_length=20000)


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(kb.docs), "chunks": len(kb.chunks)}


@app.get("/eval")
def eval_report():
    p = Path("data/metrics.json")
    return json.loads(p.read_text()) if p.exists() else {"error": "run evaluate.py first"}


@app.get("/search")
def search(q: str = Query(min_length=2), k: int = Query(3, ge=1, le=10), method: str = "hybrid"):
    return {"query": q, "method": method, "results": kb.search(q, k, method)}


@app.post("/ask")
def ask(body: Ask):
    hits = kb.search(body.question, body.k)
    if not kb.is_relevant(hits):  # guard: context nai hole answer banabo na
        return {"question": body.question, "answer": "Ami knowledge base e ei bishoy e kono tottho pai ni.",
                "found": False, "mode": "guard", "sources": []}
    answer, mode = generate(body.question, hits)
    return {"question": body.question, "answer": answer, "found": True, "mode": mode,
            "sources": [{"ref": i + 1, "doc_id": h["doc_id"], "chunk_id": h["chunk_id"], "text": h["text"]}
                        for i, h in enumerate(hits)]}


@app.post("/ingest")
def ingest(doc: Doc):
    kb.add_document(doc.doc_id, doc.text)
    return {"status": "indexed", "documents": len(kb.docs), "chunks": len(kb.chunks)}
