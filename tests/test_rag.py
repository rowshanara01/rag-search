import httpx
from fastapi.testclient import TestClient

from app.main import app
from rag import generate
from rag.chunking import chunk_document

client = TestClient(app)


def test_chunking_overlap():
    ch = chunk_document("d", "A one. B two. C three. D four.")
    assert len(ch) == 3 and "B two." in ch[0]["text"] and "B two." in ch[1]["text"]


def test_ask_returns_answer_with_citation(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/ask", json={"question": "What is the API rate limit?"}).json()
    assert r["found"] and "100 requests per minute" in r["answer"]
    assert r["sources"][0]["doc_id"] == "api_auth"


def test_out_of_domain_is_refused():
    r = client.post("/ask", json={"question": "Who won the cricket world cup?"}).json()
    assert r["found"] is False and r["sources"] == []


def test_ingest_makes_new_doc_searchable():
    client.post("/ingest", json={"doc_id": "parking", "text": "Employees can park free in basement level 2. Visitors must register at the front desk."})
    r = client.post("/ask", json={"question": "Where can employees park?"}).json()
    assert r["found"] and r["sources"][0]["doc_id"] == "parking"


def test_llm_path_uses_prompt_and_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    sent = {}

    class Fake:
        def raise_for_status(self): pass
        def json(self): return {"content": [{"text": "60 minutes [1]"}]}

    def fake_post(url, **kw):
        sent.update(kw["json"]); return Fake()

    monkeypatch.setattr(httpx, "post", fake_post)
    r = client.post("/ask", json={"question": "How long before an API token expires?"}).json()
    assert r["mode"] == "llm" and "Context:" in sent["messages"][0]["content"]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    r2 = client.post("/ask", json={"question": "How long before an API token expires?"}).json()
    assert r2["mode"] == "extractive (llm failed)" and "60 minutes" in r2["answer"]


def test_validation():
    assert client.post("/ask", json={"question": "hi", "k": 3}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422
