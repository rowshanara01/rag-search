"""Answer generation: ANTHROPIC_API_KEY thakle LLM (context theke), na thakle extractive (sorasori sentence)"""
import os

import httpx

from .chunking import split_sentences
from .text import tokenize

SYSTEM = ("Answer ONLY using the numbered context. Cite sources like [1]. "
          "If the context does not contain the answer, say you don't know.")


def build_prompt(question, hits):
    ctx = "\n".join(f"[{i + 1}] ({h['doc_id']}) {h['text']}" for i, h in enumerate(hits))
    return f"Context:\n{ctx}\n\nQuestion: {question}"


QUANTITY = ("how long", "how many", "how much", "how fast", "how quickly", "when", "cost", "limit")


def extractive_answer(question, hits):
    q = set(tokenize(question))
    wants_number = any(w in question.lower() for w in QUANTITY)
    best, best_score = "", -1
    for h in hits:
        for s in split_sentences(h["text"]):
            sc = len(q & set(tokenize(s)))
            if wants_number and any(ch.isdigit() for ch in s):
                sc += 0.5  # "koto/koto din" hole number thaka sentence ke prai
            if sc > best_score:
                best, best_score = s, sc
    return best


def llm_answer(question, hits):
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), "max_tokens": 300,
              "system": SYSTEM, "messages": [{"role": "user", "content": build_prompt(question, hits)}]},
        timeout=30)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def generate(question, hits):
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return llm_answer(question, hits), "llm"
        except Exception:  # API fail hole system bondho hobe na
            return extractive_answer(question, hits), "extractive (llm failed)"
    return extractive_answer(question, hits), "extractive"
