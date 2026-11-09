"""Retrieval: BM25 (keyword) + TF-IDF cosine + Hybrid (Reciprocal Rank Fusion)"""
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import chunk_document
from .text import tokenize

MIN_COSINE = 0.12  # er niche hole "ami jani na" bolbo (hallucination guard)


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.len = np.array([len(d) for d in docs_tokens], float)
        self.avg = self.len.mean() if len(self.len) else 1.0
        n = len(docs_tokens)
        df = Counter(t for d in docs_tokens for t in set(d))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q_tokens):
        s = np.zeros(len(self.tf))
        for t in set(q_tokens):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class KnowledgeBase:
    def __init__(self, docs_dir="data/docs"):
        self.docs = {}
        if docs_dir and Path(docs_dir).exists():
            for p in sorted(Path(docs_dir).glob("*.txt")):
                self.docs[p.stem] = p.read_text().strip()
        self._build()

    def add_document(self, doc_id, text):
        self.docs[doc_id] = text.strip()
        self._build()

    def _build(self):
        self.chunks = [c for d, t in self.docs.items() for c in chunk_document(d, t)]
        texts = [c["text"] for c in self.chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2),
                                     sublinear_tf=True)
        self.matrix = self.tfidf.fit_transform(texts)  # row gulo L2-normalized, tai dot = cosine

    def search(self, query, k=3, method="hybrid"):
        if not self.chunks:
            return []
        bm = self.bm25.scores(tokenize(query))
        cos = (self.matrix @ self.tfidf.transform([query]).T).toarray().ravel()
        if method == "bm25":
            score = bm
        elif method == "tfidf":
            score = cos
        else:  # Reciprocal Rank Fusion: duto ranking er rank jure dei
            n = len(bm)
            r_bm, r_cos = np.empty(n), np.empty(n)
            r_bm[np.argsort(-bm)] = np.arange(n)
            r_cos[np.argsort(-cos)] = np.arange(n)
            score = np.where((bm > 0) | (cos > 0), 1 / (60 + r_bm) + 1 / (60 + r_cos), 0.0)
        order = np.argsort(-score)[:k]
        return [{**self.chunks[i], "score": round(float(score[i]), 4), "cosine": round(float(cos[i]), 4)}
                for i in order if score[i] > 0]

    @staticmethod
    def is_relevant(hits):
        return bool(hits) and hits[0]["cosine"] >= MIN_COSINE
