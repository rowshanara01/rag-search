"""Retrieval quality mapa: Recall@1, Recall@3, MRR + out-of-domain refusal"""
import json

from rag.generate import extractive_answer
from rag.retrieval import KnowledgeBase

kb = KnowledgeBase()
data = json.load(open("data/eval.json"))


def first_hit_rank(hits, doc_id, ans):
    for i, h in enumerate(hits):
        if h["doc_id"] == doc_id and ans in h["text"].lower():
            return i + 1
    return None


report = {}
for method in ("bm25", "tfidf", "hybrid"):
    ranks = [first_hit_rank(kb.search(x["question"], 10, method), x["doc_id"], x["answer"]) for x in data["qa"]]
    n = len(ranks)
    report[method] = {
        "recall@1": sum(r == 1 for r in ranks) / n,
        "recall@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }
    print(f"{method:7s}", {k: round(v, 3) for k, v in report[method].items()})
    if method == "hybrid":
        failed = [x["question"] for x, r in zip(data["qa"], ranks) if r is None or r > 3]

hard = [first_hit_rank(kb.search(x["question"], 3), x["doc_id"], x["answer"]) for x in data["hard"]]
report["hard_paraphrase_recall@3"] = sum(r is not None for r in hard) / len(hard)
print("hard paraphrase recall@3 (keyword search er durbolota):", report["hard_paraphrase_recall@3"])
report["extractive_answer_accuracy"] = sum(
    x["answer"] in extractive_answer(x["question"], kb.search(x["question"], 3)).lower() for x in data["qa"]) / len(data["qa"])
print("extractive answer accuracy:", round(report["extractive_answer_accuracy"], 3))
answered = sum(kb.is_relevant(kb.search(x["question"], 3)) for x in data["qa"]) / len(data["qa"])
refused = sum(not kb.is_relevant(kb.search(q, 3)) for q in data["ood"]) / len(data["ood"])
report["guard"] = {"in_domain_answered": answered, "out_of_domain_refused": refused}
print("guard:", report["guard"])
print("hybrid misses (recall@3):", failed)
json.dump(report, open("data/metrics.json", "w"), indent=2)
