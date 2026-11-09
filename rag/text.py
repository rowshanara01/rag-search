import re

STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and", "or", "do", "does", "i",
        "can", "how", "what", "which", "when", "my", "me", "it", "be", "by", "at", "with", "that", "this",
        "we", "you", "from", "get", "per", "will", "has", "have", "must", "may"}


def tokenize(text):
    """lowercase -> word -> stopword bad -> choto stemming (days -> day)"""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out
