import re


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def chunk_document(doc_id, text, size=2, overlap=1):
    """Sliding window: 2 sentence er chunk, 1 sentence overlap (context kete na jay)"""
    sents = split_sentences(text)
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, max(1, len(sents) - size + 1), step):
        chunks.append({"doc_id": doc_id, "chunk_id": f"{doc_id}#{len(chunks)}",
                       "text": " ".join(sents[i:i + size])})
    return chunks
