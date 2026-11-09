# RAG Search (Company Knowledge Base Q&A)

RAG = Retrieval-Augmented Generation. Ekhon prai shob company te ei system ache (HR bot, support bot, docs search).

## Chalanor niyom
pip install -r requirements.txt
python evaluate.py                 # retrieval quality mapa (Recall@K, MRR)
pytest -q
uvicorn app.main:app --reload      # localhost:8000/docs
  POST /ask     {"question": "What is the API rate limit?"}
  GET  /search?q=refund&method=bm25
  POST /ingest  (notun document runtime e dao)
LLM answer chaile: export ANTHROPIC_API_KEY=...   (na dile extractive mode, key charao-o cholbe)

## Pipeline
Document -> Chunking (sliding window) -> Index (BM25 + TF-IDF) -> Hybrid retrieval (RRF)
-> Relevance guard (kom score hole "jani na") -> LLM (context theke) -> Answer + citation

## Company te ja ja use hoy (ei project e ache)
- Chunking with overlap, BM25 nijer hate lekha, hybrid search (RRF)
- Retrieval evaluation: Recall@K, MRR (LLM er age retrieval mapo)
- Hallucination guard: context nai hole answer banay na
- Citation ([1], [2]) with source chunk
- LLM fail korle fallback, runtime ingestion, validation, tests

## Shimabodhdhota (evaluate.py te dekhbe)
"hard paraphrase" question (jemon "job from my house" vs "work remotely") keyword search e fail kore,
karon shobdo mele na. Eta-i keno embeddings dorkar.

## Challenge
1. sentence-transformers diye dense embedding retriever add koro, hybrid e jur (BM25 + dense)
2. Vector DB (FAISS/Chroma/pgvector) use koro
3. Cross-encoder reranker add koro
4. PDF/Word document ingest koro
5. Bangla document support (Bangla tokenizer)
6. LLM answer er faithfulness check (answer ta context e ache kina)
