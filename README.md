#### RAG Search (Company Knowledge Base Q&A)

#### What Is RAG?

RAG stands for Retrieval-Augmented Generation.

It is a system that allows an AI application to answer questions using a company's own documents, internal knowledge base, policies, and technical documentation.

Many companies use RAG-based systems for:

- HR Chatbots.
- Customer Support Bots.
- Internal Documentation Search.
- Company Knowledge Base Q&A.
- Technical Documentation Assistants.

Instead of relying only on an LLM's pre-trained knowledge, RAG retrieves relevant information from company documents and uses that information to generate answers.

## How to Run the Project

Install the dependencies:

pip install -r requirements.txt

Evaluate retrieval quality:

python evaluate.py

This measures how effectively the system retrieves relevant documents using Recall@K and MRR.

Run automated tests:

pytest -q

Start the API:

uvicorn app.main:app --reload

Open the API documentation:

http://localhost:8000/docs

#### Available API Endpoints

**1. Ask a Question**

POST /ask

Request:

{
  "question": "What is the API rate limit?"
}

The system retrieves relevant information and generates an answer using the available document context.

**2. Search Documents**

GET /search?q=refund&method=bm25

Search the knowledge base using BM25 retrieval.

**3. Ingest New Documents**

POST /ingest

Upload new documents at runtime so the knowledge base can be updated without rebuilding the entire application manually.

### Enable LLM-Based Answers

Set your Anthropic API key:

export ANTHROPIC_API_KEY=...

If no API key is provided, the system automatically uses extractive mode.

This means the application can still retrieve and return relevant information without requiring an external LLM.

## RAG Pipeline: How the System Works

Document → Chunking (Sliding Window) → Indexing (BM25 + TF-IDF) → Hybrid Retrieval (RRF) → Relevance Guard → LLM Generation → Answer + Citations

### Step-by-Step Explanation

**1. Document Processing**

Company documents are collected and prepared for indexing.

**2. Chunking**

Large documents are divided into smaller text segments called chunks.

A sliding-window strategy with overlap is used to preserve context between neighboring chunks.

**3. Indexing**

The system builds search indexes using:

- BM25.
- TF-IDF.

These indexes allow the system to find relevant information efficiently.

**4. Hybrid Retrieval**

Combine multiple retrieval methods using Reciprocal Rank Fusion (RRF).

This helps improve retrieval quality by combining results from different search techniques.

**5. Relevance Guard**

Check whether the retrieved documents contain sufficiently relevant information.

If the relevance score is too low, the system responds:

"I don't know."

This prevents the system from generating answers without sufficient supporting context.

**6. LLM-Based Answer Generation**

Send the retrieved document chunks to the LLM as context.

The LLM generates an answer based on the provided information rather than relying entirely on its general knowledge.

**7. Citations**

Include source references such as [1] and [2] in the generated answer.

These references identify which document chunks support the answer.

## What Real-World Companies Use in RAG Systems

### 1. Chunking with Overlap

Divide documents into smaller chunks while maintaining overlapping text between neighboring chunks.

Why?

If a paragraph is split in the middle of an important explanation, overlapping chunks help preserve the surrounding context.

### 2. BM25 Search

Implement BM25 retrieval to find relevant documents based on keyword matching.

BM25 considers factors such as:

- Term frequency.
- Inverse document frequency.
- Document length.

This makes it useful for searching technical terms, product names, and exact keywords.

### 3. Hybrid Search Using RRF

Combine BM25 and TF-IDF search results using Reciprocal Rank Fusion (RRF).

Instead of depending on a single retrieval method, hybrid search combines multiple ranking signals to improve document retrieval.

### 4. Retrieval Evaluation

Evaluate retrieval quality before evaluating the LLM's answer generation.

Use:

- Recall@K: Measures whether relevant documents appear within the top K retrieved results.
- MRR (Mean Reciprocal Rank): Measures how highly the first relevant result appears in the search results.

This helps identify retrieval problems independently of LLM performance.

### 5. Hallucination Prevention

Implement a relevance guard.

If the knowledge base does not contain sufficient information, the system should avoid inventing an answer.

Instead, it should clearly state that the available documents do not provide enough information.

### 6. Source Citations

Generate answers with citations such as:

[1], [2]

Each citation points to the source chunk used to generate the answer.

This makes it easier for users to verify the information.

### 7. LLM Failure Fallback

If the external LLM fails or becomes unavailable, the application should fall back to extractive mode.

This helps maintain basic search functionality without depending entirely on an external AI service.

### 8. Runtime Document Ingestion

Allow new documents to be added while the application is running.

The system should validate incoming documents and update the search indexes accordingly.

### 9. Validation and Automated Testing

Test important components, including:

- Document ingestion.
- Chunking.
- Retrieval.
- Search ranking.
- Relevance checking.
- API responses.
- LLM fallback behavior.

This helps ensure the application remains reliable as new features are added.

## Project Limitations

The retrieval evaluation in evaluate.py reveals an important limitation.

### Hard Paraphrased Questions

For example:

Document contains:

"Employees are allowed to work remotely."

User asks:

"Can I do my job from my house?"

Keyword-based search may fail because the words used in the question are different from the words in the document.

Even though both sentences express nearly the same meaning, BM25 and TF-IDF may not recognize the semantic relationship.

This is one of the main reasons modern RAG systems use dense embeddings.

## Advanced Challenges

### 1. Add Dense Embedding Retrieval

Use Sentence Transformers to generate semantic embeddings for documents and queries.

This allows the system to understand semantic similarity rather than relying only on exact keyword matches.

For example:

"Work remotely" and "Do my job from home" can be recognized as semantically similar.

Combine dense retrieval with BM25 to build a stronger hybrid search system.

### 2. Integrate a Vector Database

Use one of the following vector databases:

- FAISS.
- Chroma.
- pgvector.

Store document embeddings and retrieve semantically similar chunks efficiently.

### 3. Add a Cross-Encoder Reranker

After retrieving the initial candidate documents, use a cross-encoder model to rerank them based on their relevance to the user's question.

This can improve the quality of the final context passed to the LLM.

### 4. Support PDF and Word Documents

Extend document ingestion to support:

- PDF files.
- Microsoft Word (.docx) files.

Extract their text, process the content, generate chunks, and add them to the knowledge base.

### 5. Add Bangla Document Support

Extend the retrieval system to support Bangla documents and queries.

Implement appropriate Bangla tokenization and text preprocessing.

The goal is to allow users to ask questions in Bangla and retrieve relevant information from Bangla documents.

### 6. Implement LLM Answer Faithfulness Checking

Add a faithfulness evaluation mechanism.

Check whether the generated answer is actually supported by the retrieved document context.

Identify unsupported claims and prevent answers that contain information not found in the source documents.

---

## Project Goal

Build an end-to-end, production-oriented Retrieval-Augmented Generation (RAG) system that allows users to search company documents and receive accurate, context-aware answers with source citations.

The goal is to develop a reliable company knowledge assistant that combines document processing, hybrid search, semantic retrieval, vector databases, reranking, LLM integration, hallucination prevention, and automated evaluation.

The system should not simply generate answers. It should retrieve the right information, understand the context, provide evidence, and acknowledge when it does not know the answer.
