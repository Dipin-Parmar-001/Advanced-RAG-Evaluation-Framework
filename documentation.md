========================================
SECTION 1: PROJECT OVERVIEW
========================================

Project Title: Collaborative Agentic RAG Pipeline
Subtitle: Hybrid Search · Reranking · Parent-Child Chunking
Version: v1.0
Author Name: Dipen Parmar
GitHub URL: [optional]
Date: June 2026

Short Description (2–4 lines):
A production-grade Retrieval-Augmented Generation (RAG) pipeline designed to index and query PDF documents. It leverages advanced search techniques, reciprocal rank fusion (RRF), cross-encoder reranking, LLM-based relevance grading, and a strict hallucination guardrail to guarantee clean, factual synthesis.

========================================
SECTION 2: TECH STACK
========================================

List each component with its role:
- FastAPI: API and web backend server layer
- ChromaDB: Vector database and persistent storage for embeddings
- LangChain Core: Document wrappers and text splitting utilities
- OpenRouter API: Embedding generation (Nemotron 1B) and query rewrite/grading models
- SiliconFlow API: Cross-encoder reranking layer (Qwen3-Reranker-8B)
- Rank-BM25: Lexical term-frequency BM25 search over vector candidates
- PyMuPDF (fitz): High-speed PDF parsing and page-by-page text extraction
- TailwindCSS & Vanilla JS: Minimalist web chat interface

========================================
SECTION 3: SYSTEM ARCHITECTURE
========================================

Describe the flow in steps (I'll turn this into a visual pipeline diagram):
1. Ingestion & Text Splitting: PyMuPDF extracts text page-by-page from PDFs. It is cleaned, split into 1200-character parent chunks, and further split into 200-character child chunks containing parent metadata.
2. Ingestion Storage: OpenRouter embeds the child chunks which are indexed into ChromaDB.
3. Query Rewriting: Raw user queries are rewritten by a fast reasoning model to strip conversational filler and extract high-intent search terms.
4. Candidate Retrieval (Hybrid Search): Vector search retrieves the top 30 child chunks. BM25 is then run over these same candidate chunks.
5. Reciprocal Rank Fusion (RRF): Vector and lexical search rankings are fused using Reciprocal Rank Fusion (RRF).
6. Context Expansion: The top fused child chunks are mapped back to their 1200-character parent documents.
7. Cross-Encoder Reranking: SiliconFlow's Qwen3-Reranker-8B evaluates the parent documents and query to score and select the top 6 chunks.
8. Confidence Auditing & Gap Analysis: If retrieve confidence is below threshold, an Information Gap Auditor LLM constructs a "Missing Information Checklist" for the user.
9. Agentic Relevance Grading: A relevance grading agent checks each chunk and filters out any irrelevant noise.
10. Synthesis & Hallucination Check: An LLM synthesizes the answer from the filtered context. A Fact-Checking Specialist agent compares the answer to the source text, blocking any hallucinations.

========================================
SECTION 4: CORE FEATURES (one per feature)
========================================

Feature 1:
  Name: Parent-Child Chunking
  Description: Chunks text into tiny child passages (200 chars) for high-accuracy vector matching, but expands them to larger parent segments (1200 chars) before generation to ensure rich context.
  Key detail / config: Parent chunk size=1200, overlap=200; Child chunk size=200, overlap=30.

Feature 2:
  Name: Hybrid Search & RRF
  Description: Combines dense vector similarity search with lexical BM25 term matching over candidate spaces to capture both semantic meaning and exact keyword matches.
  Key detail / config: Reciprocal Rank Fusion constant k=20; initial retrieval over-fetch limit=30.

Feature 3:
  Name: Cross-Encoder Reranking
  Description: Feeds retrieved parent context candidates and the user query to a cross-encoder model to determine precise, multi-turn attention score relevance.
  Key detail / config: Qwen/Qwen3-Reranker-8B model via SiliconFlow API, returning top-k=6.

Feature 4:
  Name: Information Gap Analysis
  Description: Automatically audits queries that yield low retrieval confidence or failed relevance checks and returns a structured checklist explaining what documents are missing.
  Key detail / config: Confidence threshold=0.10; uses nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free.

Feature 5:
  Name: Double-Agent Security Guardrail
  Description: Employs separate grading agents to filter out irrelevant context chunks post-retrieval and fact-check the final synthesized answer against the source context before returning it.
  Key detail: Grade relevance (YES/NO) and Hallucination grading (PASSED/FAILED).

Feature 6:
  Name: Automated Evaluation Suite
  Description: A complete test harness implementing standard retrieval evaluation metrics to compare baseline pipelines against reranked configurations.
  Key detail: Computes P@k, Recall@k, MRR, HitRate, and NDCG.

========================================
SECTION 5: EVALUATION METRICS
========================================

List each metric with its value and what it means:
- Precision@1 (Baseline): 0.60 — Ratio of top-1 results that are relevant without cross-encoder reranking.
- Precision@1 (Reranker): 0.80 — Reranker successfully boosts top-1 relevance by 33.3%.
- Recall@5 (Baseline): 0.63 — Capture rate of relevant chunks in top-5 results before reranking.
- Recall@5 (Reranker): 0.75 — Reranker captures 75% of relevant documents in the top-5 candidates.
- MRR (Baseline): 0.66 — Mean Reciprocal Rank (reciprocal rank of first correct document).
- MRR (Reranker): 0.80 — High score indicating correct results are pushed to the top position.
- NDCG@5 (Baseline): 0.59 — Normalized Discounted Cumulative Gain representing overall ordering quality.
- NDCG@5 (Reranker): 0.76 — A 28.8% improvement, signifying a highly optimized rank order.

========================================
SECTION 6: KEY MODULES / CODE COMPONENTS
========================================

List important files/modules with a 1–2 line description:
- main.py: FastAPI server exposing endpoints for static document sync, file uploads, and pipeline queries.
- retrieval_engine.py: Core class orchestrating query rewriting, hybrid retrieval, RRF scoring, parent expansion, and reranking.
- agent_engine.py: Houses the grading logic for document relevance checks and hallucination guardrails.
- embeddings.py: Handles OpenRouter (Nemotron) and Google (Gemini) API integrations for document embeddings.
- reranker.py: Interfaces with the SiliconFlow API to execute cross-encoder reranking.
- data_processing.py: Cleans raw text inputs and parses PDFs page-by-page using PyMuPDF.
- evaluation/metrics.py: Contains math implementations for precision, recall, MRR, HitRate, and NDCG.
- evaluation/evaluate.py: Script running benchmarks across datasets and saving CSV reports.

========================================
SECTION 7: RESULTS / OUTCOMES (optional)
========================================

Reranker integration results in a major performance leap:
- MRR improved from 0.66 to 0.80 (+21.2%)
- NDCG@5 improved from 0.59 to 0.76 (+28.8%)
- Precision@1 boosted from 0.60 to 0.80 (+33.3%)
These numbers prove that cross-encoder reranking over parent contexts provides highly accurate results.

========================================
SECTION 8: CHALLENGES & SOLUTIONS (optional)
========================================

Challenge 1: OpenRouter API Rate Limits during document ingestion.
Solution: Implemented batched document insertion in main.py (chunks grouped into blocks of 80, with a 2.0s delay between batches).

Challenge 2: Precision/Recall trade-off with standard chunking.
Solution: Implemented Parent-Child chunking, indexing tiny child text chunks but loading parent pages prior to rerank scoring and LLM synthesis.

========================================
SECTION 9: FUTURE IMPROVEMENTS (optional)
========================================

- Add support for indexing Word documents (.docx) and spreadsheets (.xlsx).
- Incorporate a local embeddings model (like SentenceTransformers) for local offline fallbacks.
- Develop dynamic metadata filters based on extraction fields (e.g., date ranges, authors).

========================================
SECTION 10: VISUAL PREFERENCES (for me)
========================================

Color theme: Dark slate paneling (#0f172a, #1e293b) with vibrant neon-blue and warm orange accents.
Style: Bold technical, modern startup dashboard.
Include diagrams? Yes.
