"""
retrieval_engine.py
-------------------
Advanced RAG Retrieval Engine combining query rewriting, hybrid search
(vector + BM25), Reciprocal Rank Fusion, and information gap analysis.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from reranker import SiliconFlowReranker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with associated metadata."""

    document: Document
    vector_rank: int = 0
    bm25_rank: int = 0
    rrf_score: float = 0.0
    confidence: float = 0.0

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata


@dataclass
class RetrievalResult:
    """Encapsulates the full output of a retrieval pipeline run."""

    query: str
    rewritten_query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    gap_analysis: str | None = None

    @property
    def documents(self) -> list[Document]:
        return [c.document for c in self.chunks]

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AdvancedRetrievalEngine:
    """
    A production-grade retrieval engine for RAG pipelines.

    Pipeline stages:
        1. Query Rewriting   — strips filler, extracts high-intent terms.
        2. Hybrid Search     — fuses vector similarity + BM25 keyword search
                               via Reciprocal Rank Fusion (RRF).
        3. Confidence Scoring — assigns a normalized confidence to each chunk.
        4. Gap Analysis      — if retrieval quality is low, generates a
                               structured "Missing Information Checklist".

    Args:
        openrouter_api_key: OpenRouter API key. Falls back to the
            ``OPENROUTER_API_KEY`` environment variable if not provided.
        rewrite_model: Model slug used for query rewriting and gap analysis.
        rrf_k: Smoothing constant for Reciprocal Rank Fusion (default 60).
        request_timeout: HTTP timeout in seconds for LLM calls.
    """

    _OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        openrouter_api_key: str | None = None,
        rewrite_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        rrf_k: int = 20,
        request_timeout: int = 15,
        reranker: SiliconFlowReranker = SiliconFlowReranker()
    ) -> None:
        self._api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise EnvironmentError(
                "OpenRouter API key not found. Pass it explicitly or set "
                "the OPENROUTER_API_KEY environment variable."
            )

        self.rewrite_model = rewrite_model
        self.rrf_k = rrf_k
        self.request_timeout = request_timeout
        self.reranker = reranker

        logger.info(
            "AdvancedRetrievalEngine initialized | model=%s | rrf_k=%d",
            self.rewrite_model,
            self.rrf_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        vector_store,
        k: int = 10,
        confidence_threshold: float = 0.15,
        rewrite: bool = True,
        use_reranker: bool = True
    ) -> RetrievalResult:
        """
        Run the full retrieval pipeline for a user query.

        Args:
            query: Raw user query string.
            vector_store: A LangChain-compatible vector store with
                ``similarity_search(query, k)`` support.
            k: Number of top chunks to return.
            confidence_threshold: Chunks below this score trigger gap analysis.

        Returns:
            A :class:`RetrievalResult` containing ranked chunks and,
            if relevant, a gap analysis string.
        """
        logger.info("Starting retrieval | query=%r", query)

        rewritten = (
            self.rewrite_query(query) if rewrite else query
        )
        logger.info("Rewritten query: %r", rewritten)

        chunks = self.hybrid_search(rewritten, vector_store, k=k, use_reranker=use_reranker)

        result = RetrievalResult(query=query, rewritten_query=rewritten, chunks=chunks)

        if result.is_empty or self._low_confidence(chunks, confidence_threshold):
            logger.warning(
                "Low-confidence retrieval detected — running gap analysis."
            )
            raw_docs = [{"content": c.content} for c in chunks]
            result.gap_analysis = self.analyze_information_gap(query, raw_docs)

        logger.info(
            "Retrieval complete | chunks=%d | gap_analysis=%s",
            len(chunks),
            "yes" if result.gap_analysis else "no",
        )
        return result

    # ------------------------------------------------------------------
    # Stage 1 — Query Rewriting
    # ------------------------------------------------------------------

    def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite a raw user query into a focused, high-intent search phrase.

        Uses a fast LLM to strip conversational filler and extract
        semantically rich keywords suited for both vector and BM25 search.

        Args:
            original_query: The raw string typed by the user.

        Returns:
            An optimized search phrase, or the original query on failure.
        """
        prompt = (
            "You are an expert search query optimizer.\n"
            "Take this raw user query and rewrite it into a concise, high-intent search phrase.\n"
            "Keep technical terms, specific nouns, and key action verbs.\n"
            "Remove greetings, politeness, and excessive filler words.\n"
            "Output ONLY the optimized phrase. No quotes, no preamble.\n\n"
            f'Raw Query: "{original_query}"\n'
            "Optimized Search Phrase:"
        )

        response = self._call_llm(prompt, temperature=0.1)
        if response:
            return response.replace('"', "").replace("'", "").strip()

        logger.warning("Query rewriting failed; falling back to original query.")
        return original_query

    # ------------------------------------------------------------------
    # Stage 2 — Hybrid Search (Vector + BM25 via RRF)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        vector_store,
        k: int = 5,
        use_reranker = False
    ) -> list[RetrievedChunk]:
        """
        Perform hybrid retrieval by fusing vector search and BM25 rankings
        using Reciprocal Rank Fusion (RRF).

        Args:
            query: The (optionally rewritten) search query.
            vector_store: LangChain vector store instance.
            k: Number of final chunks to return.

        Returns:
            A list of :class:`RetrievedChunk` objects sorted by RRF score.
        """

        def tokenize(text: str) -> list[str]:
            return re.findall(r'\w+', text.lower())
        
        fetch_k = 30 # Increased over-fetch to give BM25 a better candidate pool

        # --- Vector search ---
        vector_docs: list[Document] = vector_store.similarity_search(query, k=fetch_k)
        if not vector_docs:
            logger.warning("Vector search returned no results.")
            return []

        # --- BM25 keyword search over vector candidates ---
        tokenized_corpus = [tokenize(doc.page_content) for doc in vector_docs]

        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = tokenize(query)
        bm25_scores: list[float] = bm25.get_scores(tokenized_query).tolist()

        # Build BM25 rank order (index → rank position)
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked_indices)}

        # --- Reciprocal Rank Fusion ---
        rrf_scores: dict[int, float] = {}
        for vector_rank, doc_idx in enumerate(range(len(vector_docs))):
            bm25_rank = bm25_rank_map.get(doc_idx, len(vector_docs))
            rrf_scores[doc_idx] = (
                1.0 / (self.rrf_k + vector_rank + 1)
                + 1.0 / (self.rrf_k + bm25_rank + 1)
            )

        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

        # --- Deduplicate and assemble results ---
        seen: set[str] = set()
        candidate_chunks: list[RetrievedChunk] = []

        candidate_limit = 10

        for rank, idx in enumerate(sorted_indices[:candidate_limit]):
            doc = vector_docs[idx]
            if doc.page_content in seen:
                continue
            seen.add(doc.page_content)

            rrf = rrf_scores[idx]
            candidate_chunks.append(
                RetrievedChunk(
                    document=doc,
                    vector_rank=idx,
                    bm25_rank=bm25_rank_map.get(idx, len(vector_docs)),
                    rrf_score=rrf,
                    confidence=self._normalize_confidence(rrf, self.rrf_k),
                )
            )

        if use_reranker:
        
            expanded_chunks = []
            seen_parents = set()

            for chunk in candidate_chunks:
                parent_id = chunk.metadata.get("parent_id")
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                parent_doc = Document(
                    page_content=chunk.document.metadata.get("parent_text"),
                    metadata=chunk.document.metadata
                )

                expanded_chunks.append(
                    RetrievedChunk(
                        document=parent_doc,
                        vector_rank=chunk.vector_rank,
                        bm25_rank=chunk.bm25_rank,
                        rrf_score=chunk.rrf_score,
                        confidence=chunk.confidence
                    )
                )
            
            candidate_chunks = self.reranker.rerank(query, expanded_chunks)

            final_chunks = []

            for chunk, score in candidate_chunks:
                chunk.confidence = score
                final_chunks.append(chunk)

            logger.debug("Hybrid search returned %d unique chunks.", len(candidate_chunks))

            print("\nTOP CHUNKS")
            for chunk in final_chunks:
                print(chunk.metadata.get("chunk_id"), chunk.confidence)
            return final_chunks[:k]

        logger.debug("Hybrid search returned %d unique chunks.", len(candidate_chunks))

        print("\nTOP CHUNKS")
        for chunk in candidate_chunks:
            print(chunk.metadata.get("chunk_id"), chunk.confidence)
        return candidate_chunks[:k]
    
    # ------------------------------------------------------------------
    # Stage 3 — Gap Analysis
    # ------------------------------------------------------------------

    def analyze_information_gap(
        self,
        query: str,
        attempted_docs: list[dict[str, Any]],
    ) -> str:
        """
        Generate a structured gap analysis when retrieval quality is low.

        Identifies what specific facts, metrics, or context are absent from
        the knowledge base and returns a bulleted "Missing Information
        Checklist" for the user.

        Args:
            query: The original user query that triggered this analysis.
            attempted_docs: List of dicts with a ``"content"`` key — the
                closest-matching chunks that were found.

        Returns:
            A formatted string describing the information gap, or a
            fallback message on failure.
        """
        snippets = "\n---\n".join(
            d["content"] for d in attempted_docs if d.get("content")
        )

        prompt = (
            "You are an Information Gap Auditor for a RAG knowledge base.\n\n"
            f'The user asked: "{query}"\n\n'
            "The following are the closest-matching snippets retrieved from the "
            "knowledge base — but none are a high-confidence match:\n\n"
            f"{snippets}\n\n"
            "Your task:\n"
            "1. Identify exactly which facts, metrics, versions, dates, or context "
            "are missing to answer this query reliably.\n"
            "2. Politely inform the user the knowledge base lacks sufficient data.\n"
            "3. Provide a concise, bulleted **Missing Information Checklist** "
            "telling them what file or data they should upload next.\n"
            "Keep the tone helpful and professional."
        )

        response = self._call_llm(prompt, temperature=0.3)
        if response:
            return response

        return (
            "The knowledge base does not contain sufficient context to answer "
            "your question. Please upload a more relevant document and try again."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str | None:
        """
        Send a single-turn completion request to the configured LLM.

        Args:
            prompt: The full user-facing prompt string.
            temperature: Sampling temperature for the LLM.

        Returns:
            The model's text response, or ``None`` on error.
        """
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.rewrite_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        try:
            response = requests.post(
                self._OPENROUTER_URL,
                json=payload,
                headers=headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content")

            if not content:
                return None

            return content.strip()

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out after %ds.", self.request_timeout)
        except requests.exceptions.HTTPError as exc:
            logger.error("LLM HTTP error: %s", exc)
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected LLM response format: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error calling LLM: %s", exc)

        return None

    @staticmethod
    def _normalize_confidence(rrf_score: float, k: int) -> float:
        """
        Map an RRF score to a [0, 1] confidence value.

        Uses the theoretical maximum RRF score for rank-1 across both
        retrievers as the normalization ceiling.

        Args:
            rrf_score: Raw RRF fusion score.
            k: The RRF smoothing constant used during fusion.

        Returns:
            A float in [0, 1] representing retrieval confidence.
        """
        max_possible = 2.0 / (k + 1)  # Both retrievers ranked this doc #1
        return min(rrf_score / max_possible, 1.0) if max_possible > 0 else 0.0

    @staticmethod
    def _low_confidence(chunks: list[RetrievedChunk], threshold: float) -> bool:
        """
        Return True if the top chunk's confidence falls below ``threshold``.

        Args:
            chunks: Ranked list of retrieved chunks.
            threshold: Minimum acceptable confidence score.

        Returns:
            ``True`` if confidence is insufficient, ``False`` otherwise.
        """
        if not chunks:
            return True
        return chunks[0].confidence < threshold