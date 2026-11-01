import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Ensure modern SQLite3 on Linux / Streamlit Community Cloud for ChromaDB
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from langchain_chroma import Chroma

try:
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import (
        ContextualCompressionRetriever,
        EnsembleRetriever,
    )

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from src.config import settings

logger = logging.getLogger(__name__)


def compute_adaptive_weights(
    query: str, default_dense: float = 0.5, default_sparse: float = 0.5
) -> tuple[float, float]:
    """Analyzes query characteristics to dynamically balance dense semantic vs. sparse lexical search.

    - Queries with exact numbers, currency, dates, or tabular financial metrics
      tilt toward BM25 sparse lexical search (e.g. 0.3 dense / 0.7 sparse).
    - Conceptual, explanatory, or thematic questions tilt toward Chroma dense vector search
      (e.g. 0.7 dense / 0.3 sparse).
    - General queries retain default configured weights.
    """
    if not query or not str(query).strip():
        return default_dense, default_sparse

    q_lower = query.lower()

    # Numeric, currency, date, and exact metric indicators
    has_numbers = bool(re.search(r"\b\d+(\.\d+)?%?\b", query))
    has_currency = any(sym in query for sym in ["$", "€", "£", "¥", "usd", "eur", "inr"])
    has_quarters = bool(re.search(r"\bq[1-4]\b|\bfy\d{2,4}\b", q_lower))
    has_exact_metrics = bool(
        re.search(
            r"\b(net income|revenue|ebitda|margin|cash|expenses|balance|table|page|row|total|cost|price|shares)\b",
            q_lower,
        )
    )

    # Conceptual, summary, and strategic indicators
    has_conceptual = bool(
        re.search(
            r"\b(why|how|explain|describe|summarize|summary|overview|strategy|outlook|risks|headwinds|implications|analysis)\b",
            q_lower,
        )
    )

    if (has_numbers or has_currency or has_quarters or has_exact_metrics) and not has_conceptual:
        return 0.3, 0.7
    elif has_conceptual and not (has_numbers or has_currency or has_quarters):
        return 0.7, 0.3

    return default_dense, default_sparse


class BM25RetrieverAdapter(BaseRetriever):
    """A lightweight BM25 retriever with score-based filtering."""

    documents: list[Document]
    k: int = 12

    def __init__(self, documents: Sequence[Document], k: int, **kwargs: Any):
        super().__init__(documents=list(documents), k=max(1, int(k)), **kwargs)
        tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]
        object.__setattr__(self, "_bm25", BM25Okapi(tokenized_docs))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        if not query or not str(query).strip():
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        bm25 = getattr(self, "_bm25", None)
        if bm25 is None:
            tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]
            bm25 = BM25Okapi(tokenized_docs)
            object.__setattr__(self, "_bm25", bm25)

        scores = bm25.get_scores(tokenized_query)
        scored_pairs = [(idx, score) for idx, score in enumerate(scores) if score > 0.0]
        if not scored_pairs:
            q_set = set(tokenized_query)
            scored_pairs = [
                (idx, 1.0)
                for idx, doc in enumerate(self.documents)
                if q_set.intersection(self._tokenize(doc.page_content))
            ]
        if not scored_pairs:
            return []

        ranked_pairs = sorted(scored_pairs, key=lambda pair: pair[1], reverse=True)[: self.k]
        return [self.documents[idx] for idx, _ in ranked_pairs]


class HybridRetrieverBuilder:
    """Builds a hybrid retriever with FlashRank reranking and isolated collections."""

    _embeddings_cache: HuggingFaceEmbeddings | None = None

    def __init__(self, documents: list[Document], collection_name: str | None = None):
        if not documents:
            raise ValueError("At least one document is required to build the retriever.")
        self.documents = documents

        # Isolate Chroma collection by document ID or explicit name to prevent cross-contamination
        doc_id = documents[0].metadata.get("doc_id") if documents else None
        if collection_name:
            self.collection_name = collection_name
        elif doc_id:
            self.collection_name = f"{settings.CHROMA_COLLECTION_NAME}_{doc_id}"
        else:
            self.collection_name = settings.CHROMA_COLLECTION_NAME

        self.embeddings = self._get_or_create_embeddings()

    @classmethod
    def _get_or_create_embeddings(cls) -> HuggingFaceEmbeddings:
        if cls._embeddings_cache is None:
            cls._embeddings_cache = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": settings.DEVICE},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
        return cls._embeddings_cache

    def _build_dense_retriever(self):
        """Initializes Chroma Vector Store for semantic search with isolated collection."""
        vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory=str(Path(settings.CHROMA_DB_DIR)),
            collection_name=self.collection_name,
        )
        return vectorstore.as_retriever(search_kwargs={"k": settings.DENSE_K})

    def _build_sparse_retriever(self):
        """Initializes a BM25 keyword retriever without deprecated wrappers."""
        return BM25RetrieverAdapter(self.documents, settings.SPARSE_K)

    def _build_fallback_retriever(self, query: str = ""):
        """Builds a hybrid ensemble retriever using static or adaptive weights."""
        dense = self._build_dense_retriever()
        sparse = self._build_sparse_retriever()
        if settings.ENABLE_ADAPTIVE_RETRIEVAL and query:
            d_weight, s_weight = compute_adaptive_weights(query, settings.DENSE_WEIGHT, settings.SPARSE_WEIGHT)
            weights = [d_weight, s_weight]
        else:
            weights = [settings.DENSE_WEIGHT, settings.SPARSE_WEIGHT]
        return EnsembleRetriever(retrievers=[dense, sparse], weights=weights)

    def build_adaptive_ensemble(self, query: str = "") -> EnsembleRetriever:
        """Explicitly builds an EnsembleRetriever with weights tuned to the query characteristics."""
        return self._build_fallback_retriever(query)

    def get_hybrid_reranked_retriever(self) -> Any:
        """Combines dense + sparse retrievers and reranks top candidates with FlashRank."""
        ensemble = self._build_fallback_retriever()

        try:
            from langchain_community.document_compressors.flashrank_rerank import (
                FlashrankRerank,
            )
        except ImportError:
            logger.warning("FlashRank reranker is unavailable; using hybrid fallback retriever.")
            return ensemble

        try:
            compressor = FlashrankRerank(
                model=settings.RERANK_MODEL,
                top_n=min(settings.RERANK_TOP_N, len(self.documents)),
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble,
            )
            logger.info(
                "Hybrid + FlashRank retriever initialized successfully for collection: %s", self.collection_name
            )
            return compression_retriever
        except Exception as exc:  # pragma: no cover - exercised in runtime integration
            logger.warning("FlashRank reranking failed, using hybrid fallback retriever. Error: %s", exc)
            return ensemble
