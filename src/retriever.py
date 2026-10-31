import logging
import re
from pathlib import Path
from typing import Any, List, Sequence

from langchain_chroma import Chroma
try:
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from src.config import settings

logger = logging.getLogger(__name__)


class BM25RetrieverAdapter:
    """A lightweight BM25 retriever with score-based filtering."""

    def __init__(self, documents: Sequence[Document], k: int):
        self.documents = list(documents)
        self.k = max(1, int(k))
        tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def invoke(self, query: str) -> List[Document]:
        if not query or not str(query).strip():
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        scored_pairs = [
            (idx, score) for idx, score in enumerate(scores) if score > 0.0
        ]
        if not scored_pairs:
            return []

        ranked_pairs = sorted(scored_pairs, key=lambda pair: pair[1], reverse=True)[: self.k]
        return [self.documents[idx] for idx, _ in ranked_pairs]

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)

    async def ainvoke(self, query: str) -> List[Document]:
        return self.invoke(query)


class HybridRetrieverBuilder:
    """Builds a hybrid retriever with FlashRank reranking and isolated collections."""

    _embeddings_cache: HuggingFaceEmbeddings | None = None

    def __init__(self, documents: List[Document], collection_name: str | None = None):
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

    def _build_fallback_retriever(self):
        """Builds an uncompressed hybrid retriever using configured weights."""
        dense = self._build_dense_retriever()
        sparse = self._build_sparse_retriever()
        weights = [settings.DENSE_WEIGHT, settings.SPARSE_WEIGHT]
        return EnsembleRetriever(retrievers=[dense, sparse], weights=weights)

    def get_hybrid_reranked_retriever(self) -> Any:
        """Combines dense + sparse retrievers and reranks top candidates with FlashRank."""
        ensemble = self._build_fallback_retriever()

        try:
            from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
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
            logger.info("Hybrid + FlashRank retriever initialized successfully for collection: %s", self.collection_name)
            return compression_retriever
        except Exception as exc:  # pragma: no cover - exercised in runtime integration
            logger.warning("FlashRank reranking failed, using hybrid fallback retriever. Error: %s", exc)
            return ensemble
