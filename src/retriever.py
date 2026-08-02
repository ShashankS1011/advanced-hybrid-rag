import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from src.config import settings

class HybridRetrieverBuilder:
    """Builds a Hybrid Retriever enhanced with FlashRank Cross-Encoder Reranking."""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
        )

    def _build_dense_retriever(self):
        """Initializes Chroma Vector Store for semantic search."""
        vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
        )
        # Fetch 12 candidates initially for reranking
        return vectorstore.as_retriever(search_kwargs={"k": 12})

    def _build_sparse_retriever(self):
        """Initializes BM25 Keyword Retriever."""
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = 12
        return bm25_retriever

    def get_hybrid_reranked_retriever(self) -> ContextualCompressionRetriever:
        """Combines Dense + Sparse retrievers and reranks top candidates with FlashRank."""
        dense = self._build_dense_retriever()
        sparse = self._build_sparse_retriever()

        # Step 1: Broad Hybrid Retrieval (Fetch top 12 from each)
        ensemble = EnsembleRetriever(
            retrievers=[dense, sparse],
            weights=[0.5, 0.5]
        )

        # Step 2: Cross-Encoder Reranker (Narrows down to top 4 highest relevance chunks)
        compressor = FlashrankRerank(top_n=settings.TOP_K_RETRIEVAL)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble
        )

        print("Hybrid + FlashRank Reranked Retriever initialized successfully.")
        return compression_retriever