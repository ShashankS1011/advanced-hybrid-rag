import logging
import os
from collections.abc import Generator
from time import perf_counter

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-End Hybrid RAG Pipeline with FlashRank Reranking & Streaming."""

    def __init__(self, document_path: str):
        settings.validate_runtime_configuration()
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"File not found: {document_path}")

        from src.ingest import DocumentIngestor
        from src.retriever import HybridRetrieverBuilder

        ingestor = DocumentIngestor()
        docs = ingestor.load_pdf(document_path)
        self.chunks = ingestor.chunk_documents(docs)
        if not self.chunks:
            raise ValueError("No content could be extracted from the provided PDF.")

        builder = HybridRetrieverBuilder(self.chunks)
        self.retriever = builder.get_hybrid_reranked_retriever()
        self.llm = self._build_llm()
        self.metrics = {"document_pages": len(docs), "chunks_created": len(self.chunks)}

        from langchain_core.prompts import ChatPromptTemplate

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an expert AI assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, state clearly that the answer is not available in the document.

Context:
{context}

Question: {question}

Answer:"""
        )

    def _build_llm(self):
        from langchain_groq import ChatGroq

        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.1,
            streaming=True,
        )

    def _format_docs(self, docs):
        formatted = []
        for document in docs:
            page = document.metadata.get("page_number", "Unknown")
            formatted.append(f"[Page {page}]\n{document.page_content}")
        return "\n\n".join(formatted)

    def retrieve_chunks(self, query: str):
        """Retrieve and rerank top matching chunks with metadata."""
        if not query or not str(query).strip():
            return []

        start = perf_counter()
        results = self.retriever.invoke(query)
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info("retrieval query=%s hits=%s elapsed_ms=%.2f", query, len(results), elapsed_ms)
        return results

    def stream_answer(self, user_query: str, retrieved_docs: list) -> Generator[str, None, None]:
        """Yields LLM tokens in real time as a generator stream."""
        if not retrieved_docs:
            yield "I couldn't find relevant information in the supplied document."
            logger.warning("No documents were available for generation for query: %s", user_query)
            return

        from langchain_core.output_parsers import StrOutputParser

        formatted_context = self._format_docs(retrieved_docs)
        chain = self.prompt | self.llm | StrOutputParser()

        start = perf_counter()
        streamed_response = ""
        for chunk in chain.stream({"context": formatted_context, "question": user_query}):
            streamed_response += chunk
            yield chunk

        elapsed_ms = (perf_counter() - start) * 1000
        logger.info(
            "answer generation query=%s docs=%s response_chars=%s elapsed_ms=%.2f",
            user_query,
            len(retrieved_docs),
            len(streamed_response),
            elapsed_ms,
        )