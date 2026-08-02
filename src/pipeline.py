import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Any, Generator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import settings
from src.ingest import DocumentIngestor
from src.retriever import HybridRetrieverBuilder

class RAGPipeline:
    """End-to-End Hybrid RAG Pipeline with FlashRank Reranking & Streaming."""

    def __init__(self, document_path: str):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment or .env file.")

        # 1. Ingest
        ingestor = DocumentIngestor()
        docs = ingestor.load_pdf(document_path)
        self.chunks = ingestor.chunk_documents(docs)

        # 2. Reranked Retriever
        builder = HybridRetrieverBuilder(self.chunks)
        self.retriever = builder.get_hybrid_reranked_retriever()

        # 3. LLM with Streaming support
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            streaming=True
        )

        # 4. Prompt Template
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, state clearly that the answer is not available in the document.

Context:
{context}

Question: {question}

Answer:"""
        )

    def _format_docs(self, docs):
        formatted = []
        for d in docs:
            page = d.metadata.get("page_number", "Unknown")
            formatted.append(f"[Page {page}]\n{d.page_content}")
        return "\n\n".join(formatted)

    def retrieve_chunks(self, query: str):
        """Retrieve and rerank top matching chunks with metadata."""
        return self.retriever.invoke(query)

    def stream_answer(self, user_query: str, retrieved_docs: list) -> Generator[str, None, None]:
        """Yields LLM tokens in real-time as a generator stream."""
        formatted_context = self._format_docs(retrieved_docs)
        
        chain = self.prompt | self.llm | StrOutputParser()
        
        for chunk in chain.stream({"context": formatted_context, "question": user_query}):
            yield chunk