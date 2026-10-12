import logging
import os
from pathlib import Path
from typing import List

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Handles parsing and chunking of PDF documents using PyMuPDF."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or greater.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_pdf(self, file_path: str | os.PathLike[str]) -> List[Document]:
        """Loads a PDF and returns page-level documents with source metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported: {path}")

        pdf = fitz.open(str(path))
        documents: List[Document] = []
        try:
            for page_index in range(pdf.page_count):
                page = pdf[page_index]
                text = page.get_text("text").strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": page_index},
                    )
                )
        finally:
            pdf.close()

        logger.info("Loaded %s pages from %s", len(documents), path.name)
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller chunks and tags page/source metadata."""
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata or {})
            source_value = metadata.get("source") or metadata.get("source_file") or "Unknown"
            metadata["chunk_id"] = idx
            metadata["source_file"] = os.path.basename(str(source_value))
            metadata["page_number"] = int(metadata.get("page", 0) or 0) + 1
            chunk.metadata = metadata

        logger.info("Created %s text chunks.", len(chunks))
        return chunks