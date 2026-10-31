import hashlib
import logging
import os
import re
from pathlib import Path

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)


def _format_table_as_markdown(table_data: list[list[str | None]]) -> str:
    """Converts a 2D list of extracted table cells into a Markdown table."""
    if not table_data or not table_data[0]:
        return ""

    cleaned_rows = []
    for row in table_data:
        cleaned_rows.append([re.sub(r"\s+", " ", str(cell or "").strip()) for cell in row])

    col_count = max(len(r) for r in cleaned_rows)
    # Normalize rows to uniform width
    for row in cleaned_rows:
        while len(row) < col_count:
            row.append("")

    header = cleaned_rows[0]
    separator = ["---"] * col_count
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in cleaned_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def compute_file_hash(file_path: str | os.PathLike[str]) -> str:
    """Computes a quick SHA-256 hash of a file for identification and caching."""
    resolved_path = os.path.normpath(str(file_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"File not found: {resolved_path}")
    hasher = hashlib.sha256()
    with open(resolved_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


class DocumentIngestor:
    """Handles parsing and chunking of PDF documents with table awareness."""

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

    def load_pdf(self, file_path: str | os.PathLike[str]) -> list[Document]:
        """Loads a PDF and returns page-level documents with table extraction and metadata."""
        resolved_path = os.path.normpath(str(file_path))
        path = Path(resolved_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported: {path}")

        doc_hash = compute_file_hash(resolved_path)
        pdf = fitz.open(resolved_path)
        documents: list[Document] = []
        try:
            page_count = pdf.page_count
            for page_index in range(page_count):
                page = pdf[page_index]
                page_text = (page.get_text("text") or "").strip()

                # Extract tabular data if present
                tables_md = []
                try:
                    tab_finder = page.find_tables()
                    for tab in tab_finder.tables:
                        extracted = tab.extract()
                        if extracted and len(extracted) > 1:
                            md_tab = _format_table_as_markdown(extracted)
                            if md_tab:
                                tables_md.append(md_tab)
                except Exception as exc:
                    logger.debug("Table extraction skipped on page %s: %s", page_index + 1, exc)

                combined_parts = []
                if page_text:
                    combined_parts.append(page_text)
                if tables_md:
                    combined_parts.append("\n\n### Extracted Tables:\n" + "\n\n".join(tables_md))

                full_content = "\n\n".join(combined_parts).strip()
                if not full_content:
                    continue

                documents.append(
                    Document(
                        page_content=full_content,
                        metadata={
                            "source": str(path),
                            "source_file": path.name,
                            "doc_id": doc_hash,
                            "page": page_index,
                            "total_pages": page_count,
                        },
                    )
                )
        finally:
            pdf.close()

        logger.info("Loaded %s pages from %s (hash: %s)", len(documents), path.name, doc_hash)
        return documents

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Splits documents into smaller chunks and preserves page/source metadata."""
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata or {})
            source_value = metadata.get("source") or metadata.get("source_file") or "Unknown"
            metadata["chunk_id"] = idx
            metadata["source_file"] = os.path.basename(str(source_value))
            metadata["page_number"] = int(metadata.get("page", 0) or 0) + 1
            metadata["doc_id"] = metadata.get("doc_id", "default")
            chunk.metadata = metadata

        logger.info("Created %s text chunks.", len(chunks))
        return chunks
