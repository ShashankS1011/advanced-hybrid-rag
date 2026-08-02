import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings

class DocumentIngestor:
    """Handles parsing and chunking of PDF documents using PyMuPDF."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Loads a PDF using PyMuPDFLoader for rich page metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {os.path.basename(file_path)}")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller text chunks and tags page numbers."""
        chunks = self.text_splitter.split_documents(documents)
        
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            source_file = os.path.basename(chunk.metadata.get("source", "Unknown"))
            page_number = chunk.metadata.get("page", 0) + 1  # 1-indexed page
            
            chunk.metadata["source_file"] = source_file
            chunk.metadata["page_number"] = page_number

        print(f"Created {len(chunks)} text chunks.")
        return chunks