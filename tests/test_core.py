import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.config import Settings, settings
from src.ingest import DocumentIngestor, _format_table_as_markdown, compute_file_hash
from src.pipeline import RAGPipeline
from src.retriever import BM25RetrieverAdapter, HybridRetrieverBuilder


def test_settings_have_reasonable_defaults():
    assert isinstance(settings.GROQ_API_KEY, str)
    assert settings.CHROMA_DB_DIR == "./chroma_db"
    assert settings.CHROMA_COLLECTION_NAME
    assert settings.EMBEDDING_MODEL
    assert settings.CHUNK_SIZE > 0
    assert settings.CHUNK_OVERLAP >= 0
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE
    assert settings.TOP_K_RETRIEVAL > 0
    assert settings.DENSE_K > 0
    assert settings.SPARSE_K > 0
    assert settings.RERANK_TOP_N > 0
    assert 0.0 <= settings.DENSE_WEIGHT <= 1.0
    assert 0.0 <= settings.SPARSE_WEIGHT <= 1.0
    assert 0.0 <= settings.TEMPERATURE <= 2.0


def test_settings_validate_ranges():
    with pytest.raises(ValueError, match="DENSE_WEIGHT"):
        Settings(DENSE_WEIGHT=1.5)
    with pytest.raises(ValueError, match="TEMPERATURE"):
        Settings(TEMPERATURE=-0.5)


def test_document_ingestor_chunks_and_tags_metadata():
    ingestor = DocumentIngestor(chunk_size=40, chunk_overlap=10)
    documents = [
        Document(
            page_content="Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 3,
            metadata={"source": "sample.pdf", "page": 0, "doc_id": "test1234"},
        )
    ]

    chunks = ingestor.chunk_documents(documents)

    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_id"] == index
        assert chunk.metadata["source_file"] == "sample.pdf"
        assert chunk.metadata["page_number"] >= 1
        assert chunk.metadata["doc_id"] == "test1234"


def test_document_ingestor_rejects_non_pdf_files(tmp_path):
    ingestor = DocumentIngestor()
    bad_path = tmp_path / "notes.txt"
    bad_path.write_text("not a pdf")

    with pytest.raises(ValueError, match="Only PDF files"):
        ingestor.load_pdf(str(bad_path))


def test_document_ingestor_returns_empty_list_for_empty_input():
    ingestor = DocumentIngestor()

    assert ingestor.chunk_documents([]) == []


def test_table_formatter_creates_markdown_table():
    raw_table = [
        ["Metric", "Value"],
        ["Revenue", "$12.5M"],
        ["Margin", "28%"],
    ]
    md = _format_table_as_markdown(raw_table)
    assert "| Metric | Value |" in md
    assert "| --- | --- |" in md
    assert "| Revenue | $12.5M |" in md


def test_compute_file_hash(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Hello World!")
    hash_val = compute_file_hash(sample)
    assert len(hash_val) == 16
    assert hash_val == compute_file_hash(sample)


def test_hybrid_retriever_builder_requires_documents():
    with pytest.raises(ValueError, match="At least one document"):
        HybridRetrieverBuilder([])


def test_bm25_adapter_returns_ranked_documents():
    documents = [
        Document(page_content="alpha beta gamma"),
        Document(page_content="alpha delta"),
        Document(page_content="zeta eta"),
    ]
    retriever = BM25RetrieverAdapter(documents, k=2)

    results = retriever.invoke("alpha beta")

    assert len(results) == 2
    assert results[0].page_content == "alpha beta gamma"


def test_bm25_adapter_returns_empty_on_zero_match():
    documents = [
        Document(page_content="apple banana cherry"),
        Document(page_content="date fig grape"),
    ]
    retriever = BM25RetrieverAdapter(documents, k=2)

    # Search for term completely absent from documents
    results = retriever.invoke("zebra elephant rhinoceros")
    assert results == []


def test_rag_pipeline_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        RAGPipeline("/tmp/example.pdf")


def test_rag_pipeline_rejects_missing_document(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key", raising=False)

    with pytest.raises(FileNotFoundError):
        RAGPipeline(str(tmp_path / "missing.pdf"))


def test_rag_pipeline_formats_context_with_page_numbers():
    pipeline = object.__new__(RAGPipeline)
    documents = [
        Document(page_content="First answer", metadata={"page_number": 2, "source_file": "doc1.pdf"}),
        Document(page_content="Second answer", metadata={"page_number": 7, "source_file": "doc1.pdf"}),
    ]

    formatted = pipeline._format_docs(documents)

    assert "[Page 2 - doc1.pdf]" in formatted
    assert "[Page 7 - doc1.pdf]" in formatted
    assert "First answer" in formatted
    assert "Second answer" in formatted


def test_rag_pipeline_returns_empty_list_for_blank_queries():
    pipeline = object.__new__(RAGPipeline)

    assert pipeline.retrieve_chunks("   ") == []


def test_document_ingestor_can_parse_a_real_pdf(tmp_path):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(b"%PDF-1.4\n")
        temp_pdf.flush()
        temp_path = temp_pdf.name

    try:
        pdf_document = Document(page_content="Revenue grew by 20 percent in Q4.", metadata={"source": temp_path, "page": 0})
        ingestor = DocumentIngestor(chunk_size=80, chunk_overlap=20)
        chunks = ingestor.chunk_documents([pdf_document])

        assert chunks
        assert chunks[0].metadata["source_file"] == Path(temp_path).name
    finally:
        Path(temp_path).unlink(missing_ok=True)
