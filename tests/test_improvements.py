from src.ingest import DocumentIngestor
from src.pipeline import QueryCache, decompose_query
from src.retriever import compute_adaptive_weights


def test_load_text_document(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Acme Corporation strategic goals for FY2026.", encoding="utf-8")

    ingestor = DocumentIngestor()
    docs = ingestor.load_document(str(txt_file))
    assert len(docs) == 1
    assert "strategic goals" in docs[0].page_content
    assert docs[0].metadata["source_file"] == "notes.txt"
    assert docs[0].metadata["format"] == "txt"


def test_load_csv_document(tmp_path):
    csv_file = tmp_path / "financials.csv"
    csv_file.write_text("Quarter,Revenue,EBITDA\nQ3 2025,$28.4M,$6.2M\nQ4 2025,$31.2M,$7.1M\n", encoding="utf-8")

    ingestor = DocumentIngestor()
    docs = ingestor.load_document(str(csv_file))
    assert len(docs) == 1
    assert "| Quarter | Revenue | EBITDA |" in docs[0].page_content
    assert "$31.2M" in docs[0].page_content
    assert docs[0].metadata["format"] == "csv"


def test_load_documents_multiple_files(tmp_path):
    f1 = tmp_path / "file1.txt"
    f1.write_text("Content 1", encoding="utf-8")
    f2 = tmp_path / "file2.txt"
    f2.write_text("Content 2", encoding="utf-8")

    ingestor = DocumentIngestor()
    docs = ingestor.load_documents([str(f1), str(f2)])
    assert len(docs) == 2


def test_compute_adaptive_weights_numeric():
    # Numeric / exact metrics query should tilt toward sparse (BM25)
    dense_w, sparse_w = compute_adaptive_weights("What was net revenue in Q4 2025 and total EBITDA of $3.1M?")
    assert sparse_w > dense_w
    assert sparse_w == 0.7
    assert dense_w == 0.3


def test_compute_adaptive_weights_conceptual():
    # Conceptual / explanatory query should tilt toward dense (vector)
    dense_w, sparse_w = compute_adaptive_weights(
        "Explain the high-level strategy, industry headwinds, and operational risks."
    )
    assert dense_w > sparse_w
    assert dense_w == 0.7
    assert sparse_w == 0.3


def test_compute_adaptive_weights_default():
    dense_w, sparse_w = compute_adaptive_weights("General document inquiry.")
    assert dense_w == 0.5
    assert sparse_w == 0.5


def test_query_cache():
    cache = QueryCache(max_size=2)
    doc_id = "doc123"

    sample_res = {"answer": "Net income is $5M", "sources": []}
    cache.set(doc_id, "What is net income?", sample_res)

    # Normalized lookup should match
    hit = cache.get(doc_id, "  what IS NET income?  ")
    assert hit is not None
    assert hit["answer"] == "Net income is $5M"

    # Non-existent query should miss
    assert cache.get(doc_id, "Different question") is None


def test_decompose_query():
    compound_query = "What was Q4 revenue? What were operating risks?"
    parts = decompose_query(compound_query)
    assert len(parts) == 2
    assert "What was Q4 revenue?" in parts[0]
    assert "What were operating risks?" in parts[1]

    single_query = "What was the total revenue?"
    assert decompose_query(single_query) == [single_query]
