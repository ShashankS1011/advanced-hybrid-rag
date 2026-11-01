from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api import _PIPELINE_REGISTRY, app
from src.pipeline import RAGPipeline

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "embedding_model" in data


def test_list_documents_empty():
    _PIPELINE_REGISTRY.clear()
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_upload_rejects_non_pdf():
    files = {"file": ("notes.txt", b"plain text", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]


def test_query_endpoint_with_mock_pipeline():
    mock_pipe = object.__new__(RAGPipeline)
    mock_pipe.doc_id = "doc_test_abc"
    mock_pipe.document_path = "data/financial_report.pdf"
    mock_pipe.metrics = {"document_pages": 1, "chunks_created": 2}
    mock_pipe.generate_answer = MagicMock(
        return_value={
            "answer": "Net income was $3.1M [Page 1].",
            "sources": [
                {
                    "file": "financial_report.pdf",
                    "page": 1,
                    "score": 0.95,
                    "text": "Net income: $3.1M",
                }
            ],
            "retrieved_chunks": [],
            "query": "Net income?",
        }
    )

    _PIPELINE_REGISTRY["doc_test_abc"] = mock_pipe

    payload = {
        "question": "What was net income?",
        "doc_id": "doc_test_abc",
        "chat_history": [],
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Net income was $3.1M [Page 1]."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page"] == 1


def test_stream_endpoint_with_mock_pipeline():
    mock_pipe = object.__new__(RAGPipeline)
    mock_pipe.doc_id = "doc_test_stream"
    mock_pipe.document_path = "data/financial_report.pdf"
    mock_pipe.metrics = {"document_pages": 1, "chunks_created": 2}

    sample_doc = Document(
        page_content="Cash reserves: $22.0M", metadata={"page_number": 1, "source_file": "report.pdf"}
    )
    mock_pipe.retrieve_chunks = MagicMock(return_value=[sample_doc])
    mock_pipe.stream_answer = MagicMock(return_value=iter(["Cash ", "reserves: ", "$22.0M"]))

    _PIPELINE_REGISTRY["doc_test_stream"] = mock_pipe

    payload = {
        "question": "What were cash reserves?",
        "doc_id": "doc_test_stream",
        "chat_history": [],
    }
    response = client.post("/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    content = response.text
    assert "Cash " in content
    assert "[DONE]" in content


def test_query_rejects_path_traversal():
    payload = {
        "question": "What is here?",
        "document_path": "../../etc/passwd",
        "chat_history": [],
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 400


def test_query_rejects_empty_question():
    payload = {
        "question": "   ",
        "doc_id": "nonexistent",
        "chat_history": [],
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422
