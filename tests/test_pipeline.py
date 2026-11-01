from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.evaluate import (
    compute_citation_score,
    compute_token_overlap,
    evaluate_rag_pipeline,
)
from src.pipeline import RAGPipeline


@pytest.fixture
def mock_pipeline():
    pipeline = object.__new__(RAGPipeline)
    pipeline.document_path = "data/financial_report.pdf"
    pipeline.doc_id = "test_doc_123"
    pipeline.metrics = {"document_pages": 1, "chunks_created": 3}

    mock_llm = MagicMock()
    pipeline.llm = mock_llm

    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="Acme Corporation recorded total revenue of $12.5M in Q4 2025.",
            metadata={"source_file": "financial_report.pdf", "page_number": 1},
        )
    ]
    pipeline.retriever = mock_retriever
    return pipeline


def test_condense_query_returns_verbatim_when_no_history(mock_pipeline):
    result = mock_pipeline.condense_query("What was total revenue?", chat_history=[])
    assert result == "What was total revenue?"


def test_condense_query_rephrases_with_history(mock_pipeline):
    history = [
        {"role": "user", "content": "Tell me about Acme Corporation."},
        {"role": "assistant", "content": "Acme is a cloud software company."},
    ]

    with patch("langchain_core.prompts.ChatPromptTemplate.__or__") as mock_chain_or:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "What was Acme Corporation's revenue in Q4 2025?"
        mock_chain_or.return_value.__or__.return_value = mock_chain

        res = mock_pipeline.condense_query("What was its revenue in Q4?", chat_history=history)
        assert "revenue" in res.lower()


def test_retrieve_chunks_delegates_to_retriever(mock_pipeline):
    chunks = mock_pipeline.retrieve_chunks("Acme revenue")
    assert len(chunks) == 1
    assert "Acme Corporation" in chunks[0].page_content
    mock_pipeline.retriever.invoke.assert_called_once_with("Acme revenue")


def test_stream_answer_yields_tokens(mock_pipeline):
    docs = [
        Document(
            page_content="Operating expenses were $7.2 Million.",
            metadata={"source_file": "financial_report.pdf", "page_number": 1},
        )
    ]

    with patch("langchain_core.prompts.ChatPromptTemplate.__or__") as mock_chain_or:
        mock_chain = MagicMock()
        mock_chain.stream.return_value = ["Operating ", "expenses ", "were ", "$7.2M [Page 1]."]
        mock_chain_or.return_value.__or__.return_value = mock_chain

        tokens = list(mock_pipeline.stream_answer("What were expenses?", docs))
        full_text = "".join(tokens)

        assert "Operating expenses were $7.2M [Page 1]." in full_text


def test_stream_answer_handles_empty_docs(mock_pipeline):
    tokens = list(mock_pipeline.stream_answer("Any question", []))
    assert len(tokens) == 1
    assert "couldn't find relevant information" in tokens[0]


def test_generate_answer_returns_structured_dict(mock_pipeline):
    with patch("langchain_core.prompts.ChatPromptTemplate.__or__") as mock_chain_or:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Total revenue was $12.5M [Page 1]."
        mock_chain_or.return_value.__or__.return_value = mock_chain

        res = mock_pipeline.generate_answer("Total revenue?")
        assert res["answer"] == "Total revenue was $12.5M [Page 1]."
        assert len(res["sources"]) == 1
        assert res["sources"][0]["page"] == 1


def test_evaluation_metrics():
    # Citation score
    contexts = ["[Page 1 - financial_report.pdf] Revenue was $12.5M."]
    answer_with_valid_citation = "Revenue reached $12.5M [Page 1]."
    answer_with_invalid_citation = "Revenue was $12.5M [Page 99]."

    assert compute_citation_score(answer_with_valid_citation, contexts) == 1.0
    assert compute_citation_score(answer_with_invalid_citation, contexts) == 0.0

    # Token overlap
    ref = "Acme revenue was 12.5 million dollars"
    hyp = "Acme revenue was 12.5 million"
    overlap = compute_token_overlap(ref, hyp)
    assert overlap > 0.7


def test_evaluate_rag_pipeline_runner(mock_pipeline):
    with patch.object(mock_pipeline, "stream_answer", return_value=["Revenue was $12.5M [Page 1]."]):
        test_cases = [
            {
                "question": "What was the revenue?",
                "ground_truth": "Revenue was $12.5M.",
            }
        ]
        report = evaluate_rag_pipeline(mock_pipeline, test_cases, use_ragas=False)

        assert report.total_samples == 1
        assert report.citation_accuracy == 1.0
        assert report.context_coverage == 1.0
        assert report.lexical_similarity > 0.5
