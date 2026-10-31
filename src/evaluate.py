import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    question: str
    ground_truth: str
    predicted_answer: str = ""
    contexts: list[str] = None
    citations: list[int] = None

    def __post_init__(self):
        if self.contexts is None:
            self.contexts = []
        if self.citations is None:
            self.citations = []


@dataclass
class EvaluationReport:
    total_samples: int
    citation_accuracy: float
    context_coverage: float
    lexical_similarity: float
    metrics: dict[str, float]
    details: list[dict[str, Any]]


def compute_citation_score(predicted_answer: str, retrieved_contexts: list[str]) -> float:
    """Verifies that [Page X] citations in the answer correspond to extracted context pages."""
    cited_pages = [int(p) for p in re.findall(r"\[Page\s+(\d+)\]", predicted_answer, re.IGNORECASE)]
    if not cited_pages:
        return 0.0

    # Check if cited page numbers exist in the context tags
    context_str = " ".join(retrieved_contexts)
    valid_citations = 0
    for page in cited_pages:
        if f"[Page {page}" in context_str:
            valid_citations += 1

    return valid_citations / len(cited_pages) if cited_pages else 0.0


def compute_token_overlap(reference: str, hypothesis: str) -> float:
    """Calculates word-level Jaccard similarity between ground truth and hypothesis."""
    ref_tokens = set(re.findall(r"\w+", (reference or "").lower()))
    hyp_tokens = set(re.findall(r"\w+", (hypothesis or "").lower()))

    if not ref_tokens:
        return 1.0 if not hyp_tokens else 0.0

    intersection = ref_tokens.intersection(hyp_tokens)
    union = ref_tokens.union(hyp_tokens)
    return len(intersection) / len(union) if union else 0.0


def evaluate_rag_pipeline(
    pipeline: Any,
    test_cases: list[dict[str, str]],
    use_ragas: bool = False,
) -> EvaluationReport:
    """Evaluates the RAG pipeline using ground-truth test cases."""
    samples: list[EvaluationSample] = []
    citation_scores = []
    overlap_scores = []

    for item in test_cases:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        # Retrieve and generate
        retrieved_docs = pipeline.retrieve_chunks(question)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Collect citations
        stream_chunks = list(pipeline.stream_answer(question, retrieved_docs))
        predicted = "".join(stream_chunks)

        cited_pages = [int(p) for p in re.findall(r"\[Page\s+(\d+)\]", predicted, re.IGNORECASE)]
        cit_score = compute_citation_score(predicted, [pipeline._format_docs(retrieved_docs)])
        overlap = compute_token_overlap(ground_truth, predicted) if ground_truth else 1.0

        citation_scores.append(cit_score)
        overlap_scores.append(overlap)

        sample = EvaluationSample(
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted,
            contexts=contexts,
            citations=cited_pages,
        )
        samples.append(sample)

    avg_citation = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    avg_coverage = sum(1.0 for s in samples if s.contexts) / len(samples) if samples else 0.0

    metrics = {
        "citation_accuracy": round(avg_citation, 4),
        "lexical_overlap": round(avg_overlap, 4),
        "retrieval_hit_rate": round(avg_coverage, 4),
    }

    # Optional Ragas evaluation if available
    if use_ragas:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, faithfulness

            ragas_data = {
                "question": [s.question for s in samples],
                "answer": [s.predicted_answer for s in samples],
                "contexts": [s.contexts for s in samples],
                "ground_truth": [s.ground_truth for s in samples],
            }
            dataset = Dataset.from_dict(ragas_data)
            ragas_results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
            metrics.update({k: float(v) for k, v in ragas_results.items()})
        except Exception as exc:
            logger.warning("Ragas evaluation skipped: %s", exc)

    report = EvaluationReport(
        total_samples=len(samples),
        citation_accuracy=round(avg_citation, 4),
        context_coverage=round(avg_coverage, 4),
        lexical_similarity=round(avg_overlap, 4),
        metrics=metrics,
        details=[asdict(s) for s in samples],
    )
    return report


SAMPLE_BENCHMARK = [
    {
        "question": "What was the total revenue recorded by Acme Corporation in Q4 2025?",
        "ground_truth": "Acme Corporation recorded a total net revenue of $12.5 Million in Q4 2025.",
    },
    {
        "question": "What was the operating expense and EBITDA margin?",
        "ground_truth": "Operating expenses were $7.2 Million yielding an EBITDA margin of 28.5%.",
    },
    {
        "question": "What are the primary operational risks identified for FY2026?",
        "ground_truth": "Key risks include cloud vendor supply chain constraints and increased electricity costs.",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline on sample benchmarks")
    parser.add_argument("--pdf", default="data/financial_report.pdf", help="Path to evaluation PDF")
    parser.add_argument("--ragas", action="store_true", help="Run Ragas metrics if available")
    parser.add_argument("--output", default="evaluation_report.json", help="Path to save evaluation report")
    args = parser.parse_args()

    from src.pipeline import RAGPipeline

    print(f"Initializing RAG Pipeline for: {args.pdf}")
    pipeline = RAGPipeline(args.pdf)

    print("Running evaluation on benchmark questions...")
    report = evaluate_rag_pipeline(pipeline, SAMPLE_BENCHMARK, use_ragas=args.ragas)

    print("\n--- Evaluation Results ---")
    print(f"Total Questions Evaluated: {report.total_samples}")
    print(f"Citation Accuracy:        {report.citation_accuracy:.2%}")
    print(f"Retrieval Hit Rate:       {report.context_coverage:.2%}")
    print(f"Lexical Overlap Score:    {report.lexical_similarity:.2%}")
    print(f"Detailed Metrics:         {report.metrics}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()

