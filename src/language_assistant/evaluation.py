from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

from .citations import extract_citation_labels, format_context

if TYPE_CHECKING:
    from .config import AssistantConfig


@dataclass(frozen=True)
class BenchmarkItem:
    question: str
    expected_sources: list[str]
    reference_answer: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    count: int
    recall_at_k: float
    mrr: float
    citation_coverage: float
    faithfulness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "faithfulness": None if self.faithfulness is None else round(self.faithfulness, 4),
        }


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            BenchmarkItem(
                question=raw["question"],
                expected_sources=list(raw.get("expected_sources", [])),
                reference_answer=raw.get("reference_answer"),
            )
        )
    if not items:
        raise ValueError("Benchmark file is empty.")
    return items


def retrieval_metrics(retrieved_sources: list[str], expected_sources: list[str]) -> tuple[float, float]:
    if not expected_sources:
        return 0.0, 0.0
    expected = set(expected_sources)
    recall = 1.0 if expected.intersection(retrieved_sources) else 0.0
    reciprocal_rank = 0.0
    for idx, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            reciprocal_rank = 1.0 / idx
            break
    return recall, reciprocal_rank


def citation_coverage(answer: str, available_count: int) -> float:
    labels = extract_citation_labels(answer)
    if available_count <= 0:
        return 0.0
    return min(len(labels), available_count) / available_count


def evaluate(config: "AssistantConfig", benchmark_path: str | Path) -> EvaluationResult:
    from .rag import RagAssistant
    from .vector_store import load_index

    items = load_benchmark(benchmark_path)
    index = load_index(config.index_path, config.embedding_model)
    assistant = RagAssistant(config)
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    citation_coverages: list[float] = []
    faithfulness_scores: list[float] = []

    for item in items:
        docs = index.similarity_search(item.question, k=config.top_k)
        sources = [str(doc.metadata.get("source", "")) for doc in docs]
        recall, rr = retrieval_metrics(sources, item.expected_sources)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

        answer = assistant.ask(item.question)
        citation_coverages.append(citation_coverage(answer.answer, len(answer.citations)))
        if config.faithfulness_judge:
            context, _ = format_context(docs)
            faithfulness_scores.append(judge_faithfulness(config.chat_model, context, answer.answer))

    return EvaluationResult(
        count=len(items),
        recall_at_k=mean(recalls),
        mrr=mean(reciprocal_ranks),
        citation_coverage=mean(citation_coverages),
        faithfulness=mean(faithfulness_scores) if faithfulness_scores else None,
    )


def judge_faithfulness(model: str, context: str, answer: str) -> float:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)
    prompt = (
        "你是 RAG 忠实度评估器。只判断回答是否完全被上下文支持。\n"
        "输出 0 到 1 的数字，不要输出解释。\n\n"
        f"上下文：\n{context}\n\n回答：\n{answer}"
    )
    response = llm.invoke(prompt)
    try:
        return max(0.0, min(1.0, float(str(response.content).strip())))
    except ValueError:
        return 0.0
