from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

CITATION_PATTERN = re.compile(r"\[C(\d+)\]")


@dataclass(frozen=True)
class Citation:
    label: str
    source: str
    page: int | None
    chunk_id: str
    excerpt: str


def citation_label(index: int) -> str:
    return f"C{index + 1}"


def extract_citation_labels(answer: str) -> set[str]:
    return {f"C{match}" for match in CITATION_PATTERN.findall(answer)}


def document_citation(doc: Any, index: int, excerpt_chars: int = 220) -> Citation:
    metadata = doc.metadata or {}
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page")
    chunk_id = str(metadata.get("chunk_id", f"chunk-{index}"))
    excerpt = " ".join(doc.page_content.split())[:excerpt_chars]
    return Citation(citation_label(index), source, page if isinstance(page, int) else None, chunk_id, excerpt)


def format_context(documents: list[Any]) -> tuple[str, list[Citation]]:
    citations = [document_citation(doc, idx) for idx, doc in enumerate(documents)]
    blocks = []
    for citation, doc in zip(citations, documents):
        location = citation.source
        if citation.page is not None:
            location = f"{location}:page-{citation.page + 1}"
        blocks.append(f"[{citation.label}] {location}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks), citations


def format_citation_list(citations: list[Citation]) -> str:
    lines = []
    for citation in citations:
        location = citation.source
        if citation.page is not None:
            location = f"{location}:page-{citation.page + 1}"
        lines.append(f"[{citation.label}] {location}#{citation.chunk_id}")
    return "\n".join(lines)
