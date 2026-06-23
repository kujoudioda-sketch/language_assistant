from dataclasses import dataclass, field
from typing import Any
import unittest

from language_assistant.citations import extract_citation_labels, format_citation_list, format_context


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CitationTests(unittest.TestCase):
    def test_extract_citation_labels(self):
        self.assertEqual(extract_citation_labels("事实 A [C1]，事实 B [C12]。"), {"C1", "C12"})

    def test_format_context_and_citation_list(self):
        docs = [
            FakeDocument(
                page_content="Retrieval augmented generation grounds answers in retrieved evidence.",
                metadata={"source": "paper.md", "page": 0, "chunk_id": "paper-p1-0"},
            )
        ]

        context, citations = format_context(docs)
        rendered = format_citation_list(citations)

        self.assertIn("[C1]", context)
        self.assertIn("paper.md:page-1", context)
        self.assertIn("[C1] paper.md:page-1#paper-p1-0", rendered)
