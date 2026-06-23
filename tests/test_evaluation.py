import unittest

from language_assistant.evaluation import citation_coverage, retrieval_metrics


class EvaluationMetricTests(unittest.TestCase):
    def test_retrieval_metrics_hit_at_rank_two(self):
        recall, mrr = retrieval_metrics(["a.md", "b.md"], ["b.md"])

        self.assertEqual(recall, 1.0)
        self.assertEqual(mrr, 0.5)

    def test_retrieval_metrics_miss(self):
        recall, mrr = retrieval_metrics(["a.md"], ["b.md"])

        self.assertEqual(recall, 0.0)
        self.assertEqual(mrr, 0.0)

    def test_citation_coverage(self):
        self.assertEqual(citation_coverage("回答 [C1] 和 [C3]", available_count=4), 0.5)


if __name__ == "__main__":
    unittest.main()
