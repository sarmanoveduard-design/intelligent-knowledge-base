import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowledge_base.benchmark_config import BenchmarkConfigurationError
from knowledge_base.report_comparison import compare_reports


class ReportComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"

    def report(self, name, *, corpus="a" * 64, gold="b" * 64, **changes):
        directory = self.root / name
        directory.mkdir()
        metadata = {"provider": "ollama", "model_name": "bge-m3", "dimensions": 1024, "corpus_hash": corpus,
            "gold_set_hash": gold, "top_k": 10, "error_count": 0, "estimated_cost": None,
            "code_commit_sha": "c" * 40, "code_dirty": False, "status": "complete",
            "query_text": "RAW QUERY TEXT", "answer_hint": "PRIVATE ANSWER HINT", "corpus_text": "RAW CORPUS TEXT"}
        metadata.update(changes)
        (directory / "experiment.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "summary.csv").write_text("metric,value\nRecall@1,0.5\nRecall@3,1\nRecall@5,1\nRecall@10,1\nMRR,0.5\nnDCG,0.7\nCandidate Recall@50,1\nprivate_metric,RAW CORPUS TEXT\n", encoding="utf-8")
        return directory

    def test_same_hashes_compared_and_incompatible_skipped(self):
        first, second = self.report("first"), self.report("second", retrieval_mode="bm25", provider=None, model_name=None, dimensions=None)
        other_corpus = self.report("other", corpus="d" * 64)
        other_gold = self.report("gold-other", gold="e" * 64)
        counts = compare_reports([other_gold, second, other_corpus, first], self.output, corpus_hash="a" * 64, gold_hash="b" * 64)
        self.assertEqual(counts, {"compared": 2, "incompatible": 2, "invalid": 0})
        self.assertTrue((self.output / "COMPARISON.md").is_file())
        with (self.output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["retrieval_mode"] for row in rows}, {"dense", "bm25"})

    def test_output_excludes_texts_hints_and_arbitrary_metadata(self):
        first = self.report("first", model_name="RAW QUERY TEXT", provider="RAW CORPUS TEXT", code_commit_sha="PRIVATE ANSWER HINT")
        compare_reports([first], self.output)
        for path in self.output.iterdir():
            text = path.read_text(encoding="utf-8")
            for forbidden in ("RAW QUERY TEXT", "RAW CORPUS TEXT", "PRIVATE ANSWER HINT", "query_text", "private_metric"):
                self.assertNotIn(forbidden, text)

    def test_legacy_report_defaults_and_missing_candidates_remain_na(self):
        compare_reports([self.report("legacy")], self.output)
        with (self.output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["retrieval_mode"], "dense")
        self.assertEqual(row["document_representation"], "plain")
        self.assertEqual(row["candidate_k"], "")
        self.assertEqual(row["Candidate Recall@20"], "")

    def test_multiple_hash_groups_require_selection(self):
        with self.assertRaises(BenchmarkConfigurationError):
            compare_reports([self.report("first"), self.report("other", corpus="d" * 64)], self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_report_skipped_without_printing_content(self):
        first, bad = self.report("first"), self.report("bad")
        (bad / "experiment.json").write_text("PRIVATE RAW INVALID JSON", encoding="utf-8")
        self.assertEqual(compare_reports([first, bad], self.output)["invalid"], 1)

    def test_selection_validation(self):
        for kwargs in ({"corpus_hash": "a" * 64}, {"corpus_hash": "bad", "gold_hash": "b" * 64}):
            with self.assertRaises(BenchmarkConfigurationError):
                compare_reports([], self.output, **kwargs)

    def test_cli_creates_comparison_offline(self):
        from scripts.compare_retrieval_reports import main
        self.report("first")
        with patch("sys.argv", ["compare_retrieval_reports.py", "--reports-root", str(self.root), "--out", str(self.output)]), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(), 0)
            self.assertIn("Compared=1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
