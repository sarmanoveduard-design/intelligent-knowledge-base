import copy
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from knowledge_base.document_representation import (
    DocumentRepresentationError, document_embedding_text, validate_document_representation,
)
from knowledge_base.retrieval_benchmark import load_inputs, run_benchmark, write_report
from knowledge_base.vector_store import InMemoryVectorStore


class RecordingProvider:
    dimension = 2

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_texts(self, texts):
        self.document_calls.append(tuple(texts))
        return ((1.0, 0.0), (0.0, 1.0))

    def embed_query(self, text):
        self.query_calls.append(text)
        return (1.0, 0.0)


class DocumentRepresentationTests(unittest.TestCase):
    def test_plain_preserves_text_exactly(self):
        text = " \nOriginal text with whitespace\n "
        self.assertEqual(document_embedding_text(text, {"chapter": "ignored", "article_title": "ignored"}), text)

    def test_structure_order_and_exact_fixed_labels(self):
        self.assertEqual(document_embedding_text(" Original text\n", {
            "chapter": "Chapter title", "article_title": "Article title",
        }, mode="structure_aware_v1"), "Chapter: Chapter title\nArticle: Article title\n\n Original text\n")

    def test_empty_chapter(self):
        self.assertEqual(document_embedding_text("text", {"chapter": "", "article_title": "Title"},
                                                mode="structure_aware_v1"), "Article: Title\n\ntext")

    def test_empty_article_title(self):
        self.assertEqual(document_embedding_text("text", {"chapter": "Title", "article_title": ""},
                                                mode="structure_aware_v1"), "Chapter: Title\n\ntext")

    def test_both_empty_metadata_return_original_text(self):
        for metadata in ({}, {"chapter": "", "article_title": ""}, {"chapter": " \n", "article_title": "\t"}):
            with self.subTest(metadata=metadata):
                self.assertEqual(document_embedding_text(" text\n", metadata, mode="structure_aware_v1"), " text\n")

    def test_heading_edges_trimmed_and_metadata_unchanged(self):
        metadata = {"chapter": " Chapter title ", "article_title": " Article title\n"}
        before = dict(metadata)
        self.assertEqual(document_embedding_text("text", metadata, mode="structure_aware_v1"),
                         "Chapter: Chapter title\nArticle: Article title\n\ntext")
        self.assertEqual(metadata, before)

    def test_unknown_mode_safe_error(self):
        with self.assertRaises(DocumentRepresentationError) as caught:
            document_embedding_text("PRIVATE_TEXT", {}, mode="PRIVATE_MODE_VALUE")
        self.assertNotIn("PRIVATE_TEXT", str(caught.exception))
        self.assertNotIn("PRIVATE_MODE_VALUE", str(caught.exception))
        self.assertIn("plain or structure_aware_v1", str(caught.exception))

    def test_invalid_structure_metadata_safe_error(self):
        with self.assertRaisesRegex(DocumentRepresentationError, "Invalid document structure metadata"):
            document_embedding_text("PRIVATE_TEXT", {"chapter": 123}, mode="structure_aware_v1")


class BenchmarkRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus, self.gold = self.root / "corpus.json", self.root / "gold.json"
        self.corpus.write_text(json.dumps({"chunks": [
            {"id": "chunk-a", "document_id": "doc_0001", "section_ref": "section-a", "text": "PRIVATE_BODY_A",
             "metadata": {"chapter": "PRIVATE_CHAPTER", "article_title": "PRIVATE_ARTICLE"}},
            {"id": "chunk-b", "document_id": "doc_0002", "section_ref": "section-b", "text": "PRIVATE_BODY_B",
             "metadata": {"chapter": "", "article_title": "SECOND_ARTICLE"}},
        ]}), encoding="utf-8")
        self.gold.write_text(json.dumps({"queries": [{"query_id": "synthetic-query", "text": "PRIVATE_QUERY",
            "positive_chunk_ids": ["chunk-a"], "hard_negative_chunk_ids": ["chunk-b"],
            "metadata": {"answer_hint": "PRIVATE_ANSWER_HINT"}}]}), encoding="utf-8")
        self.inputs = load_inputs(self.corpus, self.gold)

    def test_default_and_explicit_plain_match_old_embedding_inputs_and_metrics(self):
        default, explicit = RecordingProvider(), RecordingProvider()
        first = run_benchmark(default, self.inputs)
        second = run_benchmark(explicit, self.inputs, document_representation="plain")
        expected = [("PRIVATE_BODY_A", "PRIVATE_BODY_B")]
        self.assertEqual(default.document_calls, expected)
        self.assertEqual(explicit.document_calls, expected)
        self.assertEqual(default.query_calls, ["PRIVATE_QUERY"])
        for metric in ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "Top-1 accuracy", "MRR", "nDCG",
                       "positive-over-hard-negative", "unresolved refs"):
            self.assertEqual(first["summary"][metric], second["summary"][metric])
        self.assertEqual(first["summary"]["MRR"], 1)
        self.assertEqual(first["experiment"]["document_representation"], "plain")

    def test_structure_transforms_only_documents_and_stores_original_chunks(self):
        provider = RecordingProvider()
        store = InMemoryVectorStore(dimension=2)
        with patch("knowledge_base.retrieval_benchmark.InMemoryVectorStore", return_value=store):
            result = run_benchmark(provider, self.inputs, document_representation="structure_aware_v1")
        self.assertEqual(provider.document_calls, [(
            "Chapter: PRIVATE_CHAPTER\nArticle: PRIVATE_ARTICLE\n\nPRIVATE_BODY_A",
            "Article: SECOND_ARTICLE\n\nPRIVATE_BODY_B",
        )])
        self.assertEqual(provider.query_calls, ["PRIVATE_QUERY"])
        results = store.search((1., 0.), top_k=2)
        self.assertIs(results[0].chunk, self.inputs.chunks[0])
        self.assertEqual(results[0].chunk.text, "PRIVATE_BODY_A")
        self.assertEqual(results[0].chunk.document_id, "doc_0001")
        self.assertEqual(result["summary"]["Recall@10"], 1)
        self.assertEqual(result["summary"]["nDCG"], 1)
        self.assertEqual(result["summary"]["positive-over-hard-negative"], 1)

    def test_source_snapshots_and_hashes_unchanged_across_modes(self):
        corpus_before, gold_before = self.corpus.read_bytes(), self.gold.read_bytes()
        inputs_before = copy.deepcopy(self.inputs)
        plain = run_benchmark(RecordingProvider(), self.inputs)
        structure = run_benchmark(RecordingProvider(), self.inputs, document_representation="structure_aware_v1")
        self.assertEqual(self.inputs, inputs_before)
        self.assertEqual(self.corpus.read_bytes(), corpus_before)
        self.assertEqual(self.gold.read_bytes(), gold_before)
        for field in ("corpus_hash", "gold_set_hash"):
            self.assertEqual(plain["experiment"][field], structure["experiment"][field])
        reloaded = load_inputs(self.corpus, self.gold)
        self.assertEqual(reloaded.corpus_hash, self.inputs.corpus_hash)
        self.assertEqual(reloaded.gold_hash, self.inputs.gold_hash)

    def test_reports_contain_representation_and_no_private_content(self):
        for mode in ("plain", "structure_aware_v1"):
            result = run_benchmark(RecordingProvider(), self.inputs, document_representation=mode)
            directory = write_report(result, self.root / "reports")
            metadata = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["document_representation"], mode)
            self.assertIn(f"| document_representation | {mode} |", (directory / "REPORT.md").read_text(encoding="utf-8"))
            for path in directory.iterdir():
                content = path.read_text(encoding="utf-8")
                for forbidden in ("PRIVATE_BODY", "PRIVATE_CHAPTER", "PRIVATE_ARTICLE", "PRIVATE_QUERY", "PRIVATE_ANSWER_HINT"):
                    self.assertNotIn(forbidden, content)

    def test_failed_structure_run_reports_no_source_text_or_hint(self):
        provider = RecordingProvider()
        with patch.object(provider, "embed_texts", side_effect=RuntimeError("PRIVATE_BODY_A PRIVATE_ANSWER_HINT")):
            result = run_benchmark(provider, self.inputs, document_representation="structure_aware_v1")
        directory = write_report(result, self.root / "reports")
        self.assertEqual(result["experiment"]["status"], "incomplete")
        for filename in ("REPORT.md", "experiment.json", "errors.csv"):
            content = (directory / filename).read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE_BODY_A", content)
            self.assertNotIn("PRIVATE_ANSWER_HINT", content)

    def test_unknown_mode_rejected_before_embedding(self):
        provider = RecordingProvider()
        with self.assertRaises(DocumentRepresentationError):
            run_benchmark(provider, self.inputs, document_representation="unknown")
        self.assertEqual(provider.document_calls, [])
        self.assertEqual(provider.query_calls, [])

    def test_metadata_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Document metadata count"):
            run_benchmark(RecordingProvider(), replace(self.inputs, document_metadata=({},)),
                          document_representation="structure_aware_v1")

    def test_legacy_inputs_without_structure_metadata_work(self):
        provider = RecordingProvider()
        run_benchmark(provider, replace(self.inputs, document_metadata=()), document_representation="structure_aware_v1")
        self.assertEqual(provider.document_calls, [("PRIVATE_BODY_A", "PRIVATE_BODY_B")])

    def test_invalid_vector_count_is_safe_index_failure(self):
        provider = RecordingProvider()
        with patch.object(provider, "embed_texts", return_value=((1., 0.),)):
            result = run_benchmark(provider, self.inputs, document_representation="structure_aware_v1")
        self.assertEqual(result["experiment"]["status"], "incomplete")
        self.assertEqual(result["errors"][0]["code"], "embedding_index_failed")

    def test_cli_environment_mode_and_plain_default(self):
        from scripts.eval_retrieval import main
        args = ["eval_retrieval.py", "--corpus", str(self.corpus), "--gold", str(self.gold)]
        for env, expected in (({}, "plain"), ({"BENCHMARK_DOCUMENT_REPRESENTATION": "structure_aware_v1"}, "structure_aware_v1")):
            with patch.dict(os.environ, env, clear=True), patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO), \
                    patch("scripts.eval_retrieval.provider_from_environment", return_value=RecordingProvider()), \
                    patch("scripts.eval_retrieval.write_report", return_value=Path("reports/fake")) as report:
                self.assertEqual(main(), 0)
                self.assertEqual(report.call_args.args[0]["experiment"]["document_representation"], expected)

    def test_cli_unknown_environment_mode_blocks_provider_creation(self):
        from scripts.eval_retrieval import main
        args = ["eval_retrieval.py", "--corpus", str(self.corpus), "--gold", str(self.gold)]
        with patch.dict(os.environ, {"BENCHMARK_DOCUMENT_REPRESENTATION": "PRIVATE_INVALID_MODE"}, clear=True), \
                patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO) as output, \
                patch("scripts.eval_retrieval.provider_from_environment") as factory:
            self.assertEqual(main(), 1)
            factory.assert_not_called()
            self.assertIn("expected plain or structure_aware_v1", output.getvalue())
            self.assertNotIn("PRIVATE_INVALID_MODE", output.getvalue())


if __name__ == "__main__":
    unittest.main()
