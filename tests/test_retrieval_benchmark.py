import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge_base.embedding_config import provider_from_environment
from knowledge_base.ollama_embeddings import OllamaEmbeddingProvider
from knowledge_base.openai_embeddings import OpenAIEmbeddingProvider
from knowledge_base.retrieval_benchmark import NOTICE, load_inputs, ranking_metrics, run_benchmark, write_report
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import InMemoryVectorStore
from test_openai_embeddings import fake_client
from test_retriever import make_chunk


class BenchmarkTests(unittest.TestCase):
    def inputs(self, root, *, unresolved=False):
        corpus, gold = root / "corpus.json", root / "gold.json"
        corpus.write_text(json.dumps({"chunks": [{"id": "private/path/a", "text": "CLOSED DOCUMENT TEXT"},
                                                   {"id": "private/path/b", "text": "OTHER CLOSED TEXT"}]}), encoding="utf-8")
        gold.write_text(json.dumps({"queries": [{"text": "CLOSED QUERY TEXT", "positive_chunk_ids": ["private/path/a"],
                                                 "hard_negative_chunk_ids": ["missing" if unresolved else "private/path/b"]}]}), encoding="utf-8")
        return load_inputs(corpus, gold)

    def test_ranking_metrics_known_example(self):
        metrics = ranking_metrics(["n", "p", "q"], ("p", "q"))
        self.assertEqual(metrics["Recall@1"], 0)
        self.assertEqual(metrics["Recall@3"], 1)
        self.assertEqual(metrics["MRR"], .5)
        self.assertAlmostEqual(metrics["nDCG"], (1 / math.log2(3) + .5) / (1 + 1 / math.log2(3)))

    def test_query_failure_counted_as_zero_and_usage_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = fake_client(2)
            original = client.embeddings.create.side_effect
            provider = OpenAIEmbeddingProvider(dimensions=2, client=client)
            provider.embed_query("prior request")
            client.embeddings.create.side_effect = [original(input=["a", "b"]), RuntimeError("private failure")]
            result = run_benchmark(provider, self.inputs(root))
            self.assertEqual(result["summary"]["MRR"], 0)
            self.assertEqual(result["experiment"]["token_usage"]["total_tokens"], 7)
            self.assertEqual(result["experiment"]["token_usage"]["error_count"], 1)
            self.assertEqual(result["per_query"][0]["status"], "error")

    def test_local_provider_without_usage_and_positive_win(self):
        from test_retriever import FakeSemanticEmbeddingProvider
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.inputs(root)
            (root / "corpus.json").write_text(json.dumps({"chunks": [{"id": "a", "text": "кошка"}, {"id": "b", "text": "машина"}]}), encoding="utf-8")
            (root / "gold.json").write_text(json.dumps({"queries": [{"text": "кошка", "positive_chunk_ids": ["a"], "hard_negative_chunk_ids": ["b"]}]}), encoding="utf-8")
            result = run_benchmark(FakeSemanticEmbeddingProvider(), load_inputs(root / "corpus.json", root / "gold.json"))
            self.assertIsNone(result["experiment"]["token_usage"]["total_tokens"])
            self.assertIsNone(result["experiment"]["estimated_cost"])
            self.assertEqual(result["summary"]["positive-over-hard-negative"], 1)

    def test_report_layout_metadata_and_privacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = OpenAIEmbeddingProvider(dimensions=2, client=fake_client(2))
            with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-secret"}):
                result = run_benchmark(provider, self.inputs(root), cost_per_million_tokens=2)
                report = write_report(result, root / "reports")
            self.assertEqual({p.name for p in report.iterdir()}, {"REPORT.md", "experiment.json", "summary.csv", "per_query.csv", "errors.csv"})
            content = "\n".join(p.read_text(encoding="utf-8") for p in report.iterdir())
            for forbidden in ("synthetic-secret", "CLOSED DOCUMENT TEXT", "CLOSED QUERY TEXT", "private/path", temporary):
                self.assertNotIn(forbidden, content)
            self.assertIn(NOTICE, content)
            metadata = json.loads((report / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["token_usage"]["total_tokens"], 14)
            self.assertEqual(metadata["estimated_cost"], 28 / 1_000_000)
            self.assertEqual(metadata["distance_metric"], "cosine")
            self.assertTrue(metadata["timestamp_utc"].endswith("+00:00"))
            self.assertEqual(result["summary"]["positive-over-hard-negative"], 0)  # equal cosine scores

    def test_api_failure_report_is_sanitized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = fake_client(2)
            client.embeddings.create.side_effect = RuntimeError("synthetic-secret CLOSED DOCUMENT TEXT")
            result = run_benchmark(OpenAIEmbeddingProvider(dimensions=2, client=client), self.inputs(root))
            report = write_report(result, root / "reports")
            content = "\n".join(p.read_text(encoding="utf-8") for p in report.iterdir())
            self.assertNotIn("synthetic-secret", content)
            self.assertNotIn("CLOSED DOCUMENT TEXT", content)
            self.assertEqual(result["experiment"]["status"], "incomplete")
            self.assertEqual(result["experiment"]["error_count"], 1)
            self.assertEqual(result["summary"]["Recall@10"], 0)

    def test_unresolved_refs_excluded_from_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_benchmark(OpenAIEmbeddingProvider(dimensions=2, client=fake_client(2)), self.inputs(root, unresolved=True))
            self.assertEqual(result["summary"]["unresolved refs"], 1)
            self.assertIsNone(result["summary"]["Recall@1"])
            self.assertEqual(result["experiment"]["evaluable_query_count"], 0)

    def test_snapshot_hash_changes_with_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.inputs(root)
            with (root / "corpus.json").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            second = load_inputs(root / "corpus.json", root / "gold.json")
            self.assertNotEqual(first.corpus_hash, second.corpus_hash)
            self.assertEqual(first.gold_hash, second.gold_hash)

    def test_invalid_gold_and_duplicate_chunk_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.inputs(root)
            (root / "gold.json").write_text(json.dumps({"queries": [{"text": "q", "positive_chunk_ids": []}]}))
            with self.assertRaises(ValueError):
                load_inputs(root / "corpus.json", root / "gold.json")
            self.inputs(root)
            (root / "corpus.json").write_text(json.dumps({"chunks": [{"id": "a", "text": "a"}, {"id": "a", "text": "b"}]}))
            with self.assertRaises(ValueError):
                load_inputs(root / "corpus.json", root / "gold.json")

    def test_top_k_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                run_benchmark(OpenAIEmbeddingProvider(dimensions=2, client=fake_client(2)), self.inputs(root), top_k=5)

    def test_unknown_model_store_rejected(self):
        store = InMemoryVectorStore(dimension=1024)
        store.add(make_chunk(0, "a"), (1.,) * 1024)
        with self.assertRaises(ValueError):
            Retriever(OllamaEmbeddingProvider(), store)

    def test_ollama_openai_cannot_share_store(self):
        store = InMemoryVectorStore(dimension=1024)
        Retriever(OllamaEmbeddingProvider(), store)
        with self.assertRaises(ValueError):
            Retriever(OpenAIEmbeddingProvider(dimensions=1024, client=fake_client(1024)), store)

    def test_ollama_environment_factory(self):
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://localhost:11434"}, clear=True):
            provider = provider_from_environment()
        self.assertEqual(provider.identity.model_name, "bge-m3")
        self.assertEqual(provider.dimension, 1024)
