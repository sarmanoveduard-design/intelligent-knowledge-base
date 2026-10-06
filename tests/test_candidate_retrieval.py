import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowledge_base.benchmark_config import BenchmarkConfigurationError, RetrievalConfig, config_from_environment
from knowledge_base.bm25 import BM25Index, tokenize
from knowledge_base.candidate_retrieval import Candidate, CandidateResult, candidate_metrics, final_candidates
from knowledge_base.fusion import reciprocal_rank_fusion, reciprocal_rank_fusion_scores
from knowledge_base.retrieval_benchmark import code_revision, load_inputs, run_benchmark, write_report


class PositionProvider:
    dimension = 2

    def __init__(self):
        self.documents, self.queries = [], []

    def embed_texts(self, texts):
        self.documents.append(tuple(texts))
        return tuple((1., float(i)) for i in range(len(texts)))

    def embed_query(self, text):
        self.queries.append(text)
        return (1., 0.)


class BM25Tests(unittest.TestCase):
    def test_known_bm25_formula(self):
        index = BM25Index(["cat cat", "dog"])
        expected = math.log(1 + (2 - 1 + .5) / (1 + .5)) * 2 * 2.5 / (2 + 1.5 * (.25 + .75 * 2 / 1.5))
        self.assertAlmostEqual(index.scores("cat")[0], expected)
        self.assertEqual(index.scores("cat")[1], 0)

    def test_russian_unicode_and_normalization(self):
        self.assertEqual(tokenize("ПЕРСОНАЛЬНЫЕ данные, ёлка; №１５!"), ("персональные", "данные", "ёлка", "no15"))
        scores = BM25Index(["персональные ДАННЫЕ", "обучение сотрудника"]).scores("Персональные данные")
        self.assertGreater(scores[0], scores[1])

    def test_determinism_and_query_term_order(self):
        index = BM25Index(["данные данные обучение", "обучение", "данные"])
        self.assertEqual(index.scores("данные обучение"), index.scores("обучение данные данные"))
        self.assertEqual(index.scores("данные"), index.scores("данные"))

    def test_empty_token_documents_and_query(self):
        self.assertEqual(BM25Index(["!!!", " "]).scores("data"), (0., 0.))
        self.assertEqual(BM25Index(["data"]).scores("!!!"), (0.,))
        self.assertEqual(BM25Index([]).scores("data"), ())

    def test_invalid_parameters(self):
        for kwargs in ({"k1": 0}, {"k1": float("nan")}, {"b": -1}, {"b": 2}):
            with self.assertRaises(ValueError):
                BM25Index(["x"], **kwargs)


class CandidateConfigTests(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = config_from_environment()
        self.assertEqual(config, RetrievalConfig("dense", 50, 60, 10))

    def test_candidate_validation(self):
        for candidate_k in (0, -1, 9, True, 10.5):
            with self.subTest(candidate_k=candidate_k), self.assertRaises(BenchmarkConfigurationError):
                RetrievalConfig(candidate_k=candidate_k)
        self.assertEqual(RetrievalConfig(candidate_k=50).effective_candidate_k(2), 2)

    def test_unknown_mode_and_bad_rrf(self):
        for kwargs in ({"retrieval_mode": "unknown"}, {"rrf_k": 0}, {"rrf_k": float("nan")}, {"rrf_k": True}):
            with self.assertRaises(BenchmarkConfigurationError):
                RetrievalConfig(**kwargs)

    def test_environment_config(self):
        with patch.dict(os.environ, {"BENCHMARK_RETRIEVAL_MODE": "hybrid_rrf", "BENCHMARK_CANDIDATE_K": "30",
                                   "BENCHMARK_RRF_K": "42", "BENCHMARK_TOP_K": "20"}, clear=True):
            self.assertEqual(config_from_environment(), RetrievalConfig("hybrid_rrf", 30, 42, 20))

    def test_numeric_error_sanitized(self):
        with patch.dict(os.environ, {"BENCHMARK_CANDIDATE_K": "PRIVATE_TEXT"}, clear=True):
            with self.assertRaises(BenchmarkConfigurationError) as caught:
                config_from_environment()
        self.assertNotIn("PRIVATE_TEXT", str(caught.exception))

    def test_candidate_recalls_and_short_corpus(self):
        metrics = candidate_metrics(["n", "p", "q"], ("p", "q"), effective_k=50, corpus_size=100)
        self.assertEqual(metrics["Candidate Recall@10"], 1)
        self.assertEqual(metrics["Candidate Recall@20"], 1)
        self.assertEqual(metrics["Candidate Recall@50"], 1)
        self.assertEqual(candidate_metrics(["p"], ("p", "q"), effective_k=2, corpus_size=2)["Candidate Recall@50"], .5)

    def test_unrequested_candidate_depth_is_na(self):
        metrics = candidate_metrics(["p"], ("p",), effective_k=10, corpus_size=100)
        self.assertEqual(metrics["Candidate Recall@10"], 1)
        self.assertIsNone(metrics["Candidate Recall@20"])
        self.assertIsNone(metrics["Candidate Recall@50"])

    def test_candidate_recall_cutoffs_with_multiple_positives(self):
        metrics = candidate_metrics([str(i) for i in range(50)], ("15", "40"), effective_k=50, corpus_size=100)
        self.assertEqual(metrics["Candidate Recall@10"], 0)
        self.assertEqual(metrics["Candidate Recall@20"], .5)
        self.assertEqual(metrics["Candidate Recall@50"], 1)

    def test_rrf_scores_and_existing_algorithm_match(self):
        rankings = ([0, 1, 2], [2, 1, 3])
        scores = reciprocal_rank_fusion_scores(rankings, k=60)
        self.assertAlmostEqual(scores[1], 2 / 62)
        self.assertEqual(set(scores), {0, 1, 2, 3})
        self.assertEqual(reciprocal_rank_fusion(rankings, limit=4), reciprocal_rank_fusion(rankings, limit=4))


class CodeOverrideTests(unittest.TestCase):
    def test_docker_overrides_without_git(self):
        with patch.dict(os.environ, {"BENCHMARK_CODE_COMMIT_SHA": "A" * 40, "BENCHMARK_CODE_DIRTY": "false"}, clear=True), \
                patch("knowledge_base.retrieval_benchmark.subprocess.run", side_effect=OSError) as git:
            self.assertEqual(code_revision(), ("a" * 40, False))
            git.assert_not_called()

    def test_partial_override_and_git_unavailable(self):
        with patch.dict(os.environ, {"BENCHMARK_CODE_DIRTY": "true"}, clear=True), \
                patch("knowledge_base.retrieval_benchmark.subprocess.run", side_effect=OSError):
            self.assertEqual(code_revision(), (None, True))

    def test_auto_git_fallback(self):
        from types import SimpleNamespace
        with patch.dict(os.environ, {}, clear=True), patch("knowledge_base.retrieval_benchmark.subprocess.run",
                side_effect=[SimpleNamespace(stdout="a" * 40 + "\n"), SimpleNamespace(stdout="")]):
            self.assertEqual(code_revision(), ("a" * 40, False))

    def test_invalid_sha_or_dirty_rejected(self):
        for env in ({"BENCHMARK_CODE_COMMIT_SHA": "da02ec0"}, {"BENCHMARK_CODE_COMMIT_SHA": "g" * 40},
                    {"BENCHMARK_CODE_COMMIT_SHA": ""}, {"BENCHMARK_CODE_DIRTY": "TRUE"},
                    {"BENCHMARK_CODE_DIRTY": "0"}, {"BENCHMARK_CODE_DIRTY": "PRIVATE_TEXT"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), self.assertRaises(BenchmarkConfigurationError):
                code_revision()


class CandidateBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus, self.gold = self.root / "corpus.json", self.root / "gold.json"
        self.corpus.write_text(json.dumps({"chunks": [{"id": f"id-{i}", "text": "уникальный термин" if i == 55 else "нейтральный текст",
            "metadata": {"chapter": "synthetic chapter", "article_title": "synthetic article"}} for i in range(60)]}), encoding="utf-8")
        self.gold.write_text(json.dumps({"queries": [{"text": "уникальный термин", "positive_chunk_ids": ["id-55"],
            "hard_negative_chunk_ids": ["id-59"], "metadata": {"answer_hint": "PRIVATE_HINT"}}]}), encoding="utf-8")
        self.inputs = load_inputs(self.corpus, self.gold)

    def test_dense_default_preserves_embedding_inputs_and_full_diagnostics(self):
        provider = PositionProvider()
        result = run_benchmark(provider, self.inputs)
        self.assertEqual(provider.documents, [tuple(c.text for c in self.inputs.chunks)])
        self.assertEqual(provider.queries, [self.inputs.queries[0].text])
        self.assertEqual(result["per_query"][0]["retrieved_chunk_numbers"], list(range(10)))
        self.assertEqual(result["summary"]["positive-over-hard-negative"], 1)
        self.assertEqual(result["experiment"]["retrieval_mode"], "dense")
        self.assertEqual(result["experiment"]["document_representation"], "plain")

    def test_bm25_russian_ranking_and_stable_zero_ties(self):
        result = run_benchmark(None, self.inputs, retrieval_mode="bm25")
        self.assertEqual(result["per_query"][0]["retrieved_chunk_numbers"], [55] + list(range(9)))
        self.assertEqual(result["summary"]["Recall@1"], 1)
        self.assertEqual(result["summary"]["Candidate Recall@50"], 1)
        self.assertEqual(result["experiment"]["bm25_k1"], 1.5)
        self.assertEqual(result["experiment"]["bm25_b"], .75)
        self.assertIsNone(result["experiment"]["provider"])

    def test_dense_legacy_metrics_at_known_positive_rank(self):
        from dataclasses import replace
        from knowledge_base.benchmark_models import BenchmarkCase
        inputs = replace(self.inputs, queries=(BenchmarkCase("query", ("id-4",), ("id-59",)),))
        result = run_benchmark(PositionProvider(), inputs)
        for key, expected in (("Recall@1", 0), ("Recall@3", 0), ("Recall@5", 1), ("Recall@10", 1),
                              ("MRR", .2), ("nDCG", 1 / math.log2(6)), ("positive-over-hard-negative", 1)):
            self.assertAlmostEqual(result["summary"][key], expected)

    def test_bm25_never_uses_passed_embedding_provider(self):
        provider = PositionProvider()
        run_benchmark(provider, self.inputs, retrieval_mode="bm25")
        self.assertEqual(provider.documents, [])
        self.assertEqual(provider.queries, [])

    def test_hybrid_candidate_union_fusion(self):
        first = run_benchmark(PositionProvider(), self.inputs, retrieval_mode="hybrid_rrf", candidate_k=10)
        second = run_benchmark(PositionProvider(), self.inputs, retrieval_mode="hybrid_rrf", candidate_k=10)
        expected = reciprocal_rank_fusion((list(range(10)), [55] + list(range(9))), limit=10)
        self.assertEqual(first["per_query"][0]["retrieved_chunk_numbers"], expected)
        self.assertEqual(first["per_query"][0]["candidate_union_size"], 11)
        self.assertEqual(first["per_query"][0]["candidate_pool_size"], 10)
        self.assertEqual(first["summary"]["Candidate Recall@10"], 1)
        self.assertEqual(first["per_query"][0]["retrieved_chunk_numbers"], second["per_query"][0]["retrieved_chunk_numbers"])

    def test_candidate_metrics_before_optional_reranker(self):
        class Reverse:
            name = "fake_reverse_v1"
            def rerank(self, query, candidates):
                return tuple(reversed(candidates))
        first = run_benchmark(None, self.inputs, retrieval_mode="bm25")
        second = run_benchmark(None, self.inputs, retrieval_mode="bm25", reranker=Reverse())
        self.assertEqual(first["summary"]["Candidate Recall@pool"], second["summary"]["Candidate Recall@pool"])
        self.assertNotEqual(first["summary"]["Recall@1"], second["summary"]["Recall@1"])
        self.assertEqual(second["experiment"]["reranker"], "fake_reverse_v1")

    def test_reranker_cannot_inject_new_candidate(self):
        class Bad:
            name = "fake_bad"
            def rerank(self, query, candidates):
                return (Candidate("outside", candidates[0].chunk, 1.),) + candidates[1:]
        result = run_benchmark(None, self.inputs, retrieval_mode="bm25", reranker=Bad())
        self.assertEqual(result["experiment"]["status"], "incomplete")
        self.assertEqual(result["errors"][0]["code"], "retrieval_failed")

    def test_hashes_and_dataset_unchanged_all_modes(self):
        before = (self.corpus.read_bytes(), self.gold.read_bytes())
        for mode in ("dense", "bm25", "hybrid_rrf"):
            result = run_benchmark(PositionProvider() if mode != "bm25" else None, self.inputs, retrieval_mode=mode,
                                   document_representation="structure_aware_v1")
            self.assertEqual(result["experiment"]["corpus_hash"], self.inputs.corpus_hash)
            self.assertEqual(result["experiment"]["gold_set_hash"], self.inputs.gold_hash)
        self.assertEqual(before, (self.corpus.read_bytes(), self.gold.read_bytes()))

    def test_bad_config_fails_before_embedding(self):
        provider = PositionProvider()
        for kwargs in ({"candidate_k": 9}, {"retrieval_mode": "unknown"}, {"rrf_k": 0}):
            with self.assertRaises(BenchmarkConfigurationError):
                run_benchmark(provider, self.inputs, **kwargs)
        self.assertEqual(provider.documents, [])
        with patch.dict(os.environ, {"BENCHMARK_CODE_DIRTY": "bad"}, clear=True), self.assertRaises(BenchmarkConfigurationError):
            run_benchmark(provider, self.inputs)
        self.assertEqual(provider.documents, [])

    def test_metadata_and_content_free_reports(self):
        for mode in ("dense", "bm25", "hybrid_rrf"):
            result = run_benchmark(PositionProvider() if mode != "bm25" else None, self.inputs, retrieval_mode=mode)
            directory = write_report(result, self.root / "reports")
            metadata = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
            for field in ("retrieval_mode", "document_representation", "candidate_k", "final_top_k", "bm25_k1", "bm25_b", "rrf_k"):
                self.assertIn(field, metadata)
                self.assertIn(field, (directory / "REPORT.md").read_text(encoding="utf-8"))
            for path in directory.iterdir():
                self.assertNotIn("PRIVATE_HINT", path.read_text(encoding="utf-8"))
                self.assertNotIn("уникальный термин", path.read_text(encoding="utf-8"))

    def test_bm25_cli_skips_provider_and_invalid_config_blocks_factory(self):
        from scripts.eval_retrieval import main
        args = ["eval_retrieval.py", "--corpus", str(self.corpus), "--gold", str(self.gold)]
        with patch.dict(os.environ, {"BENCHMARK_RETRIEVAL_MODE": "bm25"}, clear=True), patch("sys.argv", args), \
                patch("sys.stdout", new_callable=io.StringIO), patch("scripts.eval_retrieval.provider_from_environment") as factory, \
                patch("scripts.eval_retrieval.write_report", return_value=self.root / "fake"):
            self.assertEqual(main(), 0)
            factory.assert_not_called()
        with patch.dict(os.environ, {"BENCHMARK_CANDIDATE_K": "5"}, clear=True), patch("sys.argv", args), \
                patch("sys.stdout", new_callable=io.StringIO), patch("scripts.eval_retrieval.provider_from_environment") as factory:
            self.assertEqual(main(), 1)
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
