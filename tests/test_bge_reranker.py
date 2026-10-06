from contextlib import nullcontext
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from knowledge_base.benchmark_config import BenchmarkConfigurationError
from knowledge_base.bge_reranker import (
    BGEReranker, BGERerankerConfig, RerankerRuntimeError, _TransformersBackend, reranker_from_environment,
)
from knowledge_base.candidate_retrieval import Candidate, CandidateResult, rerank_candidates
from knowledge_base.retrieval_benchmark import load_inputs, run_benchmark, write_report
from knowledge_base.report_comparison import compare_reports
from test_retriever import make_chunk


class FakeBackend:
    device = "cpu"
    revision = "a" * 40

    def __init__(self, scores):
        self.scores, self.calls = iter(scores), []

    def score_pairs(self, pairs, *, max_length):
        self.calls.append((pairs, max_length))
        return [next(self.scores) for _ in pairs]


def pool(count=3):
    return tuple(Candidate(f"c{i}", make_chunk(i, f"PRIVATE_PASSAGE_{i}"), float(i)) for i in range(count))


class BGERerankerTests(unittest.TestCase):
    def test_default_none_ignores_optional_dependencies(self):
        with patch.dict(os.environ, {}, clear=True), patch("knowledge_base.bge_reranker._load_backend") as load:
            self.assertIsNone(reranker_from_environment())
            load.assert_not_called()

    def test_configuration_creates_lazy_adapter(self):
        with patch.dict(os.environ, {"BENCHMARK_RERANKER": "bge-reranker-v2-m3"}, clear=True), \
                patch("knowledge_base.bge_reranker._load_backend") as load:
            adapter = reranker_from_environment()
            self.assertEqual(adapter.config.batch_size, 4)
            self.assertEqual(adapter.config.max_length, 512)
            load.assert_not_called()

    def test_revision_metadata_never_invents_commit(self):
        for revision, expected in (("a" * 40, "a" * 40), (None, None), ("PRIVATE_PATH", None)):
            metadata = BGEReranker(backend=NS(revision=revision)).metadata
            self.assertEqual(metadata["reranker_requested_revision"], "main")
            self.assertEqual(metadata["reranker_resolved_revision"], expected)
        pinned = BGEReranker(BGERerankerConfig(revision="b" * 40), backend=NS(revision=None)).metadata
        self.assertEqual(pinned["reranker_requested_revision"], "b" * 40)
        self.assertIsNone(pinned["reranker_resolved_revision"])

    def test_runtime_import_failure_is_sanitized(self):
        from knowledge_base.bge_reranker import _load_backend
        original_import = __import__
        def broken(name, *args, **kwargs):
            if name == "torch":
                raise OSError("PRIVATE_LOCAL_PATH DLL failure")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=broken), self.assertRaises(RerankerRuntimeError) as caught:
            _load_backend(BGERerankerConfig())
        self.assertIn("runtime import failed", str(caught.exception))
        self.assertNotIn("PRIVATE", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)

    def test_initialization_failure_is_sanitized(self):
        from knowledge_base.bge_reranker import _load_backend
        modules = {"torch": NS(), "transformers": NS(AutoTokenizer=Mock(), AutoModelForSequenceClassification=Mock())}
        with patch.dict("sys.modules", modules), \
                patch("knowledge_base.bge_reranker._TransformersBackend", side_effect=RuntimeError("PRIVATE_PATH")), \
                self.assertRaises(RerankerRuntimeError) as caught:
            _load_backend(BGERerankerConfig())
        self.assertIn("initialization failed", str(caught.exception))
        self.assertNotIn("PRIVATE", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)

    def test_explicit_cuda_unavailable_does_not_load_model(self):
        torch = NS(cuda=NS(is_available=lambda: False))
        tokenizer_class, model_class = Mock(), Mock()
        with self.assertRaises(ValueError):
            _TransformersBackend(BGERerankerConfig(device="cuda"), torch, tokenizer_class, model_class)
        tokenizer_class.from_pretrained.assert_not_called()
        model_class.from_pretrained.assert_not_called()

    def test_invalid_config_safe_errors(self):
        for kwargs in ({"device": "remote"}, {"batch_size": 0}, {"batch_size": True}, {"max_length": 0},
                       {"max_length": 8193}, {"model": "PRIVATE_MODEL"}, {"revision": "PRIVATE_REVISION"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(BenchmarkConfigurationError) as caught:
                BGERerankerConfig(**kwargs)
            self.assertNotIn("PRIVATE", str(caught.exception))
        for env in ({"BENCHMARK_RERANKER": "unknown"},
                    {"BENCHMARK_RERANKER": "bge-reranker-v2-m3", "BGE_RERANKER_BATCH_SIZE": "PRIVATE_VALUE"}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(BenchmarkConfigurationError) as caught:
                reranker_from_environment()
            self.assertNotIn("PRIVATE_VALUE", str(caught.exception))

    def test_pairs_batching_scores_and_permutation(self):
        backend = FakeBackend([.1, 3., -.5])
        adapter = BGEReranker(BGERerankerConfig(batch_size=2), backend=backend)
        candidates = pool()
        result = adapter.rerank("PRIVATE_QUERY", candidates)
        self.assertEqual([c.chunk_id for c in result], ["c1", "c0", "c2"])
        self.assertEqual(result[0].score, 3.)
        self.assertEqual(backend.calls, [([["PRIVATE_QUERY", "PRIVATE_PASSAGE_0"], ["PRIVATE_QUERY", "PRIVATE_PASSAGE_1"]], 512),
                                         ([["PRIVATE_QUERY", "PRIVATE_PASSAGE_2"]], 512)])
        self.assertEqual({c.chunk_id for c in result}, {c.chunk_id for c in candidates})
        for item in result:
            self.assertIs(item.chunk, candidates[int(item.chunk_id[1:])].chunk)
        self.assertEqual(candidates[0].score, 0.)

    def test_stable_ties_preserve_candidate_order(self):
        candidates = tuple(reversed(pool()))
        result = BGEReranker(backend=FakeBackend([1., 1., 1.])).rerank("query", candidates)
        self.assertEqual([c.chunk_id for c in result], [c.chunk_id for c in candidates])

    def test_empty_pool_does_not_load(self):
        with patch("knowledge_base.bge_reranker._load_backend") as load:
            self.assertEqual(BGEReranker().rerank("query", ()), ())
            load.assert_not_called()

    def test_invalid_scores_and_counts_safe_errors(self):
        for values in ([float("nan")] * 3, [float("inf")] * 3, ["PRIVATE_QUERY"] * 3, []):
            backend = NS(score_pairs=Mock(return_value=values))
            with self.assertRaisesRegex(RerankerRuntimeError, "^Local reranker scoring failed$") as caught:
                BGEReranker(backend=backend).rerank("PRIVATE_QUERY", pool())
            self.assertIsNone(caught.exception.__context__)

    def test_model_exception_does_not_leak(self):
        backend = NS(score_pairs=Mock(side_effect=RuntimeError("PRIVATE_QUERY PRIVATE_PASSAGE PRIVATE_HINT")))
        try:
            BGEReranker(backend=backend).rerank("PRIVATE_QUERY", pool())
        except RerankerRuntimeError as error:
            self.assertIsNone(error.__context__)
            self.assertNotIn("PRIVATE_PASSAGE", traceback.format_exc())
            self.assertNotIn("PRIVATE_HINT", str(error))
        else:
            self.fail("Expected safe scoring error")

    def test_missing_optional_dependency_safe_error(self):
        original_import = __import__
        def missing(name, *args, **kwargs):
            if name in ("torch", "transformers"):
                raise ImportError("PRIVATE_LOCAL_PATH")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(RerankerRuntimeError, "requirements-reranker.txt") as caught:
                BGEReranker().prepare()
            self.assertIsNone(caught.exception.__context__)
            self.assertNotIn("PRIVATE_LOCAL_PATH", str(caught.exception))

    def test_load_once(self):
        with patch("knowledge_base.bge_reranker._load_backend", return_value=FakeBackend([1., 2., 3., 3., 2., 1.])) as load:
            adapter = BGEReranker()
            adapter.rerank("q", pool())
            adapter.rerank("q", pool())
            load.assert_called_once()

    def test_permutation_validation_rejects_drop_duplicate_injection_and_nan(self):
        candidates = pool()
        for changed in (candidates[:2], (candidates[0], candidates[0], candidates[2]),
                        (replace(candidates[0], chunk_id="outside"),) + candidates[1:],
                        (replace(candidates[0], score=float("nan")),) + candidates[1:]):
            with self.assertRaises(ValueError):
                rerank_candidates("q", CandidateResult(candidates, {}, 3), reranker=NS(rerank=lambda q, c: changed))

    def test_transformers_backend_mocked_device_and_model_calls(self):
        for available, requested, expected in ((False, "auto", "cpu"), (True, "auto", "cuda"), (True, "cpu", "cpu")):
            with self.subTest(available=available, requested=requested):
                torch = NS(cuda=NS(is_available=lambda: available), manual_seed=Mock(),
                           use_deterministic_algorithms=Mock(), backends=NS(cudnn=NS(benchmark=True)),
                           float32="float32", inference_mode=lambda: nullcontext())
                tensor = Mock()
                tensor.to.return_value = tensor
                logits = Mock(ndim=2, shape=(2, 1))
                logits.__getitem__ = Mock(return_value=logits)
                logits.float.return_value = logits
                logits.cpu.return_value = logits
                logits.tolist.return_value = [1., 2.]
                model = Mock()
                model.config = NS(_commit_hash="a" * 40)
                model.return_value = NS(logits=logits)
                tokenizer = Mock(return_value={"input_ids": tensor})
                model_class = NS(from_pretrained=Mock(return_value=model))
                tokenizer_class = NS(from_pretrained=Mock(return_value=tokenizer))
                backend = _TransformersBackend(BGERerankerConfig(device=requested), torch, tokenizer_class, model_class)
                self.assertEqual(backend.device, expected)
                model.to.assert_called_once_with(expected)
                model.eval.assert_called_once()
                self.assertFalse(model_class.from_pretrained.call_args.kwargs["trust_remote_code"])
                self.assertFalse(model_class.from_pretrained.call_args.kwargs["token"])
                self.assertTrue(model_class.from_pretrained.call_args.kwargs["use_safetensors"])
                self.assertEqual(tokenizer_class.from_pretrained.call_args.kwargs["revision"], "a" * 40)
                self.assertEqual(model_class.from_pretrained.call_args.kwargs["cache_dir"], ".model-cache/huggingface")
                self.assertEqual(tokenizer_class.from_pretrained.call_args.kwargs["cache_dir"], ".model-cache/huggingface")
                self.assertEqual(backend.score_pairs([["q", "p"], ["q", "p2"]], max_length=512), [1., 2.])
                self.assertTrue(tokenizer.call_args.kwargs["truncation"])
                self.assertEqual(tokenizer.call_args.kwargs["max_length"], 512)


class RerankerBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus, self.gold = self.root / "corpus.json", self.root / "gold.json"
        self.corpus.write_text(json.dumps({"chunks": [{"id": f"id-{i}", "text": f"PRIVATE_PASSAGE_{i}"} for i in range(25)]}))
        self.gold.write_text(json.dumps({"queries": [{"text": "PRIVATE_QUERY", "positive_chunk_ids": ["id-15"],
            "hard_negative_chunk_ids": ["id-0"], "metadata": {"answer_hint": "PRIVATE_HINT"}}]}))
        self.inputs = load_inputs(self.corpus, self.gold)

    def test_candidate_before_and_final_metrics_after_reranking(self):
        from test_candidate_retrieval import PositionProvider
        adapter = BGEReranker(backend=FakeBackend([0.] * 15 + [10.] + [0.] * 4))
        result = run_benchmark(PositionProvider(), self.inputs, candidate_k=20, reranker=adapter)
        self.assertEqual(result["summary"]["Candidate Recall@10"], 0)
        self.assertEqual(result["summary"]["Candidate Recall@20"], 1)
        self.assertEqual(result["summary"]["Recall@1"], 1)
        self.assertEqual(result["summary"]["MRR"], 1)
        self.assertEqual(result["summary"]["nDCG"], 1)
        self.assertEqual(result["summary"]["positive-over-hard-negative"], 1)
        self.assertEqual(len(result["per_query"][0]["retrieved_chunk_numbers"]), 10)
        self.assertEqual(result["per_query"][0]["retrieved_chunk_numbers"][0], 15)
        self.assertEqual(sum(len(call[0]) for call in adapter._backend.calls), 20)

    def test_post_rerank_hard_negative_can_lose_even_when_dense_wins(self):
        from test_candidate_retrieval import PositionProvider
        inputs = replace(self.inputs, queries=(replace(self.inputs.queries[0], positives=("id-0",), hard_negatives=("id-15",)),))
        plain = run_benchmark(PositionProvider(), inputs, candidate_k=20)
        ranked = run_benchmark(PositionProvider(), inputs, candidate_k=20,
                               reranker=BGEReranker(backend=FakeBackend([0.] * 15 + [10.] + [0.] * 4)))
        self.assertEqual(plain["summary"]["positive-over-hard-negative"], 1)
        self.assertEqual(ranked["summary"]["positive-over-hard-negative"], 0)

    def test_missing_hard_negative_is_na_not_win(self):
        from test_candidate_retrieval import PositionProvider
        inputs = replace(self.inputs, queries=(replace(self.inputs.queries[0], hard_negatives=("id-24",)),))
        result = run_benchmark(PositionProvider(), inputs, candidate_k=20,
                               reranker=BGEReranker(backend=FakeBackend([0.] * 15 + [10.] + [0.] * 4)))
        self.assertIsNone(result["summary"]["positive-over-hard-negative"])
        self.assertEqual(result["experiment"]["hard_negative_evaluable_query_count"], 0)

    def test_report_metadata_timing_privacy_and_comparison(self):
        from test_candidate_retrieval import PositionProvider
        result = run_benchmark(PositionProvider(), self.inputs, candidate_k=20,
                               reranker=BGEReranker(backend=FakeBackend([0.] * 15 + [10.] + [0.] * 4)))
        metadata = result["experiment"]
        for key in ("reranker", "reranker_model", "reranker_device", "reranker_candidate_k", "reranker_batch_size",
                    "reranker_max_length", "reranker_latency_seconds"):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["reranker_device"], "cpu")
        self.assertEqual(metadata["reranker_candidate_k"], 20)
        self.assertEqual(metadata["reranker_requested_revision"], "main")
        self.assertEqual(metadata["reranker_resolved_revision"], "a" * 40)
        self.assertGreaterEqual(metadata["reranker_latency_seconds"], 0)
        self.assertIsNotNone(result["summary"]["reranker p50 latency seconds"])
        directory = write_report(result, self.root / "reports")
        for path in directory.iterdir():
            text = path.read_text(encoding="utf-8")
            for forbidden in ("PRIVATE_QUERY", "PRIVATE_PASSAGE", "PRIVATE_HINT"):
                self.assertNotIn(forbidden, text)
        compare_reports([directory], self.root / "comparison")
        self.assertIn("reranker_model", (self.root / "comparison" / "COMPARISON.md").read_text(encoding="utf-8"))

    def test_scoring_failure_is_content_free_report(self):
        from test_candidate_retrieval import PositionProvider
        backend = FakeBackend([])
        backend.score_pairs = Mock(side_effect=RuntimeError("PRIVATE_QUERY PRIVATE_HINT PRIVATE_PASSAGE"))
        result = run_benchmark(PositionProvider(), self.inputs, candidate_k=20, reranker=BGEReranker(backend=backend))
        self.assertEqual(result["experiment"]["status"], "incomplete")
        self.assertEqual(result["summary"]["Candidate Recall@20"], 1)
        directory = write_report(result, self.root / "reports")
        for filename in ("REPORT.md", "experiment.json", "errors.csv"):
            self.assertNotIn("PRIVATE_HINT", (directory / filename).read_text(encoding="utf-8"))

    def test_unknown_resolved_revision_report_is_na(self):
        from test_candidate_retrieval import PositionProvider
        backend = FakeBackend([0.] * 20)
        backend.revision = None
        result = run_benchmark(PositionProvider(), self.inputs, candidate_k=20,
                               reranker=BGEReranker(backend=backend))
        self.assertIsNone(result["experiment"]["reranker_resolved_revision"])
        directory = write_report(result, self.root / "reports")
        self.assertIn("| reranker_resolved_revision | N/A |", (directory / "REPORT.md").read_text(encoding="utf-8"))
        compare_reports([directory], self.root / "comparison")
        self.assertIn("reranker_resolved_revision", (self.root / "comparison" / "comparison.csv").read_text(encoding="utf-8"))

    def test_cli_none_and_missing_optional_runtime(self):
        from scripts.eval_retrieval import main
        from test_candidate_retrieval import PositionProvider
        args = ["eval_retrieval.py", "--corpus", str(self.corpus), "--gold", str(self.gold)]
        with patch.dict(os.environ, {}, clear=True), patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO), \
                patch("scripts.eval_retrieval.provider_from_environment", return_value=PositionProvider()), \
                patch("scripts.eval_retrieval.write_report", return_value=self.root / "fake") as report:
            self.assertEqual(main(), 0)
            self.assertEqual(report.call_args.args[0]["experiment"]["reranker"], "none")
        with patch.dict(os.environ, {"BENCHMARK_RERANKER": "bge-reranker-v2-m3"}, clear=True), \
                patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO) as output, \
                patch("scripts.eval_retrieval.provider_from_environment") as provider, \
                patch("knowledge_base.bge_reranker._load_backend", side_effect=RerankerRuntimeError("Optional dependencies unavailable")):
            self.assertEqual(main(), 1)
            provider.assert_not_called()
            self.assertIn("Optional dependencies unavailable", output.getvalue())


if __name__ == "__main__":
    unittest.main()
