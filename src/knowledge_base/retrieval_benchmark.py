"""Provider-neutral retrieval evaluation and content-free experiment artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, isfinite, log2
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from knowledge_base.chunker import Chunk
from knowledge_base.benchmark_models import BenchmarkCase
from knowledge_base.benchmark_config import RetrievalConfig, validate_code_overrides
from knowledge_base.candidate_retrieval import CandidateRetriever, Reranker, candidate_metrics, rerank_candidates
from knowledge_base.document_representation import document_embedding_text, validate_document_representation
from knowledge_base.embeddings import EmbeddingProvider, embedding_identity
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import InMemoryVectorStore

NOTICE = "Это retrieval benchmark, не оценка качества LLM-ответа."


# Backward-compatible public name for the format-independent case model.
GoldQuery = BenchmarkCase


@dataclass(frozen=True)
class BenchmarkInputs:
    chunk_ids: tuple[str, ...]
    chunks: tuple[Chunk, ...]
    queries: tuple[GoldQuery, ...]
    corpus_hash: str
    gold_hash: str
    document_metadata: tuple[dict[str, str], ...] = ()


def load_inputs(corpus: Path, gold: Path) -> BenchmarkInputs:
    """Hash exact file bytes. No paths, arbitrary metadata or content in output."""
    corpus_bytes, gold_bytes = corpus.read_bytes(), gold.read_bytes()
    rows = json.loads(corpus_bytes)["chunks"]
    query_rows = json.loads(gold_bytes)["queries"]
    ids = tuple(row["id"] for row in rows)
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Corpus requires unique nonempty string chunk IDs")
    if any(not isinstance(row["text"], str) or not row["text"].strip() for row in rows):
        raise ValueError("Corpus requires nonempty text")
    chunks = tuple(Chunk(
        document_id=row.get("document_id", f"chunk-{i}"), version_id="snapshot", chunk_index=i,
        text=row["text"], source_file=row.get("metadata", {}).get("source_file", ""),
        source_format="txt", block_indices=(), page_numbers=(), paragraph_indices=(),
        section_titles=tuple(value for value in (
            row.get("metadata", {}).get("chapter", ""), row.get("metadata", {}).get("article_title", "")
        ) if value),
    ) for i, row in enumerate(rows))
    queries = []
    for row in query_rows:
        positives = row["positive_chunk_ids"]
        negatives = row.get("hard_negative_chunk_ids", [])
        if (not isinstance(row["text"], str) or not row["text"].strip()
                or not isinstance(positives, list) or not positives
                or not isinstance(negatives, list)
                or any(not isinstance(i, str) or not i for i in positives + negatives)
                or len(set(positives)) != len(positives)
                or len(set(negatives)) != len(negatives)
                or set(positives) & set(negatives)):
            raise ValueError("Invalid gold query or relevance references")
        queries.append(BenchmarkCase(row["text"], tuple(positives), tuple(negatives),
                                     row.get("query_id", ""), row.get("difficulty", ""),
                                     row.get("metadata", {})))
    if not queries:
        raise ValueError("Gold set requires queries")
    return BenchmarkInputs(ids, chunks, tuple(queries), hashlib.sha256(corpus_bytes).hexdigest(),
                           hashlib.sha256(gold_bytes).hexdigest(),
                           tuple(dict(row.get("metadata", {})) for row in rows))


def ranking_metrics(ranked: list[str], positives: tuple[str, ...]) -> dict:
    positive = set(positives)
    metrics = {f"Recall@{k}": len(positive & set(ranked[:k])) / len(positive) for k in (1, 3, 5, 10)}
    metrics["Top-1 accuracy"] = float(bool(ranked) and ranked[0] in positive)
    metrics["MRR"] = next((1 / rank for rank, ref in enumerate(ranked, 1) if ref in positive), 0.0)
    dcg = sum(1 / log2(rank + 1) for rank, ref in enumerate(ranked, 1) if ref in positive)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(len(positive), len(ranked)) + 1))
    metrics["nDCG"] = dcg / ideal if ideal else 0.0
    return metrics


def _percentile(values: list[float], fraction: float) -> float | None:
    return sorted(values)[max(0, ceil(len(values) * fraction) - 1)] if values else None


def code_revision() -> tuple[str | None, bool | None]:
    sha_override, dirty_override = validate_code_overrides()
    if sha_override is not None and dirty_override is not None:
        return sha_override, dirty_override
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout)
        sha = sha if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", sha) else None
    except (OSError, subprocess.SubprocessError):
        sha, dirty = None, None
    return (sha_override if sha_override is not None else sha,
            dirty_override if dirty_override is not None else dirty)


def run_benchmark(provider: EmbeddingProvider | None, inputs: BenchmarkInputs, *, top_k: int = 10,
                  cost_per_million_tokens: float | None = None,
                  document_representation: str = "plain", retrieval_mode: str = "dense",
                  candidate_k: int = 50, rrf_k: int = 60, reranker: Reranker | None = None) -> dict:
    config = RetrievalConfig(retrieval_mode, candidate_k, rrf_k, top_k)
    validate_document_representation(document_representation)
    sha, dirty = code_revision()  # validate overrides before any embeddings
    if type(top_k) is not int or top_k < 10:
        raise ValueError("top_k must be at least 10 for Recall@10")
    if cost_per_million_tokens is not None and (not isfinite(cost_per_million_tokens) or cost_per_million_tokens < 0):
        raise ValueError("Invalid configured token cost")
    embedding_texts = None
    if document_representation != "plain":
        if inputs.document_metadata and len(inputs.document_metadata) != len(inputs.chunks):
            raise ValueError("Document metadata count does not match corpus chunks")
        embedding_texts = tuple(document_embedding_text(
            chunk.text, inputs.document_metadata[i] if inputs.document_metadata else {},
            mode=document_representation,
        ) for i, chunk in enumerate(inputs.chunks))
    use_dense = retrieval_mode != "bm25"
    if use_dense and provider is None:
        raise ValueError("Dense retrieval requires an embedding provider")
    identity = embedding_identity(provider) if use_dense else None
    reranker_name = reranker.name if reranker is not None else None
    if reranker_name is not None and (not isinstance(reranker_name, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", reranker_name)):
        raise ValueError("Invalid reranker name")
    if reranker is not None and callable(getattr(reranker, "prepare", None)):
        reranker.prepare()
    retriever = Retriever(provider, InMemoryVectorStore(dimension=provider.dimension)) if use_dense else None
    candidates = CandidateRetriever(chunks=inputs.chunks, chunk_ids=inputs.chunk_ids,
                                    config=config, dense=retriever, document_texts=embedding_texts)
    before = getattr(provider, "telemetry", {})
    errors, rows, latencies, reranker_latencies = [], [], [], []
    started = perf_counter()
    indexed = True
    try:
        candidates.index()
    except Exception:
        indexed = False
        errors.append({"query_number": None, "stage": "index", "code": "embedding_index_failed"})
    index_latency = perf_counter() - started
    known = set(inputs.chunk_ids)
    unresolved = 0
    for number, query in enumerate(inputs.queries, 1):
        missing = set(query.positives + query.hard_negatives) - known
        row = {"query_number": number, "status": "ok", "latency_seconds": None,
               "reranker_latency_seconds": None,
               "retrieved_chunk_numbers": [], "positive-over-hard-negative": None,
               "candidate_pool_size": 0, "candidate_union_size": 0}
        row.update(candidate_metrics([], query.positives, effective_k=candidates.limit, corpus_size=len(inputs.chunks)))
        if missing:
            unresolved += len(missing)
            row["status"] = "unresolved_refs"
            errors.append({"query_number": number, "stage": "references", "code": "unresolved_refs"})
        else:
            ranked = []
            if indexed:
                started = perf_counter()
                try:
                    pool = candidates.retrieve(query.text)
                    pool_ids = [c.chunk_id for c in pool.candidates]
                    row.update(candidate_metrics(pool_ids, query.positives, effective_k=candidates.limit,
                                                 corpus_size=len(inputs.chunks)))
                    row["candidate_pool_size"] = len(pool.candidates)
                    row["candidate_union_size"] = pool.union_size
                    if reranker is not None:
                        rerank_started = perf_counter()
                        try:
                            reranked = rerank_candidates(query.text, pool, reranker=reranker)
                        finally:
                            row["reranker_latency_seconds"] = perf_counter() - rerank_started
                            reranker_latencies.append(row["reranker_latency_seconds"])
                    else:
                        reranked = pool.candidates
                    final = reranked[:top_k]
                    ranked = [c.chunk_id for c in final]
                    row["retrieved_chunk_numbers"] = [c.chunk.chunk_index for c in final]
                    if query.hard_negatives:
                        scores = {c.chunk_id: c.score for c in reranked} if reranker is not None else pool.diagnostic_scores
                        positives = [scores[i] for i in query.positives if i in scores]
                        if not positives:
                            row["positive-over-hard-negative"] = 0.0
                        elif all(i in scores for i in query.hard_negatives):
                            row["positive-over-hard-negative"] = float(
                                max(positives) > max(scores[i] for i in query.hard_negatives))
                except Exception:
                    row["status"] = "error"
                    errors.append({"query_number": number, "stage": "query", "code": "retrieval_failed"})
                finally:
                    row["latency_seconds"] = perf_counter() - started
                    latencies.append(row["latency_seconds"])
            else:
                row["status"] = "index_failed"
            row.update(ranking_metrics(ranked, query.positives))
            if row["status"] != "ok" and query.hard_negatives:
                row["positive-over-hard-negative"] = 0.0
        rows.append(row)
    eligible = [row for row in rows if row["status"] != "unresolved_refs"]
    metrics = {key: sum(row[key] for row in eligible) / len(eligible) if eligible else None
               for key in ranking_metrics([], ("placeholder",))}
    for key in candidate_metrics([], ("placeholder",), effective_k=candidates.limit, corpus_size=len(inputs.chunks)):
        values = [row[key] for row in eligible if row[key] is not None]
        metrics[key] = sum(values) / len(values) if values else None
    hn = [r["positive-over-hard-negative"] for r in eligible if r["positive-over-hard-negative"] is not None]
    metrics["positive-over-hard-negative"] = sum(hn) / len(hn) if hn else None
    metrics.update({"unresolved refs": unresolved, "p50 latency seconds": _percentile(latencies, .5),
                    "p95 latency seconds": _percentile(latencies, .95),
                    "reranker p50 latency seconds": _percentile(reranker_latencies, .5),
                    "reranker p95 latency seconds": _percentile(reranker_latencies, .95)})
    after = getattr(provider, "telemetry", {})
    usage = {}
    for key in ("total_tokens", "prompt_tokens", "api_calls", "error_count", "latency_seconds"):
        value = after.get(key)
        usage[key] = value - (before.get(key) or 0) if isinstance(value, (int, float)) else None
    tokens = usage["total_tokens"]
    return {
        "experiment": {
            "schema_version": 3, "provider": identity.provider if identity else None,
            "model_name": identity.model_name if identity else None,
            "retrieval_mode": retrieval_mode, "candidate_k": candidates.limit,
            "candidate_k_requested": candidate_k, "final_top_k": top_k,
            "bm25_k1": 1.5 if retrieval_mode != "dense" else None,
            "bm25_b": .75 if retrieval_mode != "dense" else None,
            "rrf_k": rrf_k if retrieval_mode == "hybrid_rrf" else None,
            "reranker": reranker_name or "none",
            "reranker_candidate_k": candidates.limit if reranker is not None else None,
            "reranker_latency_seconds": sum(reranker_latencies) if reranker is not None else None,
            **{key: getattr(reranker, "metadata", {}).get(key) for key in (
                "reranker_model", "reranker_device", "reranker_batch_size", "reranker_max_length",
                "reranker_revision", "reranker_revision_resolved", "reranker_load_latency_seconds",
                "reranker_requested_revision", "reranker_resolved_revision",
                "reranker_score_type", "reranker_precision")},
            "hard_negative_evaluable_query_count": len(hn),
            "hard_negative_scope": "reranker_candidate_pool" if reranker is not None else (
                "dense_full_corpus" if retrieval_mode == "dense" else (
                    "bm25_full_corpus" if retrieval_mode == "bm25" else "hybrid_candidate_union")),
            "document_representation": document_representation,
            "dimensions": identity.dimensions if identity else None,
            "distance_metric": "cosine" if use_dense else None,
            "normalization": "cosine norm division at search; stored vectors unchanged" if use_dense else None,
            "corpus_hash": inputs.corpus_hash, "gold_set_hash": inputs.gold_hash,
            "top_k": top_k, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(), "code_commit_sha": sha, "code_dirty": dirty,
            "code_commit_sha_source": "environment_override" if "BENCHMARK_CODE_COMMIT_SHA" in os.environ else ("git" if sha else "unavailable"),
            "code_dirty_source": "environment_override" if "BENCHMARK_CODE_DIRTY" in os.environ else ("git" if dirty is not None else "unavailable"),
            "chunk_count": len(inputs.chunks), "query_count": len(rows),
            "evaluable_query_count": len(eligible), "successful_query_count": sum(r["status"] == "ok" for r in rows),
            "index_latency_seconds": index_latency, "query_latency_seconds": sum(latencies),
            "token_usage": usage, "error_count": len(errors),
            "cost_per_million_tokens": cost_per_million_tokens,
            "estimated_cost": tokens * cost_per_million_tokens / 1_000_000
                if tokens is not None and cost_per_million_tokens is not None else None,
            "status": "complete" if indexed and not errors else "incomplete",
        }, "summary": metrics, "per_query": rows, "errors": errors,
    }


def write_report(result: dict, root: Path = Path("reports")) -> Path:
    """Only accepts the controlled result produced by run_benchmark."""
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    directory = root / experiment_id
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(result["experiment"], experiment_id=experiment_id)
    (directory / "experiment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for filename, rows, fields in (
        ("summary.csv", [{"metric": k, "value": v} for k, v in result["summary"].items()], ["metric", "value"]),
        ("per_query.csv", result["per_query"], ["query_number", "status", "latency_seconds", "reranker_latency_seconds", "retrieved_chunk_numbers",
            "candidate_pool_size", "candidate_union_size", "Candidate Recall@10", "Candidate Recall@20",
            "Candidate Recall@50", "Candidate Recall@pool",
            "Recall@1", "Recall@3", "Recall@5", "Recall@10", "Top-1 accuracy", "MRR", "nDCG", "positive-over-hard-negative"]),
        ("errors.csv", result["errors"], ["query_number", "stage", "code"]),
    ):
        with (directory / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Retrieval benchmark", "", NOTICE, "", "## Experiment", "",
             "| Field | Value |", "| --- | --- |"]
    lines.extend(f"| {k} | {v if v is not None else 'N/A'} |" for k, v in metadata.items())
    lines.extend(["", "## Metrics", "", "| Metric | Value |", "| --- | --- |"])
    lines.extend(f"| {k} | {v if v is not None else 'N/A'} |" for k, v in result["summary"].items())
    lines.extend(["", "MRR and binary nDCG use top_k. Recall is macro-averaged across queries.",
                  "Unresolved-reference queries are excluded; failed retrievals count as zero.",
                  "Positive-over-hard-negative: best positive score strictly exceeds best hard-negative score; ties fail.",
                  "Candidate Recall is measured before reranking; hybrid pool is the top candidate_k RRF union.",
                  "Candidate Recall@k is N/A when k exceeds the requested pool and unsearched corpus remains.",
                  "Without reranker: dense/BM25 hard-negative diagnostics use full-corpus scores; hybrid uses RRF scores (absent=0).",
                  "With reranker: hard-negative scores are post-rerank within the full candidate pool, before final top_k truncation.",
                  "Absent all positives count as zero; absent hard-negative scores otherwise mean N/A, not success.",
                  "Latency includes candidate retrieval and optional reranking; percentiles use nearest rank.",
                  "Token usage covers indexing and queries. N/A means unavailable, not zero.",
                  "Corpus and gold snapshots are identified by SHA-256 of exact input bytes."])
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory
