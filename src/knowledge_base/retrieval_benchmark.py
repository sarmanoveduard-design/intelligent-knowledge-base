"""Provider-neutral retrieval evaluation and content-free experiment artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
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
from knowledge_base.embeddings import EmbeddingProvider, embedding_identity
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import InMemoryVectorStore

NOTICE = "Это retrieval benchmark, не оценка качества LLM-ответа."


@dataclass(frozen=True)
class GoldQuery:
    text: str
    positives: tuple[str, ...]
    hard_negatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkInputs:
    chunk_ids: tuple[str, ...]
    chunks: tuple[Chunk, ...]
    queries: tuple[GoldQuery, ...]
    corpus_hash: str
    gold_hash: str


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
        document_id=f"chunk-{i}", version_id="snapshot", chunk_index=i,
        text=row["text"], source_file="", source_format="txt", block_indices=(),
        page_numbers=(), paragraph_indices=(), section_titles=(),
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
        queries.append(GoldQuery(row["text"], tuple(positives), tuple(negatives)))
    if not queries:
        raise ValueError("Gold set requires queries")
    return BenchmarkInputs(ids, chunks, tuple(queries), hashlib.sha256(corpus_bytes).hexdigest(),
                           hashlib.sha256(gold_bytes).hexdigest())


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
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout)
        return (sha if re.fullmatch(r"[0-9a-f]{40,64}", sha) else None), dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def run_benchmark(provider: EmbeddingProvider, inputs: BenchmarkInputs, *, top_k: int = 10,
                  cost_per_million_tokens: float | None = None) -> dict:
    if type(top_k) is not int or top_k < 10:
        raise ValueError("top_k must be at least 10 for Recall@10")
    if cost_per_million_tokens is not None and (not isfinite(cost_per_million_tokens) or cost_per_million_tokens < 0):
        raise ValueError("Invalid configured token cost")
    identity = embedding_identity(provider)
    retriever = Retriever(provider, InMemoryVectorStore(dimension=provider.dimension))
    before = getattr(provider, "telemetry", {})
    errors, rows, latencies = [], [], []
    started = perf_counter()
    indexed = True
    try:
        retriever.index_chunks(inputs.chunks)
    except Exception:
        indexed = False
        errors.append({"query_number": None, "stage": "index", "code": "embedding_index_failed"})
    index_latency = perf_counter() - started
    known = set(inputs.chunk_ids)
    unresolved = 0
    for number, query in enumerate(inputs.queries, 1):
        missing = set(query.positives + query.hard_negatives) - known
        row = {"query_number": number, "status": "ok", "latency_seconds": None,
               "retrieved_chunk_numbers": [], "positive-over-hard-negative": None}
        if missing:
            unresolved += len(missing)
            row["status"] = "unresolved_refs"
            errors.append({"query_number": number, "stage": "references", "code": "unresolved_refs"})
        else:
            ranked = []
            if indexed:
                started = perf_counter()
                try:
                    # Full ranking also evaluates hard negatives outside top_k.
                    results = retriever.search(query.text, top_k=max(top_k, len(inputs.chunks)))
                    full = [inputs.chunk_ids[r.chunk.chunk_index] for r in results]
                    ranked = full[:top_k]
                    row["retrieved_chunk_numbers"] = [r.chunk.chunk_index for r in results[:top_k]]
                    if query.hard_negatives:
                        scores = {inputs.chunk_ids[r.chunk.chunk_index]: r.score for r in results}
                        row["positive-over-hard-negative"] = float(
                            max(scores[i] for i in query.positives) > max(scores[i] for i in query.hard_negatives))
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
    hn = [r["positive-over-hard-negative"] for r in eligible if r["positive-over-hard-negative"] is not None]
    metrics["positive-over-hard-negative"] = sum(hn) / len(hn) if hn else None
    metrics.update({"unresolved refs": unresolved, "p50 latency seconds": _percentile(latencies, .5),
                    "p95 latency seconds": _percentile(latencies, .95)})
    after = getattr(provider, "telemetry", {})
    usage = {}
    for key in ("total_tokens", "prompt_tokens", "api_calls", "error_count", "latency_seconds"):
        value = after.get(key)
        usage[key] = value - (before.get(key) or 0) if isinstance(value, (int, float)) else None
    tokens = usage["total_tokens"]
    sha, dirty = code_revision()
    return {
        "experiment": {
            "schema_version": 1, "provider": identity.provider, "model_name": identity.model_name,
            "dimensions": identity.dimensions, "distance_metric": "cosine",
            "normalization": "cosine norm division at search; stored vectors unchanged",
            "corpus_hash": inputs.corpus_hash, "gold_set_hash": inputs.gold_hash,
            "top_k": top_k, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(), "code_commit_sha": sha, "code_dirty": dirty,
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
        ("per_query.csv", result["per_query"], ["query_number", "status", "latency_seconds", "retrieved_chunk_numbers",
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
                  "Latency includes query embedding and full corpus search; percentiles use nearest rank.",
                  "Token usage covers indexing and queries. N/A means unavailable, not zero.",
                  "Corpus and gold snapshots are identified by SHA-256 of exact input bytes."])
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory
