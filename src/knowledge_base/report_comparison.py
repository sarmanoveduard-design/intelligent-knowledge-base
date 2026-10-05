"""Compare allowlisted numeric report fields; never read query/corpus artifacts."""
import csv
import json
from math import isfinite
from pathlib import Path
import re

from knowledge_base.benchmark_config import BenchmarkConfigurationError

METRICS = ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG",
           "positive-over-hard-negative", "Candidate Recall@10", "Candidate Recall@20",
           "Candidate Recall@50", "Candidate Recall@pool", "p50 latency seconds", "p95 latency seconds",
           "reranker p50 latency seconds", "reranker p95 latency seconds")
FIELDS = ("experiment", "status", "provider", "model", "dimensions", "retrieval_mode",
          "document_representation", "candidate_k", "final_top_k", "bm25_k1", "bm25_b", "rrf_k", "reranker",
          *METRICS, "errors", "query_latency_seconds", "index_latency_seconds", "estimated_cost",
          "commit_sha", "code_dirty", "reranker_model", "reranker_device", "reranker_candidate_k",
          "reranker_batch_size", "reranker_max_length", "reranker_latency_seconds", "hard_negative_scope",
          "hard_negative_evaluable_query_count", "reranker_requested_revision", "reranker_resolved_revision")


def _hash(value):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


def _number(value):
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if isfinite(number) and number >= 0 else None


def _label(value):
    return value if (isinstance(value, str)
                     and not re.match(r"[A-Za-z]:[/\\]", value)
                     and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value)) else None


def _read_report(directory):
    metadata = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Invalid report metadata")
    pair = (_hash(metadata.get("corpus_hash")), _hash(metadata.get("gold_set_hash")))
    if None in pair:
        raise ValueError("Invalid report snapshot hashes")
    with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != ["metric", "value"]:
            raise ValueError("Invalid summary schema")
        summary = {}
        for item in reader:
            if None in item or any(value is None for value in item.values()) or item["metric"] in summary:
                raise ValueError("Invalid summary rows")
            summary[item["metric"]] = _number(item["value"])
    row = {key: summary.get(key) for key in METRICS}
    experiment_id = metadata.get("experiment_id")
    row.update({
        "experiment": experiment_id if isinstance(experiment_id, str) and re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}", experiment_id) else None,
        "status": metadata.get("status") if metadata.get("status") in ("complete", "incomplete") else None,
        "provider": _label(metadata.get("provider")), "model": _label(metadata.get("model_name")),
        "dimensions": _number(metadata.get("dimensions")),
        "retrieval_mode": metadata.get("retrieval_mode", "dense"),
        "document_representation": metadata.get("document_representation", "plain"),
        "candidate_k": _number(metadata.get("candidate_k")),
        "final_top_k": _number(metadata.get("final_top_k", metadata.get("top_k"))),
        "bm25_k1": _number(metadata.get("bm25_k1")), "bm25_b": _number(metadata.get("bm25_b")),
        "rrf_k": _number(metadata.get("rrf_k")), "reranker": _label(metadata.get("reranker")),
        "errors": _number(metadata.get("error_count")),
        "query_latency_seconds": _number(metadata.get("query_latency_seconds")),
        "index_latency_seconds": _number(metadata.get("index_latency_seconds")),
        "estimated_cost": _number(metadata.get("estimated_cost")),
        "commit_sha": metadata.get("code_commit_sha") if isinstance(metadata.get("code_commit_sha"), str)
            and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", metadata["code_commit_sha"]) else None,
        "code_dirty": metadata.get("code_dirty") if type(metadata.get("code_dirty")) is bool else None,
        "reranker_model": _label(metadata.get("reranker_model")),
        "reranker_requested_revision": _label(metadata.get("reranker_requested_revision")),
        "reranker_resolved_revision": metadata.get("reranker_resolved_revision") if isinstance(metadata.get("reranker_resolved_revision"), str)
            and re.fullmatch(r"[0-9a-f]{40}", metadata["reranker_resolved_revision"]) else None,
        "reranker_device": metadata.get("reranker_device") if metadata.get("reranker_device") in ("cpu", "cuda", "auto") else None,
        **{key: _number(metadata.get(key)) for key in ("reranker_candidate_k", "reranker_batch_size",
            "reranker_max_length", "reranker_latency_seconds", "hard_negative_evaluable_query_count")},
        "hard_negative_scope": metadata.get("hard_negative_scope") if metadata.get("hard_negative_scope") in (
            "dense_full_corpus", "bm25_full_corpus", "hybrid_candidate_union", "reranker_candidate_pool") else None,
    })
    if (row["retrieval_mode"] not in ("dense", "bm25", "hybrid_rrf")
            or row["document_representation"] not in ("plain", "structure_aware_v1")):
        raise ValueError("Invalid report retrieval configuration")
    return pair, row


def compare_reports(directories, output_root: Path, *, corpus_hash=None, gold_hash=None):
    if (corpus_hash is None) != (gold_hash is None):
        raise BenchmarkConfigurationError("Comparison requires both corpus and gold hashes")
    if corpus_hash is not None and (_hash(corpus_hash) is None or _hash(gold_hash) is None):
        raise BenchmarkConfigurationError("Comparison snapshot hashes must be SHA-256 hex")
    parsed, invalid = [], 0
    for directory in sorted(set(Path(path) for path in directories)):
        try:
            pair, row = _read_report(directory)
        except (OSError, ValueError, csv.Error, TypeError, KeyError):
            invalid += 1
            continue
        parsed.append((pair, row))
    pairs = {pair for pair, _ in parsed}
    if corpus_hash is None:
        if len(pairs) != 1:
            raise BenchmarkConfigurationError("Select corpus/gold hashes when reports contain zero or multiple datasets")
        selected = next(iter(pairs))
    else:
        selected = corpus_hash, gold_hash
    rows = [row for pair, row in parsed if pair == selected]
    if not rows:
        raise BenchmarkConfigurationError("No compatible retrieval reports found")
    incompatible = sum(pair != selected for pair, _ in parsed)
    for number, row in enumerate(rows, 1):
        row["experiment"] = row["experiment"] or f"report-{number}"
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Retrieval report comparison", "", "Это retrieval benchmark, не оценка качества LLM-ответа.", "",
             f"Corpus SHA256: {selected[0]}", f"Gold SHA256: {selected[1]}", "",
             f"Compared: {len(rows)}; incompatible datasets skipped: {incompatible}; invalid reports skipped: {invalid}.", "",
             "Rows share snapshot hashes; retrieval configuration, top_k, candidate_k, representation and commits may differ.",
             "Incomplete runs remain marked; no automatic winner is selected. N/A means unavailable, not zero.",
             "Legacy reports default to dense/plain; unrecorded candidate settings and metrics stay N/A.", "",
             "| " + " | ".join(FIELDS) + " |", "| " + " | ".join("---" for _ in FIELDS) + " |"]
    lines.extend("| " + " | ".join(str(row[field]) if row[field] is not None else "N/A" for field in FIELDS) + " |" for row in rows)
    (output_root / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"compared": len(rows), "incompatible": incompatible, "invalid": invalid}
