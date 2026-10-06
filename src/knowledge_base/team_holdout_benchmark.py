"""Offline normalized HOLDOUT bridge to the existing team benchmark adapters."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from knowledge_base.team_benchmark import (
    DOCUMENT_ID, TeamCorpusAdapter, TeamDatasetError, TeamGoldAdapter,
    benchmark_snapshots, canonical_json, extract_document_id, snapshot_hash,
    validate_embedding_maps,
)

MODES = ("approved-only", "all")
FIELDS = {
    "query_id", "query", "difficulty", "answer_hint", "document_id",
    "positive_section_refs", "hard_negatives", "source_file", "source_row", "status",
}


class TeamHoldoutGoldAdapter:
    """Validate every normalized row before status selection; preserve source metadata."""

    def from_source(self, raw, corpus, *, mode="approved-only"):
        if mode not in MODES:
            raise TeamDatasetError("invalid_holdout_mode")
        try:
            rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        except (UnicodeError, json.JSONDecodeError):
            raise TeamDatasetError("invalid_holdout_jsonl") from None
        if not rows:
            raise TeamDatasetError("empty_gold")
        converted = []
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict) or set(row) != FIELDS:
                raise TeamDatasetError("invalid_holdout_schema", row=number)
            if row["status"] not in ("approved", "provisional"):
                raise TeamDatasetError("invalid_holdout_status", row=number)
            string_fields = FIELDS - {"positive_section_refs", "hard_negatives", "source_row"}
            if any(not isinstance(row[field], str) for field in string_fields):
                raise TeamDatasetError("invalid_holdout_schema", row=number)
            if (type(row["source_row"]) is not int or row["source_row"] < 1
                    or not row["source_file"].strip() or not DOCUMENT_ID.fullmatch(row["document_id"])):
                raise TeamDatasetError("invalid_holdout_provenance_or_document", row=number)
            positives, negatives = row["positive_section_refs"], row["hard_negatives"]
            if (not isinstance(positives, list) or not positives
                    or any(not isinstance(ref, str) or not ref.strip() for ref in positives)):
                raise TeamDatasetError("invalid_holdout_positive_refs", row=number)
            if (not isinstance(negatives, list) or any(
                    not isinstance(ref, dict) or set(ref) != {"document_id", "section_ref"}
                    or any(not isinstance(ref[field], str) or not ref[field].strip()
                           for field in ("document_id", "section_ref")) for ref in negatives)):
                raise TeamDatasetError("invalid_holdout_hard_negatives", row=number)
            converted.append({"query_id": row["query_id"], "query": row["query"],
                              "difficulty": row["difficulty"], "answer_hint": row["answer_hint"],
                              "document_id": row["document_id"],
                              "positive_refs": [{"document_id": row["document_id"], "section_ref": ref}
                                                for ref in positives],
                              "hard_negative_refs": negatives})
        # Validate all rows/refs, including provisional rows excluded by the default mode.
        cases = TeamGoldAdapter().from_source(
            ".jsonl", b"".join(canonical_json(row) for row in converted), corpus)
        by_id = {row["query_id"].strip(): row for row in rows}
        selected = tuple(replace(case, metadata={**case.metadata,
            **{field: by_id[case.query_id][field] for field in ("status", "source_file", "source_row")}})
            for case in cases if mode == "all" or by_id[case.query_id]["status"] == "approved")
        if not selected:
            raise TeamDatasetError("empty_holdout_selection")
        counts = Counter(row["status"] for row in rows)
        return selected, {"source_question_count": len(rows), "selected_question_count": len(selected),
                          "approved_count": counts["approved"], "provisional_count": counts["provisional"],
                          "selected_approved_count": sum(case.metadata["status"] == "approved" for case in selected),
                          "selected_provisional_count": sum(case.metadata["status"] == "provisional" for case in selected)}


def prepare_team_holdout_benchmark(corpus_dir: Path, gold: Path, out: Path, *, mode="approved-only") -> dict:
    """Write deterministic snapshots only after the entire source dataset passes validation."""
    corpus_dir, gold, out = Path(corpus_dir), Path(gold), Path(out)
    if mode not in MODES:
        raise TeamDatasetError("invalid_holdout_mode")
    for protected in (corpus_dir.resolve(), gold.parent.resolve()):
        if out.resolve() == protected or protected in out.resolve().parents:
            raise TeamDatasetError("holdout_output_overlaps_inputs")
    corpus_sources = {p.name: p.read_bytes() for p in sorted(corpus_dir.glob("*_embeddable.csv"))}
    map_sources = {p.name: p.read_bytes() for p in sorted(corpus_dir.glob("*_embedding_map.csv"))}
    gold_raw = gold.read_bytes()
    chunks = TeamCorpusAdapter().from_sources(corpus_sources)
    corpus_docs = [extract_document_id(name) for name in corpus_sources]
    map_docs = [extract_document_id(name, "_embedding_map.csv") for name in map_sources]
    if (len(set(corpus_docs)) != len(corpus_docs) or len(set(map_docs)) != len(map_docs)
            or set(corpus_docs) != set(map_docs)):
        raise TeamDatasetError("holdout_requires_one_map_per_document")
    validate_embedding_maps(map_sources, chunks)
    cases, counts = TeamHoldoutGoldAdapter().from_source(gold_raw, chunks, mode=mode)
    corpus_bytes, gold_bytes = benchmark_snapshots(chunks, cases)
    sources = ([{"source_file": name, "role": "corpus", "sha256": snapshot_hash(raw)}
                for name, raw in sorted(corpus_sources.items())]
               + [{"source_file": name, "role": "embedding_map", "sha256": snapshot_hash(raw)}
                  for name, raw in sorted(map_sources.items())]
               + [{"source_file": gold.name, "role": "normalized_gold", "sha256": snapshot_hash(gold_raw)}])
    manifest = {"schema_version": 1, "mode": mode, **counts,
                "document_count": len(set(corpus_docs)), "document_ids": sorted(set(corpus_docs)),
                "chunk_count": len(chunks), "query_count": len(cases), "unresolved_refs_count": 0,
                "reference_validation_scope": "all source questions, before mode filtering",
                "embeddable_csv_count": len(corpus_sources), "embedding_map_csv_count": len(map_sources),
                "source_files": sources, "corpus_snapshot_hash": snapshot_hash(corpus_bytes),
                "gold_snapshot_hash": snapshot_hash(gold_bytes)}
    out.mkdir(parents=True, exist_ok=True)
    for name, raw in (("corpus.json", corpus_bytes), ("gold.json", gold_bytes),
                      ("manifest.json", canonical_json(manifest))):
        (out / name).write_bytes(raw)
    return manifest
