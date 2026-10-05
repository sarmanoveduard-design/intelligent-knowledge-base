"""Strict import boundary for team datasets; no embedding/provider dependencies."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from knowledge_base.benchmark_models import BenchmarkCase, CorpusChunk

DOCUMENT_ID = re.compile(r"doc_[0-9]{4}")


class TeamDatasetError(ValueError):
    """Structural diagnostics without source texts, hints or absolute paths."""

    def __init__(self, code: str, *, row: int | None = None, unresolved_refs_count: int = 0):
        self.code = code
        self.row = row
        self.unresolved_refs_count = unresolved_refs_count
        super().__init__(f"{code}" + (f" (row {row})" if row is not None else "")
                         + (f"; unresolved refs: {unresolved_refs_count}" if unresolved_refs_count else ""))


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def snapshot_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def extract_document_id(filename: str, suffix: str = "_embeddable.csv") -> str:
    match = re.search(r"(?:\((doc_[0-9]{4})\)|(doc_[0-9]{4}))" + re.escape(suffix) + r"$", filename)
    if not match:
        raise TeamDatasetError("invalid_document_filename")
    document_id = match.group(1) or match.group(2)
    # Unparenthesized IDs must start at a token boundary.
    if match.group(2) and match.start() and filename[match.start() - 1].isalnum():
        raise TeamDatasetError("invalid_document_filename")
    return document_id


def chunk_id(document_id: str, section_ref: str) -> str:
    if not DOCUMENT_ID.fullmatch(document_id):
        raise TeamDatasetError("invalid_document_id")
    if not isinstance(section_ref, str) or not section_ref.strip():
        raise TeamDatasetError("empty_section_ref")
    # Canonical tuple encoding is collision-safe even for punctuation in section refs.
    digest = snapshot_hash(canonical_json([document_id, section_ref.strip()]))
    return f"{document_id}:{digest}"


def _csv_rows(raw: bytes, required: set[str]) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        raise TeamDatasetError("csv_requires_utf8") from None
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), dialect=dialect, strict=True)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields) or not required.issubset(fields):
            raise TeamDatasetError("missing_or_duplicate_csv_columns")
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise TeamDatasetError("invalid_csv_row_width", row=number)
            rows.append(row)
        return rows
    except csv.Error:
        raise TeamDatasetError("invalid_csv") from None


class TeamCorpusAdapter:
    def load(self, files: Iterable[Path]) -> tuple[CorpusChunk, ...]:
        paths = tuple(Path(path) for path in files)
        if len({path.name for path in paths}) != len(paths):
            raise TeamDatasetError("duplicate_source_filename")
        return self.from_sources({path.name: path.read_bytes() for path in paths})

    def from_sources(self, sources: dict[str, bytes]) -> tuple[CorpusChunk, ...]:
        chunks, identities = [], set()
        if not sources:
            raise TeamDatasetError("no_embeddable_csv")
        for name, raw in sorted(sources.items()):
            document_id = extract_document_id(name)
            rows = _csv_rows(raw, {"section_ref", "chapter", "article_title", "text"})
            if not rows:
                raise TeamDatasetError("empty_corpus_file")
            for number, row in enumerate(rows, 2):
                ref = row["section_ref"].strip()
                identity = (document_id, ref)
                if identity in identities:
                    raise TeamDatasetError("duplicate_section_ref", row=number)
                if not row["text"].strip():
                    raise TeamDatasetError("empty_corpus_text", row=number)
                if "document_id" in row and row["document_id"].strip() != document_id:
                    raise TeamDatasetError("document_id_filename_mismatch", row=number)
                identifier = chunk_id(document_id, ref)
                identities.add(identity)
                chunks.append(CorpusChunk(identifier, document_id, ref, row["text"], {
                    "chapter": row["chapter"], "article_title": row["article_title"], "source_file": name,
                }))
        return tuple(sorted(chunks, key=lambda item: (item.document_id, item.section_ref)))


def _refs(value: object, default_document: str, *, row: int) -> tuple[str, ...]:
    """Explicit refs: objects or doc_0001:section; plain sections need an explicit default doc."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        if value.startswith(("[", "{")):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise TeamDatasetError("invalid_refs_json", row=row) from None
        else:
            value = value.split(";")
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise TeamDatasetError("invalid_refs_format", row=row)
    refs = []
    for ref in value:
        if isinstance(ref, dict):
            document = ref.get("document_id", default_document)
            section = ref.get("section_ref")
        elif isinstance(ref, str):
            ref = ref.strip()
            qualified = re.fullmatch(r"(doc_[0-9]{4})[:#](.+)", ref)
            if qualified:
                document, section = qualified.groups()
            else:
                # Reject a malformed qualified doc ref rather than treating it as a section.
                if ref.startswith("doc_"):
                    raise TeamDatasetError("invalid_document_ref", row=row)
                document, section = default_document, ref
        else:
            raise TeamDatasetError("invalid_ref", row=row)
        if not isinstance(document, str) or not DOCUMENT_ID.fullmatch(document):
            raise TeamDatasetError("invalid_document_id", row=row)
        if not isinstance(section, str) or not section.strip():
            raise TeamDatasetError("empty_section_ref", row=row)
        refs.append(chunk_id(document, section))
    if len(set(refs)) != len(refs):
        raise TeamDatasetError("duplicate_ref", row=row)
    return tuple(sorted(refs))


class TeamGoldAdapter:
    def load(self, path: Path, corpus: Iterable[CorpusChunk]) -> tuple[BenchmarkCase, ...]:
        return self.from_source(path.suffix.lower(), path.read_bytes(), corpus)

    def from_source(self, suffix: str, raw: bytes, corpus: Iterable[CorpusChunk]) -> tuple[BenchmarkCase, ...]:
        if suffix == ".csv":
            rows = _csv_rows(raw, {"query_id", "query", "difficulty", "document_id", "section_ref"})
        elif suffix == ".jsonl":
            try:
                rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
            except (UnicodeError, json.JSONDecodeError):
                raise TeamDatasetError("invalid_gold_jsonl") from None
        else:
            raise TeamDatasetError("unsupported_gold_format")
        if not rows:
            raise TeamDatasetError("empty_gold")
        known = {item.chunk_id for item in corpus}
        cases, query_ids = [], set()
        missing_count = 0
        first_missing_row = None
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise TeamDatasetError("invalid_gold_row", row=number)
            query_id, query, difficulty = row.get("query_id"), row.get("query"), row.get("difficulty")
            if not isinstance(query_id, str) or not query_id.strip():
                raise TeamDatasetError("empty_query_id", row=number)
            query_id = query_id.strip()
            if query_id in query_ids:
                raise TeamDatasetError("duplicate_query_id", row=number)
            query_ids.add(query_id)
            if not isinstance(query, str) or not query.strip():
                raise TeamDatasetError("empty_query", row=number)
            if difficulty not in ("easy", "medium", "hard"):
                raise TeamDatasetError("invalid_difficulty", row=number)
            default = row.get("document_id", "")
            if not isinstance(default, str) or (default and not DOCUMENT_ID.fullmatch(default)):
                raise TeamDatasetError("invalid_document_id", row=number)
            # CSV document_id/section_ref always validated, even when positive_refs is supplied.
            primary = ()
            if suffix == ".csv" or "section_ref" in row:
                primary = _refs([{"document_id": default, "section_ref": row.get("section_ref")}], default, row=number)
            if suffix == ".jsonl" and "positive_refs" not in row:
                raise TeamDatasetError("missing_positive_refs", row=number)
            positives = _refs(row["positive_refs"], default, row=number) if "positive_refs" in row else primary
            if not positives:
                raise TeamDatasetError("empty_positive_refs", row=number)
            supported_negative_columns = {"hard_negative_refs", "hard_negative_section_refs",
                                          "hard_negative_section_ref", "hard_negative_document_id"}
            if any("negative" in key.lower() and key not in supported_negative_columns and value
                   for key, value in row.items()):
                raise TeamDatasetError("unsupported_hard_negative_column", row=number)
            negative_columns = [key for key in ("hard_negative_refs", "hard_negative_section_refs") if row.get(key)]
            if len(negative_columns) > 1:
                raise TeamDatasetError("ambiguous_hard_negative_columns", row=number)
            negatives = _refs(row[negative_columns[0]], default, row=number) if negative_columns else ()
            if row.get("hard_negative_section_ref"):
                if negatives:
                    raise TeamDatasetError("ambiguous_hard_negative_columns", row=number)
                negatives = _refs([{"document_id": row.get("hard_negative_document_id") or default,
                                    "section_ref": row["hard_negative_section_ref"]}], default, row=number)
            elif row.get("hard_negative_document_id"):
                raise TeamDatasetError("missing_hard_negative_section_ref", row=number)
            if set(positives) & set(negatives):
                raise TeamDatasetError("positive_hard_negative_overlap", row=number)
            missing = (set(positives) | set(negatives) | set(primary)) - known
            if missing:
                missing_count += len(missing)
                first_missing_row = first_missing_row or number
            metadata = {}
            for name in ("answer_hint", "source_document"):
                value = row.get(name, "")
                if not isinstance(value, str):
                    raise TeamDatasetError("invalid_gold_metadata", row=number)
                metadata[name] = value
            cases.append(BenchmarkCase(query, positives, negatives, query_id, difficulty, metadata))
        if missing_count:
            raise TeamDatasetError("unresolved_refs", row=first_missing_row, unresolved_refs_count=missing_count)
        return tuple(sorted(cases, key=lambda item: item.query_id))


def _validate_maps(sources: dict[str, bytes], chunks: tuple[CorpusChunk, ...]) -> None:
    by_document: dict[str, set[str]] = {}
    for item in chunks:
        by_document.setdefault(item.document_id, set()).add(item.section_ref)
    for name, raw in sources.items():
        document = extract_document_id(name, "_embedding_map.csv")
        rows = _csv_rows(raw, {"section_ref"})
        refs = {row["section_ref"].strip() for row in rows}
        if "" in refs or document not in by_document or refs != by_document[document]:
            raise TeamDatasetError("embedding_map_coverage_mismatch")
        if any("document_id" in row and row["document_id"].strip() != document for row in rows):
            raise TeamDatasetError("embedding_map_document_mismatch")


def prepare_team_benchmark(corpus_dir: Path, gold: Path, out: Path) -> dict:
    files = sorted(corpus_dir.glob("*_embeddable.csv"))
    maps = sorted(corpus_dir.glob("*_embedding_map.csv"))
    # Read each source once: manifest hashes always describe the bytes actually parsed.
    corpus_sources = {path.name: path.read_bytes() for path in files}
    map_sources = {path.name: path.read_bytes() for path in maps}
    gold_bytes = gold.read_bytes()
    chunks = TeamCorpusAdapter().from_sources(corpus_sources)
    _validate_maps(map_sources, chunks)
    cases = TeamGoldAdapter().from_source(gold.suffix.lower(), gold_bytes, chunks)
    corpus_bytes = canonical_json({"chunks": [
        {"id": item.chunk_id, "document_id": item.document_id, "section_ref": item.section_ref,
         "text": item.text, "metadata": item.metadata} for item in chunks]})
    gold_snapshot = canonical_json({"queries": [
        {"query_id": item.query_id, "text": item.text, "difficulty": item.difficulty,
         "positive_chunk_ids": item.positives, "hard_negative_chunk_ids": item.hard_negatives,
         "metadata": item.metadata} for item in cases]})
    difficulty = Counter(item.difficulty for item in cases)
    sources = ([{"source_file": name, "role": "corpus", "sha256": snapshot_hash(raw)}
                for name, raw in sorted(corpus_sources.items())]
               + [{"source_file": name, "role": "embedding_map", "sha256": snapshot_hash(raw)}
                  for name, raw in sorted(map_sources.items())]
               + [{"source_file": gold.name, "role": "gold", "sha256": snapshot_hash(gold_bytes)}])
    manifest = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv_count": len(files) + len(maps) + int(gold.suffix.lower() == ".csv"),
        "embeddable_csv_count": len(files), "embedding_map_csv_count": len(maps),
        "document_count": len({item.document_id for item in chunks}), "chunk_count": len(chunks),
        "query_count": len(cases), "difficulty_counts": {level: difficulty[level] for level in ("easy", "medium", "hard")},
        "source_files": sources, "corpus_snapshot_hash": snapshot_hash(corpus_bytes),
        "gold_snapshot_hash": snapshot_hash(gold_snapshot), "unresolved_refs_count": 0,
    }
    # No output is touched until every row/reference/map has passed validation.
    out.mkdir(parents=True, exist_ok=True)
    for name, raw in (("corpus.json", corpus_bytes), ("gold.json", gold_snapshot),
                      ("manifest.json", canonical_json(manifest))):
        (out / name).write_bytes(raw)
    return manifest
