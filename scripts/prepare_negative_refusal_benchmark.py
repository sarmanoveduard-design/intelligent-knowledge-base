"""Prepare data-independent DEV evaluation snapshots offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "benchmarks/generation/negative_refusal_v1/questions.json"
FROZEN_PROMPT = ROOT / "benchmarks/generation/main_119/generation_v1_benchmark.txt"
DEFAULT_OUT = ROOT / "reports/negative_refusal/prepared/negative_refusal_v1/dev"
PROMPT_SHA256 = "f05cb6f1ec4e05d20aeb1cf7bfc8c9d2e508450b42073698707707a6de923a6c"
CASE_TYPES = frozenset((
    "ABSENT_FACT", "PARTIAL_CONTEXT", "WRONG_SCOPE", "CONFLICTING_CONTEXT",
    "OUT_OF_CONTEXT_KNOWLEDGE", "MISSING_CONDITION", "ANSWERABLE_CONTROL"))
BEHAVIORS = {"REFUSE", "CLARIFY", "PARTIAL_ANSWER_WITH_LIMITATION", "FLAG_CONFLICT", "ANSWER"}
ITEM_FIELDS = {
    "query_id", "group_id", "split", "case_type", "query", "synthetic", "contexts",
    "expected_behavior", "expected_refusal", "missing_information", "allowed_conclusion",
    "forbidden_claims", "checks", "provenance",
}
CONTEXT_FIELDS = {"rank", "chunk_id", "document_id", "section_ref", "text", "synthetic"}
SOURCE_FIELDS = {"chunk_id", "document_id", "section_ref"}
CHECK_FIELDS = {"refusal_patterns", "definitive_answer_patterns", "uncertainty_patterns",
                "citation_policy", "required_sources"}
PROVENANCE_FIELDS = {"origin", "source_snapshot_sha256", "source_refs", "transformation",
                     "review_status", "review_note"}


def exact_fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"invalid fields: {label}")


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected nonempty string: {label}")


def canonical_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def patterns(value, label, *, required=False):
    if not isinstance(value, list) or (required and not value):
        raise ValueError(f"invalid patterns: {label}")
    if len(value) != len(set(x for x in value if isinstance(x, str))):
        raise ValueError(f"duplicate or invalid patterns: {label}")
    for pattern in value:
        nonempty(pattern, label)
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"invalid regex: {label}") from error
        if compiled.search(""):
            raise ValueError(f"regex matches empty text: {label}")


def source_ref(value, label):
    exact_fields(value, SOURCE_FIELDS, label)
    for field in SOURCE_FIELDS:
        nonempty(value[field], label)


def probes(value, label):
    if not isinstance(value, list):
        raise ValueError(f"invalid probes: {label}")
    ids = set()
    for probe in value:
        exact_fields(probe, {"id", "description", "match_any"}, label)
        nonempty(probe["id"], label)
        nonempty(probe["description"], label)
        if probe["id"] in ids:
            raise ValueError(f"duplicate probe id: {label}")
        ids.add(probe["id"])
        patterns(probe["match_any"], label, required=True)


def validate_dataset(dataset):
    exact_fields(dataset, {"schema_version", "benchmark_id", "context_top_k", "queries"}, "dataset")
    if type(dataset["schema_version"]) is not int or dataset["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    nonempty(dataset["benchmark_id"], "benchmark_id")
    if type(dataset["context_top_k"]) is not int or dataset["context_top_k"] < 0:
        raise ValueError("context_top_k must be a nonnegative integer")
    if not isinstance(dataset["queries"], list) or not dataset["queries"]:
        raise ValueError("queries must be a nonempty list")
    ids, group_splits = set(), {}
    for item in dataset["queries"]:
        exact_fields(item, ITEM_FIELDS, "item")
        for field in ("query_id", "group_id", "query", "allowed_conclusion"):
            nonempty(item[field], field)
        if item["query_id"] in ids:
            raise ValueError("duplicate query_id")
        ids.add(item["query_id"])
        if not isinstance(item["case_type"], str) or item["case_type"] not in CASE_TYPES:
            raise ValueError("invalid case_type")
        if not isinstance(item["expected_behavior"], str) or item["expected_behavior"] not in BEHAVIORS:
            raise ValueError("invalid expected_behavior")
        if type(item["expected_refusal"]) is not bool:
            raise ValueError("expected_refusal must be boolean")
        answerable = item["case_type"] == "ANSWERABLE_CONTROL"
        if (item["expected_refusal"] != (item["expected_behavior"] != "ANSWER")
                or answerable != (item["expected_behavior"] == "ANSWER")):
            raise ValueError("expected_refusal/expected_behavior/case_type mismatch")
        if type(item["synthetic"]) is not bool:
            raise ValueError("synthetic must be boolean")
        provenance = item["provenance"]
        exact_fields(provenance, PROVENANCE_FIELDS, "provenance")
        if item["split"] == "holdout" and provenance["review_status"] != "approved":
            raise ValueError("holdout must be approved")
        if item["split"] != "dev":
            raise ValueError("only DEV is supported; holdout preparation is not enabled")
        group = item["group_id"]
        if group in group_splits and group_splits[group] != item["split"]:
            raise ValueError("group_id cannot cross splits")
        group_splits[group] = item["split"]
        if provenance["origin"] not in ("synthetic", "source_based", "derived"):
            raise ValueError("invalid provenance origin")
        if provenance["review_status"] not in ("draft", "approved"):
            raise ValueError("invalid review_status")
        if provenance["origin"] == "synthetic":
            if (not item["synthetic"] or provenance["source_snapshot_sha256"] is not None
                    or provenance["transformation"] is not None):
                raise ValueError("synthetic provenance requires synthetic=true and null snapshot/transformation")
        else:
            digest = provenance["source_snapshot_sha256"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("source-based/derived provenance requires SHA-256")
            if provenance["origin"] == "source_based":
                if item["synthetic"] or provenance["transformation"] is not None:
                    raise ValueError("source_based provenance requires synthetic=false and no transformation")
            else:
                nonempty(provenance["transformation"], "transformation")
        nonempty(provenance["review_note"], "review_note")
        contexts = item["contexts"]
        if not isinstance(contexts, list) or len(contexts) > dataset["context_top_k"]:
            raise ValueError("contexts exceed dataset context_top_k or are not a list")
        if answerable and not contexts:
            raise ValueError("ANSWERABLE_CONTROL requires evidence contexts")
        by_chunk = {}
        for rank, context in enumerate(contexts, 1):
            exact_fields(context, CONTEXT_FIELDS, "context")
            if type(context["rank"]) is not int or context["rank"] != rank:
                raise ValueError("nonsequential ranks")
            if type(context["synthetic"]) is not bool:
                raise ValueError("context synthetic must be boolean")
            if (provenance["origin"] == "synthetic" and not context["synthetic"]
                    or provenance["origin"] == "source_based" and context["synthetic"]
                    or context["synthetic"] and not item["synthetic"]):
                raise ValueError("context synthetic flag contradicts provenance/item")
            for field in ("chunk_id", "document_id", "section_ref", "text"):
                nonempty(context[field], field)
            if any(character in context[field] for field in SOURCE_FIELDS for character in "|[]\r\n"):
                raise ValueError("source identifiers contain citation delimiters")
            if context["chunk_id"] in by_chunk:
                raise ValueError("duplicate chunk_id inside item")
            by_chunk[context["chunk_id"]] = context
        checks = item["checks"]
        exact_fields(checks, CHECK_FIELDS, "checks")
        for field in ("refusal_patterns", "definitive_answer_patterns", "uncertainty_patterns"):
            patterns(checks[field], field, required=True)
        if checks["citation_policy"] not in ("optional", "required"):
            raise ValueError("invalid citation_policy")
        if checks["citation_policy"] == "required" and not contexts:
            raise ValueError("required citations need contexts")
        for label, sources in (("required_sources", checks["required_sources"]),
                               ("source_refs", provenance["source_refs"])):
            if not isinstance(sources, list):
                raise ValueError(f"invalid {label}")
            seen = set()
            for source in sources:
                source_ref(source, label)
                context = by_chunk.get(source["chunk_id"])
                if context is None or any(context[field] != source[field] for field in SOURCE_FIELDS):
                    raise ValueError(f"{label} outside contexts or inconsistent provenance")
                if source["chunk_id"] in seen:
                    raise ValueError(f"duplicate {label}")
                seen.add(source["chunk_id"])
        if checks["required_sources"] and checks["citation_policy"] != "required":
            raise ValueError("required_sources require citation_policy=required")
        probes(item["missing_information"], "missing_information")
        probes(item["forbidden_claims"], "forbidden_claims")
        if item["expected_refusal"] and not item["missing_information"]:
            raise ValueError("negative item requires missing_information")
        if answerable and item["missing_information"]:
            raise ValueError("answerable control cannot require missing information")


def model_input_projection(dataset):
    """Project only allowed fields, ordered by query_id; never copy labels."""
    queries = sorted(dataset["queries"], key=lambda item: item["query_id"])
    return {**{field: dataset[field] for field in ("schema_version", "benchmark_id", "context_top_k")},
            "queries": [{"query_id": item["query_id"], "query": item["query"], "contexts": [
                {field: context[field] for field in ("rank", "chunk_id", "document_id", "section_ref", "text")}
                for context in item["contexts"]]} for item in queries]}


def write_bundle(out, files, protected=()):
    """Publish a complete new directory; identical reruns are read-only."""
    out = Path(out).resolve()
    for name in ("benchmarks", "docs", "scripts", "tests", "notebooks", "prompts", ".git"):
        directory = (ROOT / name).resolve()
        if out == directory or directory in out.parents:
            raise ValueError("output overlaps protected repository files")
    for path in protected:
        path = Path(path).resolve()
        if out == path or out in path.parents:
            raise ValueError("output overlaps protected inputs")
    if out.exists():
        if (not out.is_dir() or {p.name for p in out.iterdir()} != set(files)
                or any((out / name).read_bytes() != raw for name, raw in files.items())):
            raise ValueError("output already exists with different contents; choose a new directory")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".negative-refusal-", dir=out.parent))
    try:
        for name, raw in files.items():
            with (staging / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        staging.rename(out)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()


def prepare(questions=DEFAULT_QUESTIONS, out=DEFAULT_OUT):
    questions = Path(questions)
    raw = questions.read_bytes()
    dataset = load_json(questions)
    if raw != questions.read_bytes():
        raise ValueError("questions changed while reading")
    validate_dataset(dataset)
    prompt = FROZEN_PROMPT.read_bytes()
    if sha256(prompt) != PROMPT_SHA256:
        raise ValueError("frozen MAIN119 prompt hash mismatch")
    queries = sorted(dataset["queries"], key=lambda item: item["query_id"])
    metadata = {field: dataset[field] for field in ("schema_version", "benchmark_id", "context_top_k")}
    files = {"model_input.json": canonical_json(model_input_projection(dataset)),
             "references.json": canonical_json({**metadata, "queries": queries}), "prompt.txt": prompt}
    manifest = {**metadata, "split": "dev", "query_count": len(queries),
                "synthetic": all(item["synthetic"] for item in queries),
                "purpose": "evaluator development only; not model quality or independent holdout",
                "source_questions_sha256": sha256(raw),
                "model_input_sha256": sha256(files["model_input.json"]),
                "references_sha256": sha256(files["references.json"]),
                "prompt_sha256": sha256(prompt)}
    files["manifest.json"] = canonical_json(manifest)
    write_bundle(out, files, (questions, FROZEN_PROMPT))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.questions, args.out)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(f"Prepared {manifest['query_count']} DEV items: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
