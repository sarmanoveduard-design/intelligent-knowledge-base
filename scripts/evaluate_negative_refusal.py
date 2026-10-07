"""Intentionally transparent, deterministic, offline DEV response evaluator."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
from pathlib import Path
import re

try:
    from scripts.prepare_negative_refusal_benchmark import (
        canonical_json, exact_fields, load_json, model_input_projection, nonempty,
        sha256, validate_dataset, write_bundle,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/evaluate_negative_refusal.py
    from prepare_negative_refusal_benchmark import (
        canonical_json, exact_fields, load_json, model_input_projection, nonempty,
        sha256, validate_dataset, write_bundle,
    )

LIMITATIONS = [
    "Intentionally transparent and deterministic; metrics use pre-annotated probes, not semantic verification.",
    "Zero forbidden_claims hits does NOT prove the absence of all hallucinations.",
    "Semantic support may later be reviewed manually or by a separately authorized judge.",
    "Quote and local negation suppression is heuristic; paraphrases and complex negation can be missed.",
    "A valid chunk_id proves source membership/provenance, not that its text supports the conclusion.",
    "DEV develops evaluation infrastructure; synthetic fixtures test the evaluator, not model quality or holdout performance.",
]
METRICS = ("answered_when_should_refuse", "refused_when_should_answer", "unsupported_claim",
           "invalid_source", "source_outside_context", "explicit_uncertainty",
           "missing_info_identified", "strict_format_compliant")
BRACKETS = re.compile(r"\[([^\[\]\r\n]*)\]")
QUOTES = re.compile(r'«[^»]*»|“[^”]*”|"[^"\n]*"|`[^`]*`')
NEGATION = re.compile(r"(?:\bне\b|\bнет\b|\bнельзя\b|\bневерно\b|\bошибочно\b|\bложно\b|\bnot\b)\s*[^.!?;\n]{0,60}$", re.I)


def active_matches(text, patterns, *, suppress=True):
    quoted = [match.span() for match in QUOTES.finditer(text)]
    seen, matches = set(), []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            start, end = match.span()
            line_start = text.rfind("\n", 0, start) + 1
            if (any(left <= start < right for left, right in quoted)
                    or text[line_start:start].lstrip().startswith(">")):
                continue
            if suppress:
                clause_start = max(text.rfind(character, 0, start) for character in ".!?;\n") + 1
                if NEGATION.search(text[clause_start:start]):
                    continue
            if (start, end) not in seen:
                matches.append({"matched_text": match.group(), "position": {"start": start, "end": end}})
                seen.add((start, end))
    return sorted(matches, key=lambda hit: (hit["position"]["start"], hit["position"]["end"]))


def citation_metrics(answer, contexts, checks):
    by_chunk = {context["chunk_id"]: context for context in contexts}
    resolved, errors = set(), []
    outside, canonical = 0, True
    matches = list(BRACKETS.finditer(answer))
    for match in matches:
        fields = [field.strip() for field in match.group(1).split("|")]
        context, chunk = None, None
        if len(fields) == 3 and all(fields):
            chunk, doc, section = fields
            context = by_chunk.get(chunk)
            if context is None:
                outside += 1
            elif context["document_id"] != doc or context["section_ref"] != section:
                errors.append({"citation": match.group(), "reason": "provenance_mismatch"})
                continue
        elif len(fields) == 2 and all(fields):
            candidates = [ctx for ctx in contexts if (ctx["document_id"], ctx["section_ref"]) == tuple(fields)]
            if len(candidates) == 1:
                context = candidates[0]
                chunk = context["chunk_id"]
            elif len(candidates) > 1:
                errors.append({"citation": match.group(), "reason": "ambiguous_chunk_id"})
                continue
            else:
                outside += 1
        else:
            canonical = False
            errors.append({"citation": match.group(), "reason": "malformed_citation"})
            continue
        if context is None:
            errors.append({"citation": match.group(), "reason": "source_outside_context"})
        else:
            resolved.add(chunk)
    # Malformed/unclosed brackets must not silently disappear from strict-format checks.
    if answer.count("[") != len(matches) or answer.count("]") != len(matches):
        canonical = False
        errors.append({"citation": "", "reason": "unbalanced_brackets"})
    atom = r"\[[^\[\]\r\n]+\]"
    final_block = bool(re.search(r"(?:^|\n)\s*Источник\s*:\s*" + atom + r"\s*$", answer, re.I)
                       or re.search(r"(?:^|\n)\s*Источники\s*:\s*\n(?:\s*-\s*" + atom + r"\s*\n?)+\s*$", answer, re.I))
    orphan_header = bool(re.search(r"(?:^|\n)\s*Источники?\s*:", answer, re.I)) and not final_block
    required = {source["chunk_id"] for source in checks["required_sources"]}
    strict = (canonical and not errors and not orphan_header and required <= resolved
              and (not matches or final_block)
              and (checks["citation_policy"] == "optional" or bool(matches)))
    return {"invalid_source": bool(errors), "source_outside_context": bool(outside),
            "invalid_source_count": len(errors), "source_outside_context_count": outside,
            "citation_errors": errors, "resolved_chunk_ids": sorted(resolved),
            "strict_format_compliant": strict}


def evaluate_item(item, response):
    row = {"query_id": item["query_id"], "case_type": item["case_type"], "split": item["split"],
           "expected_behavior": item["expected_behavior"], "expected_refusal": item["expected_refusal"],
           "observed_behavior": "UNCLASSIFIED", **{metric: None for metric in METRICS},
           "response_missing": response is None, "response_empty": False, "evaluation_error": False,
           "error_reason": None, "forbidden_claim_hits": [], "missing_info_hits": [],
           "invalid_source_count": 0, "source_outside_context_count": 0,
           "citation_errors": [], "resolved_chunk_ids": [], "missing_info_matched": 0,
           "missing_info_expected": len(item["missing_information"])}
    if response is None:
        return row
    if (set(response) - {"query_id", "answer", "status"} or "answer" not in response
            or not isinstance(response["answer"], str)
            or response.get("status", "ok") not in ("ok", "error")):
        row.update(evaluation_error=True, error_reason="invalid_response_schema")
        return row
    if response.get("status", "ok") == "error":
        row.update(evaluation_error=True, error_reason="response_status_error")
        return row
    answer = response["answer"]
    if not answer.strip():
        row["response_empty"] = True
        return row
    checks = item["checks"]
    # Refusal/uncertainty expressions intentionally include negation ("не могу ответить").
    refusal = bool(active_matches(answer, checks["refusal_patterns"], suppress=False))
    definitive = bool(active_matches(answer, checks["definitive_answer_patterns"]))
    row["observed_behavior"] = ("MIXED" if refusal and definitive else "REFUSAL" if refusal
                                else "ANSWER" if definitive else "UNCLASSIFIED")
    for probe in item["forbidden_claims"]:
        row["forbidden_claim_hits"].extend({"probe_id": probe["id"], **hit}
            for hit in active_matches(answer, probe["match_any"]))
    for probe in item["missing_information"]:
        hits = active_matches(answer, probe["match_any"], suppress=False)
        if hits:
            row["missing_info_matched"] += 1
            row["missing_info_hits"].extend({"probe_id": probe["id"], **hit} for hit in hits)
    row.update(
        answered_when_should_refuse=definitive if item["expected_refusal"] else None,
        refused_when_should_answer=(refusal and not definitive) if not item["expected_refusal"] else None,
        unsupported_claim=bool(row["forbidden_claim_hits"]),
        explicit_uncertainty=bool(active_matches(answer, checks["uncertainty_patterns"], suppress=False)),
        missing_info_identified=(row["missing_info_matched"] == row["missing_info_expected"]
                                 if row["missing_info_expected"] else None),
    )
    row.update(citation_metrics(answer, item["contexts"], checks))
    return row


def summarize(rows):
    usable = [row for row in rows if not any(row[key] for key in
              ("response_missing", "response_empty", "evaluation_error"))]
    metrics = {}
    for metric in METRICS:
        values = [row[metric] for row in usable if row[metric] is not None]
        metrics[metric] = {"count": sum(values), "denominator": len(values),
                           "rate": sum(values) / len(values) if values else None}
    return {"total_expected": len(rows), "total_received": sum(not r["response_missing"] for r in rows),
            "missing": sum(r["response_missing"] for r in rows),
            "empty": sum(r["response_empty"] for r in rows),
            "errors": sum(r["evaluation_error"] for r in rows), "evaluable": len(usable),
            "unclassified": sum(r["observed_behavior"] == "UNCLASSIFIED" for r in usable),
            "observed_behavior_counts": dict(Counter(r["observed_behavior"] for r in usable)),
            "forbidden_claim_hit_count": sum(len(r["forbidden_claim_hits"]) for r in usable),
            "invalid_source_count": sum(r["invalid_source_count"] for r in usable),
            "source_outside_context_count": sum(r["source_outside_context_count"] for r in usable),
            "metrics": metrics}


def evaluate(references, model_input, responses, out):
    references, model_input, responses = map(Path, (references, model_input, responses))
    refs, inputs, answers = map(load_json, (references, model_input, responses))
    validate_dataset(refs)
    exact_fields(inputs, {"schema_version", "benchmark_id", "context_top_k", "queries"}, "model_input")
    if not isinstance(inputs["queries"], list):
        raise ValueError("model_input queries must be a list")
    input_ids = set()
    for item in inputs["queries"]:
        exact_fields(item, {"query_id", "query", "contexts"}, "model_input item")
        nonempty(item["query_id"], "model_input query_id")
        if item["query_id"] in input_ids:
            raise ValueError("duplicate model_input query_id")
        input_ids.add(item["query_id"])
    inputs["queries"] = sorted(inputs["queries"], key=lambda item: item["query_id"])
    expected_projection = model_input_projection(refs)
    # JSON comparison preserves types: Python alone considers True == 1.
    if canonical_json(inputs) != canonical_json(expected_projection):
        raise ValueError("model_input/references mismatch or label leakage")
    exact_fields(answers, {"schema_version", "model", "responses"}, "responses")
    if type(answers["schema_version"]) is not int or answers["schema_version"] != 1:
        raise ValueError("unsupported response schema_version")
    nonempty(answers["model"], "model")
    if not isinstance(answers["responses"], list):
        raise ValueError("responses must be a list")
    by_id = {}
    ids = {item["query_id"] for item in refs["queries"]}
    for response in answers["responses"]:
        if not isinstance(response, dict):
            raise ValueError("response must be an object with query_id")
        nonempty(response.get("query_id"), "response query_id")
        qid = response["query_id"]
        if qid in by_id or qid not in ids:
            raise ValueError("duplicate or unknown response query_id")
        by_id[qid] = response
    rows = [evaluate_item(item, by_id.get(item["query_id"]))
            for item in sorted(refs["queries"], key=lambda item: item["query_id"])]
    summary = {"schema_version": 1, "benchmark_id": refs["benchmark_id"], "model": answers["model"],
               "evaluator_sha256": sha256(Path(__file__).read_bytes()),
               "references_sha256": sha256(references.read_bytes()),
               "model_input_sha256": sha256(model_input.read_bytes()),
               "responses_sha256": sha256(responses.read_bytes()), "limitations": LIMITATIONS,
               **summarize(rows),
               "by_case_type": {key: summarize([row for row in rows if row["case_type"] == key])
                                for key in sorted({row["case_type"] for row in rows})},
               "by_split": {key: summarize([row for row in rows if row["split"] == key])
                            for key in sorted({row["split"] for row in rows})}}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: canonical_json(value).decode().strip() if isinstance(value, list) else value
                         for key, value in row.items()})
    write_bundle(out, {"per_item.csv": stream.getvalue().encode("utf-8-sig"),
                       "summary.json": canonical_json(summary)}, (references, model_input, responses))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--model-input", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = evaluate(args.references, args.model_input, args.responses, args.out)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(f"Expected={summary['total_expected']} received={summary['total_received']} "
          f"missing={summary['missing']} errors={summary['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
