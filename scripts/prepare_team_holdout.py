"""Offline, loss-preserving TEAM HOLDOUT import (Python standard library only)."""
from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile


DEFAULT_RAW = Path("data/team/holdout/questions/raw")
DEFAULT_OUT = Path("data/team/holdout/normalized")
ALIASES = {
    "query_id": ("query_id", "question_id", "id вопроса", "ид вопроса"),
    "query": ("query", "question", "вопрос", "текст вопроса"),
    "difficulty": ("difficulty", "сложность", "уровень сложности"),
    "answer_hint": ("answer_hint", "подсказка для проверки", "подсказка", "ответ", "эталонный ответ"),
    "document_id": ("document_id", "doc_id", "id документа", "ид документа"),
    "positive_section_refs": ("section_ref", "positive_section_refs", "правильный section_ref", "article_tag", "правильные section_refs"),
    "status": ("status", "статус"),
    "human_checked": ("human_checked", "проверено человеком"),
    "hard_negative_refs": ("hard_negative_refs", "hard_negatives"),
}
DOC_PATTERN = re.compile(r"(?<![a-z0-9])doc[_-]?(\d{4})(?!\d)", re.I)
REF_PATTERN = re.compile(r"[^\W\d_][\w.-]*\Z", re.UNICODE)


def clean(value):
    return "" if value is None else str(value).strip()


def key(value):
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", clean(value)).casefold()).strip()


ALIAS_MAP = {key(alias): field for field, aliases in ALIASES.items() for alias in aliases}


def column(value):
    normalized = key(value)
    if normalized in ALIAS_MAP:
        return ALIAS_MAP[normalized]
    match = re.fullmatch(r"hard negative(?: (\d+))?(?: (section ref|document id))?", normalized)
    if match:
        return f"hn_{match[1] or '1'}_{'document_id' if match[2] == 'document id' else 'section_ref'}"
    match = re.fullmatch(r"похожий неверный(?: (\d+))? (section ref|document id)(?: (\d+))?", normalized)
    if match:
        return f"hn_{match[1] or match[3] or '1'}_{match[2].replace(' ', '_')}"
    return None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def issue(kind, file, sheet="", row=None, **details):
    return {"kind": kind, "source_file": file, "source_sheet": sheet, "source_row": row, **details}


def header(row):
    mapping = {i: column(cell) for i, cell in enumerate(row) if column(cell)}
    return mapping if "query" in mapping.values() and "positive_section_refs" in mapping.values() else None


def read_csv(path):
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    else:
        try:
            data.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            encoding = "cp1251"
    content = data.decode(encoding)
    candidates = []
    for delimiter in (",", ";", "\t", "|"):
        try:
            reader = csv.reader(io.StringIO(content, newline=""), delimiter=delimiter, strict=True)
            rows = []
            for row in reader:
                # Physical start line, including quoted multiline fields.
                start = rows[-1][2] + 1 if rows else 1
                rows.append((start, row, reader.line_num))
            score = max((len(header(row) or {}) for _, row, _ in rows), default=0)
            candidates.append((score, delimiter, rows))
        except csv.Error:
            continue
    if not candidates or max(c[0] for c in candidates) == 0:
        raise ValueError("CSV header/delimiter could not be identified")
    _, delimiter, rows = max(candidates, key=lambda c: c[0])
    return [("", [(n, row) for n, row, _ in rows], [])], {"encoding": encoding, "delimiter": delimiter}


def children(node, tag):
    return [child for child in node if child.tag.rsplit("}", 1)[-1] == tag]


def text_nodes(node):
    # Ignore phonetic annotations; concatenate rich-text runs.
    tag = node.tag.rsplit("}", 1)[-1]
    if tag in {"rPh", "phoneticPr"}:
        return ""
    if tag == "t":
        return node.text or ""
    return "".join(text_nodes(child) for child in node)


def read_xlsx(path):
    sheets = []
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [text_nodes(si) for si in children(ET.fromstring(archive.read("xl/sharedStrings.xml")), "si")]
        relations = {r.get("Id"): r for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.iter():
            if sheet.tag.rsplit("}", 1)[-1] != "sheet":
                continue
            name = sheet.get("name", "")
            rid = next((v for k, v in sheet.attrib.items() if k.rsplit("}", 1)[-1] == "id"), None)
            relation = relations[rid]
            if relation.get("TargetMode") == "External":
                raise ValueError("External worksheet relationship")
            target = relation.get("Target", "")
            member = posixpath.normpath(target.lstrip("/") if target.startswith("/") else "xl/" + target)
            if not member.startswith("xl/"):
                raise ValueError("Worksheet target outside workbook")
            rows, problems = [], []
            xml = ET.fromstring(archive.read(member))
            for row in xml.iter():
                if row.tag.rsplit("}", 1)[-1] != "row":
                    continue
                number = int(row.get("r", len(rows) + 1))
                values = {}
                for cell in children(row, "c"):
                    address = cell.get("r", "")
                    letters = re.match(r"[A-Z]+", address)
                    if not letters:
                        raise ValueError("Cell without a valid address")
                    index = 0
                    for letter in letters[0]:
                        index = index * 26 + ord(letter) - ord("A") + 1
                    v = children(cell, "v")
                    value = v[0].text or "" if v else ""
                    kind = cell.get("t")
                    if kind == "s":
                        value = strings[int(value)]
                    elif kind == "inlineStr":
                        value = text_nodes(cell)
                    if kind == "e" or children(cell, "f"):
                        problems.append(issue("unsupported_cell", path.name, name, number, cell=address,
                                              reason="Excel error or formula (not evaluated)", value=value))
                    values[index - 1] = value
                cells = [values.get(i, "") for i in range(max(values, default=-1) + 1)]
                rows.append((number, cells))
            sheets.append((name, rows, problems))
    return sheets, {"encoding": "OOXML"}


def document_ids(value):
    return sorted({"doc_" + m[1] for m in DOC_PATTERN.finditer(value)})


def refs(value):
    value = clean(value)
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(v, str) for v in parsed):
            raise ValueError("refs JSON must be an array of strings")
        return [clean(v) for v in parsed if clean(v)]
    return [part.strip() for part in re.split(r"[;,|\r\n]+", value) if part.strip()]


def provisional(value):
    normalized = key(value)
    return any(marker in normalized for marker in ("чернов", "согласован", "draft", "provisional", "pending", "review"))


def normalize(fields, file, sheet, row, issues, context=""):
    errors = []
    explicit_doc = fields.get("document_id", "")
    inferred = document_ids(file) or document_ids(fields.get("query_id", ""))
    doc = explicit_doc or (inferred[0] if len(inferred) == 1 else "")
    if not explicit_doc and doc:
        issues.append(issue("inferred_document_id", file, sheet, row, document_id=doc))
    if explicit_doc and inferred and explicit_doc not in inferred:
        errors.append("document_id conflicts with source filename/query_id")
    try:
        positives = refs(fields.get("positive_section_refs", ""))
    except (ValueError, json.JSONDecodeError) as error:
        positives = [fields.get("positive_section_refs", "")]
        errors.append(str(error))
    negatives = []
    slots = sorted({k.split("_")[1] for k in fields if k.startswith("hn_")}, key=int)
    for slot in slots:
        ndoc = fields.get(f"hn_{slot}_document_id", "")
        raw_ref = fields.get(f"hn_{slot}_section_ref", "")
        if not ndoc and not raw_ref:
            continue
        try:
            negative_refs = refs(raw_ref)
        except ValueError as error:
            negative_refs = [raw_ref]
            errors.append(str(error))
        if ndoc and not negative_refs:
            errors.append(f"hard_negative_{slot}: document_id without section_ref")
            negatives.append({"document_id": ndoc, "section_ref": ""})
        for ref in negative_refs:
            if ":" in ref:
                ref_doc, ref = ref.split(":", 1)
                if ndoc and ndoc != ref_doc:
                    errors.append(f"hard_negative_{slot}: conflicting document_id")
            else:
                ref_doc = ndoc or doc
            negatives.append({"document_id": ref_doc, "section_ref": ref})
    aggregate = fields.get("hard_negative_refs", "")
    if aggregate:
        try:
            parsed = json.loads(aggregate) if aggregate.startswith("[") else refs(aggregate)
            if not isinstance(parsed, list):
                raise ValueError("hard_negatives must be an array")
            for entry in parsed:
                if isinstance(entry, dict):
                    negatives.append({"document_id": clean(entry.get("document_id")) or doc,
                                      "section_ref": clean(entry.get("section_ref"))})
                elif isinstance(entry, str):
                    ndoc, ref = entry.split(":", 1) if ":" in entry else (doc, entry)
                    negatives.append({"document_id": ndoc, "section_ref": ref})
                else:
                    raise ValueError("Invalid hard negative entry")
        except (ValueError, TypeError) as error:
            errors.append(str(error))
    result = {"query_id": fields.get("query_id", ""), "query": fields.get("query", ""),
              "difficulty": fields.get("difficulty", ""), "answer_hint": fields.get("answer_hint", ""),
              "document_id": doc, "positive_section_refs": positives, "hard_negatives": negatives,
              "source_file": file, "source_row": row, "status": "approved"}
    for field in ("query_id", "query", "document_id"):
        if not result[field]:
            errors.append(f"Missing {field}")
    if doc and not re.fullmatch(r"doc_\d{4}", doc):
        errors.append("Malformed document_id")
    if not positives:
        errors.append("Missing positive_section_refs")
    for ref in positives + [n["section_ref"] for n in negatives]:
        if not REF_PATTERN.fullmatch(ref):
            errors.append(f"Malformed section_ref: {ref}")
    for negative in negatives:
        if not re.fullmatch(r"doc_\d{4}", negative["document_id"]):
            errors.append("Missing/malformed hard negative document_id")
        if negative["document_id"] == doc and negative["section_ref"] in positives:
            errors.append("Hard negative overlaps a positive ref")
    status = fields.get("status", "")
    approved_values = {"approved", "утвержден", "утверждён", "согласовано", "согласован", "готово", "готов"}
    file_draft = provisional(file + " " + sheet + " " + context)
    human_no = key(fields.get("human_checked", "")) in {"нет", "no", "false", "0"}
    if file_draft or human_no or (status and key(status) not in {key(v) for v in approved_values}):
        result["status"] = "provisional"
    if status and key(status) not in {key(v) for v in approved_values} and not provisional(status):
        issues.append(issue("unknown_status", file, sheet, row, value=status))
    if errors:
        result["status"] = "provisional"
        issues.append(issue("unresolved_row", file, sheet, row, reasons=sorted(set(errors)), fields=fields))
    issues.append(issue("row_status", file, sheet, row, raw_status=status,
                        human_checked=fields.get("human_checked", ""), status=result["status"]))
    return result


def find_duplicates(records, threshold):
    ids, docs = {}, {}
    for index, record in enumerate(records):
        if record["query_id"]:
            ids.setdefault(record["query_id"], []).append(index)
        if record["document_id"] and record["query"]:
            docs.setdefault(record["document_id"], []).append(index)
    id_groups = [{"query_id": qid, "record_indices": indices} for qid, indices in ids.items() if len(indices) > 1]
    exact, near = [], []
    for doc, indices in docs.items():
        for position, left in enumerate(indices):
            a = key(records[left]["query"]).replace("ё", "е")
            for right in indices[position + 1:]:
                b = key(records[right]["query"]).replace("ё", "е")
                similarity = SequenceMatcher(None, a, b, autojunk=False).ratio()
                entry = {"document_id": doc, "record_indices": [left, right], "similarity": round(similarity, 6)}
                if a == b:
                    exact.append(entry)
                elif similarity >= threshold:
                    near.append(entry)
    return {"query_id_groups": id_groups, "exact_query_pairs": exact, "near_query_pairs": near}


def select_questions(records, mode="approved-only"):
    """Explicit status selection for future benchmark adapters; never deduplicates."""
    if mode not in {"approved-only", "all"}:
        raise ValueError("mode must be approved-only or all")
    return [r for r in records if mode == "all" or r["status"] == "approved"]


def prepare_holdout(raw_dir=DEFAULT_RAW, out_dir=DEFAULT_OUT, *, near_threshold=0.92):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    if not 0 < near_threshold <= 1:
        raise ValueError("near_threshold must be in (0, 1]")
    if not raw_dir.is_dir():
        raise ValueError("Raw directory does not exist")
    sources = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"})
    if not sources:
        raise ValueError("No CSV/XLSX source files")
    if raw_dir.resolve() == out_dir.resolve() or raw_dir.resolve() in out_dir.resolve().parents:
        raise ValueError("Output directory must be outside raw directory")
    records, issues, source_metadata, record_locations = [], [], [], []
    for path in sources:
        meta = {"source_file": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size,
                "imported_questions": 0, "sheets": []}
        source_metadata.append(meta)
        try:
            sheets, format_meta = read_csv(path) if path.suffix.lower() == ".csv" else read_xlsx(path)
            meta.update(format_meta)
        except (ValueError, OSError, KeyError, IndexError, zipfile.BadZipFile, ET.ParseError) as error:
            issues.append(issue("unresolved_file", path.name, reason=str(error)))
            continue
        for sheet, rows, cell_issues in sheets:
            issues.extend(cell_issues)
            mapping, header_row = None, None
            context = ""
            sheet_meta = {"name": sheet, "blank_rows": 0, "imported_questions": 0}
            meta["sheets"].append(sheet_meta)
            for number, cells in rows:
                if not any(clean(v) for v in cells):
                    sheet_meta["blank_rows"] += 1
                    continue
                detected = header(cells)
                if detected:
                    mapping, header_row = detected, number
                    sheet_meta.setdefault("headers", []).append({"source_row": number,
                        "columns": {str(i + 1): {"raw": cells[i], "field": field} for i, field in mapping.items()},
                        "unmapped_columns": [v for i, v in enumerate(cells) if i not in mapping and clean(v)]})
                    if len(set(mapping.values())) != len(mapping):
                        issues.append(issue("ambiguous_header", path.name, sheet, number, columns=cells))
                    continue
                if mapping is None:
                    issues.append(issue("preamble_row", path.name, sheet, number, cells=cells))
                    context += " " + " ".join(clean(v) for v in cells)
                    continue
                fields = {field: clean(cells[i]) if i < len(cells) else "" for i, field in mapping.items()}
                ambiguous = len(set(mapping.values())) != len(mapping)
                header_width = next(len(v) for n, v in rows if n == header_row)
                width_error = len(cells) > header_width and any(clean(v) for v in cells[header_width:])
                if ambiguous or width_error:
                    issues.append(issue("malformed_row", path.name, sheet, number,
                                        reason="Ambiguous columns or excess cells", cells=cells))
                record = normalize(fields, path.name, sheet, number, issues, context)
                if ambiguous or width_error or any(p["source_row"] == number for p in cell_issues):
                    record["status"] = "provisional"
                    issues[-1]["status"] = "provisional"  # normalize's final row_status entry
                record_locations.append({"record_index": len(records), "source_file": path.name,
                                         "source_row": number, "source_sheet": sheet})
                records.append(record)
                meta["imported_questions"] += 1
                sheet_meta["imported_questions"] += 1
            if mapping is None and any(any(clean(v) for v in cells) for _, cells in rows):
                issues.append(issue("unresolved_sheet", path.name, sheet, reason="No recognizable header"))
        if sha256(path) != meta["sha256"]:
            raise ValueError("Source changed during import")
    duplicates = find_duplicates(records, near_threshold)
    counts = {"query_id_groups": len(duplicates["query_id_groups"]),
              "query_id_excess_rows": sum(len(g["record_indices"]) - 1 for g in duplicates["query_id_groups"]),
              "exact_query_pairs": len(duplicates["exact_query_pairs"]),
              "near_query_pairs": len(duplicates["near_query_pairs"])}
    locations = lambda kinds: {(i["source_file"], i["source_sheet"], i["source_row"]) for i in issues if i["kind"] in kinds}
    doc_ids = sorted({r["document_id"] for r in records if r["document_id"]})
    manifest = {"schema_version": 1, "source_file_count": len(sources), "imported_question_count": len(records),
                "document_count": len(doc_ids), "document_ids": doc_ids,
                "approved_count": sum(r["status"] == "approved" for r in records),
                "provisional_count": sum(r["status"] == "provisional" for r in records),
                "duplicate_counts": counts, "malformed_row_count": len(locations({"malformed_row", "unsupported_cell"})),
                "unresolved_row_count": len(locations({"unresolved_row"})),
                "unresolved_source_count": sum(i["kind"] in {"unresolved_file", "unresolved_sheet"} for i in issues),
                "source_files": source_metadata, "near_duplicate_threshold": near_threshold,
                "reference_validation": "syntax_only; corpus membership not checked",
                "status_policy": "Missing status is approved; draft/review filename, sheet or preamble, negative human_checked, unknown status and invalid rows are provisional",
                "deduplication_policy": "report_only; no rows removed; duplicate indices are zero-based gold line indices"}
    validation = {"summary": {k: v for k, v in manifest.items() if k != "source_files"},
                  "issues": issues, "duplicates": duplicates,
                  "record_locations": record_locations}
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = out_dir / "gold.jsonl"
    gold.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8", newline="\n")
    manifest["gold_sha256"] = sha256(gold)
    validation["summary"]["gold_sha256"] = manifest["gold_sha256"]
    for name, payload in (("manifest.json", manifest), ("validation.json", validation)):
        (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = ["# TEAM HOLDOUT validation", "", f"Source files: {len(sources)}; questions: {len(records)}; documents: {len(doc_ids)}.",
             f"Approved: {manifest['approved_count']}; provisional: {manifest['provisional_count']}.",
             f"Duplicates: {json.dumps(counts)}.",
             f"Malformed rows: {manifest['malformed_row_count']}; unresolved rows: {manifest['unresolved_row_count']}; unresolved sources: {manifest['unresolved_source_count']}.",
             "", "No duplicate rows removed. References checked for syntax only; corpus membership has not been checked.",
             "Missing status defaults to approved. Draft/review markers, human_checked=no, unknown status and invalid rows are provisional.",
             "All source row statuses, raw invalid rows, sheet provenance and duplicate record indices are in validation.json.",
             "", "## Sources", ""]
    for source in source_metadata:
        lines.append(f"- {source['source_file']}: {source['imported_questions']} questions; SHA256 `{source['sha256']}`")
    lines.extend(["", "## Findings", ""])
    for finding in issues:
        if finding["kind"] not in {"row_status", "preamble_row"}:
            lines.append("- " + json.dumps(finding, ensure_ascii=False))
    for kind, groups in duplicates.items():
        for group in groups:
            lines.append(f"- {kind}: {json.dumps(group, ensure_ascii=False)}")
    (out_dir / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--near-threshold", type=float, default=0.92)
    args = parser.parse_args()
    try:
        manifest = prepare_holdout(args.raw_dir, args.out, near_threshold=args.near_threshold)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Holdout import failed: {error}\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_files"}, ensure_ascii=False, indent=2))
    return 1 if manifest["unresolved_source_count"] or manifest["unresolved_row_count"] or manifest["malformed_row_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
