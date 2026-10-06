"""Synthetic fixtures only: no team data, models or network access."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import zipfile

spec = importlib.util.spec_from_file_location("prepare_team_holdout", Path(__file__).resolve().parents[1] / "scripts/prepare_team_holdout.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


def write_csv(path, rows, delimiter=";", encoding="utf-8-sig"):
    stream = io.StringIO(newline="")
    csv.writer(stream, delimiter=delimiter).writerows(rows)
    path.write_bytes(stream.getvalue().encode(encoding))


def write_xlsx(path, sheets, *, shared=False, formula=False):
    """Small OOXML packages exercise sparse cells, relationships and rich text."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    strings = []
    with zipfile.ZipFile(path, "w") as archive:
        workbook, relations = [], []
        for index, (name, rows) in enumerate(sheets, 1):
            workbook.append(f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>')
            relations.append(f'<Relationship Id="rId{index}" Target="worksheets/custom{index}.xml" Type="{rel}/worksheet"/>')
            xml_rows = []
            for number, row in rows:
                cells = []
                for col, value in enumerate(row):
                    if value is None:
                        continue
                    address = f"{chr(65 + col)}{number}"
                    if shared:
                        strings.append(value)
                        cell = f'<c r="{address}" t="s"><v>{len(strings)-1}</v></c>'
                    else:
                        cell = f'<c r="{address}" t="inlineStr"><is><r><t>{escape(value)}</t></r></is></c>'
                    if formula and number == 4 and col == 1:
                        cell = f'<c r="{address}" t="str"><f>"cached"</f><v>cached</v></c>'
                    cells.append(cell)
                xml_rows.append(f'<row r="{number}">{"".join(cells)}</row>')
            archive.writestr(f"xl/worksheets/custom{index}.xml", f'<worksheet xmlns="{ns}"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>{"".join(workbook)}</sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(relations)}</Relationships>')
        if shared:
            archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">' + "".join(f'<si><t>{escape(v)}</t></si>' for v in strings) + '</sst>')


HEADER = ["query_id", "query", "document_id", "section_ref", "hard_negative_1_document_id", "hard_negative_1_section_ref"]


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.out = self.root / "normalized"

    def run_import(self, **options):
        before = {p.name: p.read_bytes() for p in self.raw.iterdir()}
        manifest = importer.prepare_holdout(self.raw, self.out, **options)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.raw.iterdir()})
        records = [json.loads(line) for line in (self.out / "gold.jsonl").read_text(encoding="utf-8").splitlines()]
        report = json.loads((self.out / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["gold_sha256"], hashlib.sha256((self.out / "gold.jsonl").read_bytes()).hexdigest())
        self.assertEqual(set(p.name for p in self.out.iterdir()), {"gold.jsonl", "manifest.json", "validation.json", "VALIDATION.md"})
        return manifest, records, report

    def test_csv_delimiters_encodings_multiline_and_multiple_refs(self):
        for index, (delimiter, encoding) in enumerate(((";", "utf-8-sig"), (",", "utf-8"), ("\t", "utf-16"), ("|", "cp1251"))):
            write_csv(self.raw / f"source{index}.csv", [HEADER, [f"q{index}", "Вопрос, с\nдвумя строками?", "doc_0999", "clause_11_1; clause_11_2", "", ""]], delimiter, encoding)
        manifest, records, _ = self.run_import()
        self.assertEqual(manifest["imported_question_count"], 4)
        self.assertEqual(manifest["unresolved_row_count"], 0)
        self.assertEqual(manifest["approved_count"], 4)
        for record in records:
            self.assertEqual(record["query"], "Вопрос, с\nдвумя строками?")
            self.assertEqual(record["positive_section_refs"], ["clause_11_1", "clause_11_2"])
            self.assertEqual(record["hard_negatives"], [])
            self.assertEqual(record["source_row"], 2)

    def test_russian_xlsx_headers_preamble_sparse_blanks_and_draft(self):
        headings = ["query_id", "Вопрос", "Сложность", "Подсказка для проверки", "document_id", "Правильный section_ref", "Похожий неверный section_ref", "Статус"]
        row = ["q1", "Как работает правило?", "easy", "Подсказка", "doc_0999", "clause_1;clause_2", "clause_3;clause_4", "Черновик — на согласовании"]
        write_xlsx(self.raw / "synthetic.xlsx", [("Вопросы", [(1, ["Преамбула"]), (3, headings), (4, row), (1000, [None, None])])], shared=True)
        manifest, records, report = self.run_import()
        self.assertEqual(manifest["provisional_count"], 1)
        self.assertEqual(records[0]["source_row"], 4)
        self.assertEqual(records[0]["answer_hint"], "Подсказка")
        self.assertEqual(records[0]["hard_negatives"], [{"document_id": "doc_0999", "section_ref": "clause_3"}, {"document_id": "doc_0999", "section_ref": "clause_4"}])
        self.assertEqual(report["record_locations"][0]["source_sheet"], "Вопросы")

    def test_review_schema_document_inference_and_three_negatives(self):
        write_csv(self.raw / "synthetic_doc_0999_REVIEW.csv", [["query_id", "question", "article_tag", "hard_negative_1", "hard_negative_2", "hard_negative_3", "human_checked"], ["q01", "Вопрос?", "art_1", "art_2", "doc_0998:art_3", "", "НЕТ"]])
        manifest, records, report = self.run_import()
        self.assertEqual(manifest["document_ids"], ["doc_0999"])
        self.assertEqual(manifest["provisional_count"], 1)
        self.assertEqual(manifest["unresolved_row_count"], 0)
        self.assertEqual(records[0]["hard_negatives"][1], {"document_id": "doc_0998", "section_ref": "art_3"})
        self.assertTrue(any(i["kind"] == "inferred_document_id" for i in report["issues"]))

    def test_duplicates_reported_without_deleting_and_scoped_to_document(self):
        rows = [HEADER,
                ["same", "Как получить разрешение на работу?", "doc_0999", "a1", "", ""],
                ["same", "Как получить разрешение на работу!", "doc_0999", "a1", "", ""],
                ["near", "Как получить разрешение на работы?", "doc_0999", "a2", "", ""],
                ["other", "Как получить разрешение на работу?", "doc_0998", "a1", "", ""]]
        write_csv(self.raw / "source.csv", rows)
        manifest, records, report = self.run_import()
        self.assertEqual(len(records), 4)
        self.assertEqual(manifest["duplicate_counts"], {"query_id_groups": 1, "query_id_excess_rows": 1, "exact_query_pairs": 1, "near_query_pairs": 2})
        self.assertEqual(report["duplicates"]["query_id_groups"][0]["record_indices"], [0, 1])

    def test_invalid_rows_preserved_and_empty_negatives_valid(self):
        write_csv(self.raw / "source.csv", [HEADER,
            ["valid", "Вопрос?", "doc_0999", "a1", "", ""],
            ["missing", "", "", "", "", ""],
            ["partial", "Вопрос?", "doc_0999", "a1", "doc_0998", ""],
            ["overlap", "Вопрос?", "doc_0999", "a1", "", "a1"],
            ["width", "Вопрос?", "doc_0999", "a1", "", "", "EXTRA"]])
        manifest, records, _ = self.run_import()
        self.assertEqual(len(records), 5)
        self.assertEqual(manifest["unresolved_row_count"], 3)
        self.assertEqual(manifest["malformed_row_count"], 1)
        self.assertEqual(manifest["approved_count"], 1)
        self.assertEqual(len(importer.select_questions(records)), 1)
        self.assertEqual(len(importer.select_questions(records, "all")), 5)
        with self.assertRaises(ValueError):
            importer.select_questions(records, "invalid")

    def test_unknown_status_and_filename_draft_cannot_be_approved(self):
        write_csv(self.raw / "draft.csv", [HEADER + ["status"], ["q1", "Question?", "doc_0999", "a1", "", "", "approved"]])
        write_csv(self.raw / "source.csv", [HEADER + ["Статус"], ["q2", "Question?", "doc_0999", "a1", "", "", "неизвестно"]])
        manifest, records, report = self.run_import()
        self.assertEqual(manifest["provisional_count"], 2)
        self.assertTrue(any(i["kind"] == "unknown_status" for i in report["issues"]))

    def test_multiple_sheets_and_formula_cells(self):
        rows = [(1, HEADER), (4, ["q1", "Вопрос?", "doc_0999", "a1", "", ""])]
        write_xlsx(self.raw / "source.xlsx", [("Первый", rows), ("Второй", rows)], formula=True)
        manifest, records, report = self.run_import()
        self.assertEqual(len(records), 2)
        self.assertEqual(manifest["malformed_row_count"], 2)
        self.assertEqual(manifest["provisional_count"], 2)
        self.assertEqual([r["source_sheet"] for r in report["record_locations"]], ["Первый", "Второй"])

    def test_aggregate_json_refs_and_negatives(self):
        write_csv(self.raw / "source.csv", [["query_id", "query", "document_id", "positive_section_refs", "hard_negatives"], ["q1", "Вопрос?", "doc_0999", '["a1", "a2"]', '[{"document_id":"doc_0998","section_ref":"a1"}]']])
        manifest, records, _ = self.run_import()
        self.assertEqual(manifest["unresolved_row_count"], 0)
        self.assertEqual(records[0]["positive_section_refs"], ["a1", "a2"])
        self.assertEqual(records[0]["hard_negatives"][0]["document_id"], "doc_0998")

    def test_cyrillic_subclause_refs_are_valid_and_preamble_draft_is_retained(self):
        write_xlsx(self.raw / "source.xlsx", [("Questions", [(1, ["Черновик — на согласовании"]),
            (3, HEADER), (4, ["q1", "Вопрос?", "doc_0999", "clause_5_а;clause_5_б", "", "clause_5_в"])])])
        manifest, records, _ = self.run_import()
        self.assertEqual(manifest["unresolved_row_count"], 0)
        self.assertEqual(manifest["provisional_count"], 1)
        self.assertEqual(records[0]["positive_section_refs"], ["clause_5_а", "clause_5_б"])

    def test_broken_sources_reported_and_deterministic_outputs(self):
        (self.raw / "broken.xlsx").write_bytes(b"not an xlsx")
        write_csv(self.raw / "source.csv", [HEADER, ["q1", "Вопрос?", "doc_0999", "a1", "", ""]])
        manifest, _, _ = self.run_import()
        self.assertEqual(manifest["source_file_count"], 2)
        self.assertEqual(manifest["unresolved_source_count"], 1)
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        self.run_import()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_ambiguous_header_is_reported_and_raw_row_retained(self):
        write_csv(self.raw / "source.csv", [["query_id", "query", "Вопрос", "document_id", "section_ref"], ["q1", "First?", "Second?", "doc_0999", "a1"]])
        manifest, records, report = self.run_import()
        self.assertEqual(manifest["malformed_row_count"], 1)
        self.assertEqual(records[0]["status"], "provisional")
        self.assertTrue(any(i.get("cells", [])[1:3] == ["First?", "Second?"] for i in report["issues"]))

    def test_output_cannot_be_inside_raw_and_bad_threshold_rejected(self):
        write_csv(self.raw / "source.csv", [HEADER, ["q1", "Вопрос?", "doc_0999", "a1", "", ""]])
        with self.assertRaises(ValueError):
            importer.prepare_holdout(self.raw, self.raw / "output")
        with self.assertRaises(ValueError):
            importer.prepare_holdout(self.raw, self.out, near_threshold=0)

    def test_rich_text_ignores_phonetic_annotations(self):
        node = ET.fromstring('<si xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><r><t>Воп</t></r><r><t>рос</t></r><rPh><t>phonetic</t></rPh></si>')
        self.assertEqual(importer.text_nodes(node), "Вопрос")

    def test_csv_source_row_tracks_physical_multiline_and_blank_lines(self):
        write_csv(self.raw / "source.csv", [HEADER, ["q1", "Вопрос\nпродолжение?", "doc_0999", "a1", "", ""], [], ["q2", "Второй?", "doc_0999", "a2", "", ""]])
        _, records, _ = self.run_import()
        self.assertEqual([r["source_row"] for r in records], [2, 5])


if __name__ == "__main__":
    unittest.main()
