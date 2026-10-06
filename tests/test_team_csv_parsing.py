import csv
import io
import unittest

from knowledge_base.team_benchmark import TeamCorpusAdapter, TeamDatasetError, _csv_rows


CORPUS_HEADER = "section_ref;chapter;article_title;text"
MAP_HEADER = "section_ref;chapter;article_title;embedded;point_id;covered_by"
CORPUS_FIELDS = set(CORPUS_HEADER.split(";"))


def encode_csv(header, rows, *, delimiter=";", bom=False):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig" if bom else "utf-8")


class TeamCSVParsingTests(unittest.TestCase):
    def test_real_semicolon_header_with_long_quoted_multiline_text(self):
        text = ('Много, запятых, внутри, текста; с "кавычками"\n' * 400) + "Конец."
        raw = encode_csv(CORPUS_HEADER.split(";"), [["s1", "chapter", "title", text]])
        self.assertEqual(raw.decode().splitlines()[0], CORPUS_HEADER)
        self.assertGreater(len(raw.decode()), 8192)
        # The old body sample ends inside this row's quoted multiline text.
        chunks = TeamCorpusAdapter().from_sources({"(doc_0026)_embeddable.csv": raw})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_exact_real_embedding_map_header(self):
        rows = [["s1", "chapter", 'title, with; "quotes"\nand newline', "да", "", ""],
                ["parent", "chapter", "title", "нет", "", "s1"]]
        raw = encode_csv(MAP_HEADER.split(";"), rows, bom=True)
        self.assertEqual(raw.decode("utf-8-sig").splitlines()[0], MAP_HEADER)
        parsed = _csv_rows(raw, set(MAP_HEADER.split(";")))
        self.assertEqual(parsed[0]["article_title"], rows[0][2])
        self.assertEqual(parsed[1]["covered_by"], "s1")

    def test_comma_csv(self):
        text = 'a, b; c\n"quoted"'
        raw = encode_csv(CORPUS_HEADER.split(";"), [["s1", "c", "t", text]], delimiter=",")
        self.assertEqual(_csv_rows(raw, CORPUS_FIELDS)[0]["text"], text)

    def test_tab_csv(self):
        text = 'a, b; c\tinside text\n"quoted"'
        raw = encode_csv(CORPUS_HEADER.split(";"), [["s1", "c", "t", text]], delimiter="\t")
        self.assertEqual(_csv_rows(raw, CORPUS_FIELDS)[0]["text"], text)

    def test_utf8_bom_for_each_supported_delimiter(self):
        for delimiter in (";", ",", "\t"):
            with self.subTest(delimiter=delimiter):
                raw = encode_csv(CORPUS_HEADER.split(";"), [["s1", "глава", "статья", "текст"]],
                                 delimiter=delimiter, bom=True)
                self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
                self.assertEqual(_csv_rows(raw, CORPUS_FIELDS)[0]["text"], "текст")

    def test_malformed_row_width_too_many_or_too_few_columns(self):
        for row in ("s1;c;t", "s1;c;t;text;extra"):
            with self.subTest(row=row), self.assertRaisesRegex(TeamDatasetError, "invalid_csv_row_width"):
                _csv_rows((CORPUS_HEADER + "\n" + row + "\n").encode(), CORPUS_FIELDS)

    def test_invalid_body_quoting_rejected(self):
        for row in ('s1;c;t;"unclosed\nmore text', 's1;c;t;"closed"unexpected'):
            with self.subTest(row=row), self.assertRaisesRegex(TeamDatasetError, "^invalid_csv$"):
                _csv_rows((CORPUS_HEADER + "\n" + row).encode(), CORPUS_FIELDS)

    def test_duplicate_columns_rejected(self):
        raw = b"section_ref;chapter;article_title;text;text\ns1;c;t;a;b\n"
        with self.assertRaisesRegex(TeamDatasetError, "missing_or_duplicate_csv_columns"):
            _csv_rows(raw, CORPUS_FIELDS)

    def test_missing_or_unsupported_header_rejected(self):
        for header in ("section_ref;chapter;text", "section_ref|chapter|article_title|text", "", "unrelated"):
            with self.subTest(header=header), self.assertRaisesRegex(TeamDatasetError, "missing_or_duplicate_csv_columns"):
                _csv_rows((header + "\n").encode(), CORPUS_FIELDS)

    def test_ambiguous_header_rejected(self):
        # Both comma and semicolon produce unique columns containing section_ref.
        with self.assertRaisesRegex(TeamDatasetError, "ambiguous_csv_delimiter"):
            _csv_rows(b"section_ref;extra,section_ref\n", {"section_ref"})

    def test_invalid_utf8_rejected(self):
        with self.assertRaisesRegex(TeamDatasetError, "csv_requires_utf8"):
            _csv_rows(CORPUS_HEADER.encode() + b"\n\xff", CORPUS_FIELDS)

    def test_quoted_header_names_supported(self):
        raw = b'"section_ref";"chapter";"article_title";"text"\ns1;c;t;text\n'
        self.assertEqual(_csv_rows(raw, CORPUS_FIELDS)[0]["section_ref"], "s1")


if __name__ == "__main__":
    unittest.main()
