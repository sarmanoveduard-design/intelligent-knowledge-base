import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge_base.benchmark_models import BenchmarkCase, CorpusChunk
from knowledge_base.retrieval_benchmark import load_inputs, run_benchmark, write_report
from knowledge_base.team_benchmark import (
    TeamCorpusAdapter, TeamDatasetError, TeamGoldAdapter, canonical_json,
    chunk_id, extract_document_id, prepare_team_benchmark,
)


def csv_bytes(rows, fields=None, delimiter=","):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def corpus_rows(*refs):
    return [{"section_ref": ref, "chapter": "Глава 1", "article_title": "Статья, заголовок",
             "text": "Синтетический текст, с запятой\nи второй строкой."} for ref in refs]


def gold_row(**changes):
    row = {"query_id": "q1", "query": "Синтетический вопрос?", "difficulty": "easy",
           "answer_hint": "HINT_ONLY_NOT_RETRIEVAL", "document_id": "doc_0026", "section_ref": "s1",
           "hard_negative_refs": "doc_0030:s1", "source_document": "synthetic source"}
    row.update(changes)
    return row


TEAM_GOLD_HEADER = (
    "query_id;query;difficulty;answer_hint;document_id;section_ref;"
    "hard_negative_1_document_id;hard_negative_1_section_ref;"
    "hard_negative_2_document_id;hard_negative_2_section_ref;"
    "hard_negative_3_document_id;hard_negative_3_section_ref;source_document"
)


def numbered_gold_row(**changes):
    row = dict.fromkeys(TEAM_GOLD_HEADER.split(";"), "")
    row.update({key: value for key, value in gold_row().items() if key in row})
    row.update(changes)
    return row


def map_row(ref, embedded="да", covered_by=""):
    return {"section_ref": ref, "chapter": "Synthetic chapter", "article_title": "Synthetic title",
            "embedded": embedded, "point_id": "", "covered_by": covered_by}


class TeamAdapterTests(unittest.TestCase):
    def setUp(self):
        self.sources = {"law(doc_0026)_embeddable.csv": csv_bytes(corpus_rows("s1", "s2")),
                        "other(doc_0030)_embeddable.csv": csv_bytes(corpus_rows("s1"))}
        self.corpus = TeamCorpusAdapter().from_sources(self.sources)

    def gold_csv(self, *rows):
        return TeamGoldAdapter().from_source(".csv", csv_bytes(rows or [gold_row()]), self.corpus)

    def test_document_id_extraction(self):
        self.assertEqual(extract_document_id("Federal law(doc_0026)_embeddable.csv"), "doc_0026")
        self.assertEqual(extract_document_id("doc_0030_embeddable.csv"), "doc_0030")

    def test_invalid_document_filenames(self):
        for name in ("law_embeddable.csv", "(doc_26)_embeddable.csv", "(doc_00026)_embeddable.csv",
                     "(DOC_0026)_embeddable.csv", "badadoc_0026_embeddable.csv"):
            with self.subTest(name=name), self.assertRaises(TeamDatasetError):
                extract_document_id(name)

    def test_multiple_documents_and_same_section(self):
        self.assertEqual(len(self.corpus), 3)
        self.assertEqual({item.document_id for item in self.corpus}, {"doc_0026", "doc_0030"})
        self.assertEqual(len({item.chunk_id for item in self.corpus}), 3)
        self.assertIsInstance(self.corpus[0], CorpusChunk)
        self.assertEqual(self.corpus[0].metadata["source_file"], "law(doc_0026)_embeddable.csv")

    def test_duplicate_section_in_document(self):
        with self.assertRaisesRegex(TeamDatasetError, "duplicate_section_ref"):
            TeamCorpusAdapter().from_sources({"(doc_0026)_embeddable.csv": csv_bytes(corpus_rows("s1", "s1"))})

    def test_duplicate_section_across_files_for_same_document(self):
        sources = dict(self.sources)
        sources["copy(doc_0026)_embeddable.csv"] = csv_bytes(corpus_rows("s1"))
        with self.assertRaisesRegex(TeamDatasetError, "duplicate_section_ref"):
            TeamCorpusAdapter().from_sources(sources)

    def test_stable_collision_safe_chunk_ids(self):
        self.assertEqual(chunk_id("doc_0026", "s1"), chunk_id("doc_0026", " s1 "))
        self.assertNotEqual(chunk_id("doc_0026", "s1"), chunk_id("doc_0030", "s1"))
        self.assertNotEqual(chunk_id("doc_0026", "a:b"), chunk_id("doc_0026", "ab"))

    def test_does_not_guess_document_from_text(self):
        sources = {"no_id_embeddable.csv": csv_bytes(corpus_rows("doc_0026"))}
        with self.assertRaisesRegex(TeamDatasetError, "invalid_document_filename"):
            TeamCorpusAdapter().from_sources(sources)

    def test_filename_document_conflict_rejected(self):
        rows = corpus_rows("s1")
        rows[0]["document_id"] = "doc_0030"
        with self.assertRaisesRegex(TeamDatasetError, "filename_mismatch"):
            TeamCorpusAdapter().from_sources({"(doc_0026)_embeddable.csv": csv_bytes(rows)})

    def test_csv_parsing_bom_quotes_and_newlines(self):
        self.assertEqual(self.corpus[0].text, corpus_rows("s1")[0]["text"])
        self.assertEqual(self.corpus[0].metadata["article_title"], "Статья, заголовок")
        case = self.gold_csv(gold_row(query='Вопрос, "цитата"\nновая строка'))[0]
        self.assertEqual(case.text, 'Вопрос, "цитата"\nновая строка')

    def test_semicolon_csv_parsing(self):
        cases = TeamGoldAdapter().from_source(".csv", csv_bytes([gold_row()], delimiter=";"), self.corpus)
        self.assertEqual(cases[0].query_id, "q1")

    def test_csv_missing_columns_and_bad_row_width(self):
        for raw in (b"section_ref,text\ns1,t\n", b"section_ref,chapter,article_title,text\ns1,c,a,t,extra\n"):
            with self.subTest(raw=raw), self.assertRaises(TeamDatasetError):
                TeamCorpusAdapter().from_sources({"(doc_0026)_embeddable.csv": raw})

    def test_empty_corpus_text_and_section_rejected(self):
        for ref, text in (("", "text"), ("s1", "  ")):
            row = corpus_rows(ref)[0]
            row["text"] = text
            with self.assertRaises(TeamDatasetError):
                TeamCorpusAdapter().from_sources({"(doc_0026)_embeddable.csv": csv_bytes([row])})

    def test_csv_gold_to_internal_case(self):
        case = self.gold_csv()[0]
        self.assertIsInstance(case, BenchmarkCase)
        self.assertEqual(case.positives, (chunk_id("doc_0026", "s1"),))
        self.assertEqual(case.hard_negatives, (chunk_id("doc_0030", "s1"),))
        self.assertEqual(case.metadata["answer_hint"], "HINT_ONLY_NOT_RETRIEVAL")
        self.assertNotIn(case.metadata["answer_hint"], case.text)

    def test_jsonl_objects_and_cross_document_refs(self):
        row = {"query_id": "q-json", "query": "Вопрос?", "difficulty": "hard",
               "positive_refs": [{"document_id": "doc_0026", "section_ref": "s1"},
                                 {"document_id": "doc_0030", "section_ref": "s1"}],
               "hard_negative_refs": [{"document_id": "doc_0026", "section_ref": "s2"}],
               "answer_hint": "HINT_ONLY_NOT_RETRIEVAL"}
        case = TeamGoldAdapter().from_source(".jsonl", (json.dumps(row) + "\n\n").encode(), self.corpus)[0]
        self.assertEqual(len(case.positives), 2)
        self.assertEqual(case.difficulty, "hard")

    def test_jsonl_string_refs_and_explicit_default_document(self):
        row = {"query_id": "q", "query": "q", "difficulty": "medium", "document_id": "doc_0026",
               "positive_refs": ["s1"], "hard_negative_refs": ["doc_0030#s1"]}
        case = TeamGoldAdapter().from_source(".jsonl", json.dumps(row).encode(), self.corpus)[0]
        self.assertEqual(case.positives, (chunk_id("doc_0026", "s1"),))

    def test_csv_json_array_refs(self):
        rows = [gold_row(positive_refs=json.dumps([{"document_id": "doc_0026", "section_ref": "s1"}]),
                         hard_negative_refs=json.dumps([{"document_id": "doc_0030", "section_ref": "s1"}]))]
        self.assertEqual(self.gold_csv(*rows)[0].hard_negatives, (chunk_id("doc_0030", "s1"),))

    def test_csv_separate_negative_columns(self):
        row = gold_row(hard_negative_refs="", hard_negative_document_id="doc_0030", hard_negative_section_ref="s1")
        self.assertEqual(self.gold_csv(row)[0].hard_negatives, (chunk_id("doc_0030", "s1"),))

    def test_missing_positive(self):
        with self.assertRaisesRegex(TeamDatasetError, "unresolved_refs") as caught:
            self.gold_csv(gold_row(section_ref="absent"))
        self.assertEqual(caught.exception.unresolved_refs_count, 1)

    def test_missing_hard_negative(self):
        with self.assertRaisesRegex(TeamDatasetError, "unresolved_refs"):
            self.gold_csv(gold_row(hard_negative_refs="doc_0030:absent"))

    def test_missing_document_reference(self):
        with self.assertRaisesRegex(TeamDatasetError, "unresolved_refs"):
            self.gold_csv(gold_row(document_id="doc_9999"))

    def test_aggregate_unresolved_refs(self):
        with self.assertRaises(TeamDatasetError) as caught:
            self.gold_csv(gold_row(section_ref="absent", hard_negative_refs="doc_0030:missing"),
                          gold_row(query_id="q2", section_ref="absent2"))
        self.assertEqual(caught.exception.unresolved_refs_count, 3)

    def test_duplicate_query_id(self):
        with self.assertRaisesRegex(TeamDatasetError, "duplicate_query_id"):
            self.gold_csv(gold_row(), gold_row())

    def test_invalid_query_difficulty_and_document_id(self):
        for changes in ({"query_id": " "}, {"query": "  "}, {"difficulty": "expert"},
                        {"document_id": "doc_26"}, {"hard_negative_refs": "doc_bad:s1"}):
            with self.subTest(changes=changes), self.assertRaises(TeamDatasetError):
                self.gold_csv(gold_row(**changes))

    def test_overlap_and_duplicate_refs(self):
        for value in ("doc_0026:s1", "doc_0030:s1;doc_0030:s1"):
            with self.assertRaises(TeamDatasetError):
                self.gold_csv(gold_row(hard_negative_refs=value))

    def test_malformed_jsonl_and_empty_positive(self):
        for raw in (b"{broken}", b"[]", b'{"query_id":"q","query":"q","difficulty":"easy","positive_refs":[]}'):
            with self.subTest(raw=raw), self.assertRaises(TeamDatasetError):
                TeamGoldAdapter().from_source(".jsonl", raw, self.corpus)

    def test_unsupported_negative_column_not_silently_ignored(self):
        with self.assertRaisesRegex(TeamDatasetError, "unsupported_hard_negative_column"):
            self.gold_csv(gold_row(hard_negative_refs="", hard_negatives="doc_0030:missing"))


class TeamPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input with spaces"
        self.source.mkdir()
        for name, refs in (("law(doc_0026)_embeddable.csv", ("s2", "s1")),
                           ("law(doc_0030)_embeddable.csv", ("s1",))):
            (self.source / name).write_bytes(csv_bytes(corpus_rows(*refs)))
        self.gold = self.source / "gold_qa.csv"
        self.gold.write_bytes(csv_bytes([gold_row(), gold_row(query_id="q3", difficulty="hard"),
                                         gold_row(query_id="q2", difficulty="medium")]))
        self.out = self.root / "benchmark"

    def prepare(self):
        return prepare_team_benchmark(self.source, self.gold, self.out)

    def test_manifest_counts_and_exact_file_hashes(self):
        manifest = self.prepare()
        self.assertEqual(manifest["document_count"], 2)
        self.assertEqual(manifest["chunk_count"], 3)
        self.assertEqual(manifest["query_count"], 3)
        self.assertEqual(manifest["source_csv_count"], 3)
        self.assertEqual(manifest["difficulty_counts"], {"easy": 1, "medium": 1, "hard": 1})
        self.assertEqual(manifest["unresolved_refs_count"], 0)
        self.assertTrue(manifest["timestamp_utc"].endswith("+00:00"))
        for record in manifest["source_files"]:
            self.assertEqual(record["sha256"], hashlib.sha256((self.source / record["source_file"]).read_bytes()).hexdigest())
        self.assertEqual(manifest["corpus_snapshot_hash"], hashlib.sha256((self.out / "corpus.json").read_bytes()).hexdigest())
        self.assertEqual(manifest["gold_snapshot_hash"], hashlib.sha256((self.out / "gold.json").read_bytes()).hexdigest())

    def test_manifest_contains_no_texts_hints_or_absolute_paths(self):
        self.prepare()
        text = (self.out / "manifest.json").read_text(encoding="utf-8")
        for forbidden in ("Синтетический текст", "Синтетический вопрос", "HINT_ONLY_NOT_RETRIEVAL", str(self.root)):
            self.assertNotIn(forbidden, text)

    def test_identical_inputs_have_identical_canonical_snapshots(self):
        first = self.prepare()
        corpus, gold = (self.out / "corpus.json").read_bytes(), (self.out / "gold.json").read_bytes()
        second = self.prepare()
        self.assertEqual(corpus, (self.out / "corpus.json").read_bytes())
        self.assertEqual(gold, (self.out / "gold.json").read_bytes())
        self.assertEqual(corpus, canonical_json(json.loads(corpus)))
        self.assertEqual(first["corpus_snapshot_hash"], second["corpus_snapshot_hash"])
        self.assertEqual(first["gold_snapshot_hash"], second["gold_snapshot_hash"])
        self.assertNotIn(b"timestamp", corpus + gold)

    def test_row_and_input_file_order_do_not_change_snapshot(self):
        first = self.prepare()
        (self.source / "law(doc_0026)_embeddable.csv").write_bytes(csv_bytes(corpus_rows("s1", "s2")))
        self.gold.write_bytes(csv_bytes([gold_row(query_id="q2", difficulty="medium"),
                                        gold_row(query_id="q3", difficulty="hard"), gold_row()]))
        second = self.prepare()
        self.assertEqual(first["corpus_snapshot_hash"], second["corpus_snapshot_hash"])
        self.assertEqual(first["gold_snapshot_hash"], second["gold_snapshot_hash"])
        paths = list(self.source.glob("*_embeddable.csv"))
        self.assertEqual(TeamCorpusAdapter().load(paths), TeamCorpusAdapter().load(reversed(paths)))

    def test_invalid_refs_create_no_outputs_and_preserve_existing_outputs(self):
        self.gold.write_bytes(csv_bytes([gold_row(section_ref="missing")]))
        with self.assertRaises(TeamDatasetError):
            self.prepare()
        self.assertFalse(self.out.exists())
        self.gold.write_bytes(csv_bytes([gold_row()]))
        self.prepare()
        before = {path.name: path.read_bytes() for path in self.out.iterdir()}
        self.gold.write_bytes(csv_bytes([gold_row(section_ref="missing")]))
        with self.assertRaises(TeamDatasetError):
            self.prepare()
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.out.iterdir()})

    def test_map_coverage_and_manifest(self):
        path = self.source / "law(doc_0026)_embedding_map.csv"
        path.write_bytes(csv_bytes([map_row("s1"), map_row("s2")]))
        manifest = self.prepare()
        self.assertEqual(manifest["embedding_map_csv_count"], 1)
        self.assertEqual(manifest["source_csv_count"], 4)
        path.write_bytes(csv_bytes([map_row("s1")]))
        with self.assertRaisesRegex(TeamDatasetError, "coverage_mismatch"):
            self.prepare()

    def test_real_map_structural_refs_do_not_create_chunks(self):
        path = self.source / "law(doc_0026)_embedding_map.csv"
        rows = [map_row("covered_parent", "нет", "s1"), map_row("oversized_parent", "нет"),
                map_row("s1", "да"), map_row("s2", "да")]
        path.write_bytes(csv_bytes(rows, delimiter=";"))
        manifest = self.prepare()
        chunks = json.loads((self.out / "corpus.json").read_bytes())["chunks"]
        self.assertEqual(manifest["chunk_count"], 3)
        self.assertEqual({row["section_ref"] for row in chunks}, {"s1", "s2"})
        self.assertEqual(len(chunks), 3)  # same s1 in a second document is allowed

    def test_map_embedded_normalizes_case_and_outer_whitespace(self):
        path = self.source / "law(doc_0026)_embedding_map.csv"
        path.write_bytes(csv_bytes([map_row("s1", " ДА "), map_row("s2", "да"),
                                   map_row("parent", " НеТ ", " s2 ")]))
        self.assertEqual(self.prepare()["chunk_count"], 3)

    def test_map_unknown_embedded_values_rejected(self):
        path = self.source / "law(doc_0026)_embedding_map.csv"
        for value in ("", "yes", "no", "true", "1", "да/нет"):
            with self.subTest(value=value):
                path.write_bytes(csv_bytes([map_row("s1", value), map_row("s2")]))
                with self.assertRaisesRegex(TeamDatasetError, "invalid_embedding_map_embedded"):
                    self.prepare()

    def test_map_duplicate_refs_rejected_including_nonembedded(self):
        path = self.source / "law(doc_0026)_embedding_map.csv"
        for extras in ([map_row(" s1 ", "нет")], [map_row("parent", "нет"), map_row("parent", "нет")]):
            path.write_bytes(csv_bytes([map_row("s1"), map_row("s2")] + extras))
            with self.assertRaisesRegex(TeamDatasetError, "duplicate_embedding_map_section_ref"):
                self.prepare()

    def test_map_covered_by_requires_embedded_target_in_same_document(self):
        path = self.source / "law(doc_0026)_embedding_map.csv"
        for target in ("absent", "parent", "other_parent", "doc_0030:s1"):
            with self.subTest(target=target):
                path.write_bytes(csv_bytes([map_row("s1"), map_row("s2"),
                                           map_row("parent", "нет", target), map_row("other_parent", "нет")]))
                with self.assertRaisesRegex(TeamDatasetError, "invalid_embedding_map_covered_by"):
                    self.prepare()
        self.assertFalse(self.out.exists())

    def test_map_cannot_cover_ref_only_embedded_in_other_document(self):
        (self.source / "law(doc_0030)_embeddable.csv").write_bytes(csv_bytes(corpus_rows("s1", "foreign")))
        (self.source / "law(doc_0026)_embedding_map.csv").write_bytes(csv_bytes(
            [map_row("s1"), map_row("s2"), map_row("parent", "нет", "foreign")]))
        with self.assertRaisesRegex(TeamDatasetError, "invalid_embedding_map_covered_by"):
            self.prepare()

    def test_map_embedded_set_must_match_corpus_exactly(self):
        path = self.source / "law(doc_0026)_embedding_map.csv"
        for rows in ([map_row("s1"), map_row("s2", "нет")],
                     [map_row("s1"), map_row("s2"), map_row("extra")]):
            path.write_bytes(csv_bytes(rows))
            with self.assertRaisesRegex(TeamDatasetError, "coverage_mismatch"):
                self.prepare()

    def test_map_requires_real_columns(self):
        (self.source / "law(doc_0026)_embedding_map.csv").write_bytes(csv_bytes([{"section_ref": "s1"}]))
        with self.assertRaisesRegex(TeamDatasetError, "missing_or_duplicate_csv_columns"):
            self.prepare()

    def test_jsonl_preparation_and_loader_compatibility(self):
        self.gold = self.source / "gold_qa.jsonl"
        row = {"query_id": "q1", "query": "query only", "difficulty": "easy",
               "positive_refs": [{"document_id": "doc_0026", "section_ref": "s1"}],
               "hard_negative_refs": ["doc_0030:s1"], "answer_hint": "HINT_ONLY_NOT_RETRIEVAL"}
        self.gold.write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest = self.prepare()
        self.assertEqual(manifest["source_csv_count"], 2)
        inputs = load_inputs(self.out / "corpus.json", self.out / "gold.json")
        self.assertEqual(inputs.corpus_hash, manifest["corpus_snapshot_hash"])
        self.assertEqual(inputs.gold_hash, manifest["gold_snapshot_hash"])
        self.assertEqual(inputs.chunks[0].document_id, "doc_0026")
        self.assertEqual(inputs.queries[0].query_id, "q1")
        self.assertEqual(inputs.queries[0].metadata["answer_hint"], "HINT_ONLY_NOT_RETRIEVAL")

    def test_answer_hint_never_passed_to_provider_or_report(self):
        from test_retriever import FakeSemanticEmbeddingProvider
        self.prepare()
        inputs = load_inputs(self.out / "corpus.json", self.out / "gold.json")
        provider = FakeSemanticEmbeddingProvider()
        with patch.object(provider, "embed_query", wraps=provider.embed_query) as query:
            result = run_benchmark(provider, inputs)
        self.assertEqual([call.args[0] for call in query.call_args_list], [case.text for case in inputs.queries])
        report = write_report(result, self.root / "reports")
        for path in report.iterdir():
            self.assertNotIn("HINT_ONLY_NOT_RETRIEVAL", path.read_text(encoding="utf-8"))

    def test_cli_success_and_sanitized_validation_failure(self):
        from scripts.prepare_team_benchmark import main
        args = ["prepare_team_benchmark.py", "--corpus-dir", str(self.source), "--gold", str(self.gold), "--out", str(self.out)]
        with patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(), 0)
            self.assertIn("documents=2", output.getvalue())
        self.gold.write_bytes(csv_bytes([gold_row(section_ref="missing")]))
        with patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(), 1)
            self.assertIn("unresolved_refs", output.getvalue())
            self.assertNotIn(str(self.root), output.getvalue())


class TeamNumberedGoldTests(unittest.TestCase):
    def setUp(self):
        self.corpus = TeamCorpusAdapter().from_sources({
            "(doc_0026)_embeddable.csv": csv_bytes(corpus_rows("s1", "s2")),
            "(doc_0030)_embeddable.csv": csv_bytes(corpus_rows("s1")),
            "(doc_0040)_embeddable.csv": csv_bytes(corpus_rows("s1")),
        })

    def parse(self, row):
        raw = csv_bytes([row], delimiter=";")
        # Assert the exact real header, not only an equivalent set of field names.
        self.assertEqual(raw.decode("utf-8-sig").splitlines()[0], TEAM_GOLD_HEADER)
        return TeamGoldAdapter().from_source(".csv", raw, self.corpus)[0]

    def test_real_header_supports_zero_to_three_hard_negatives(self):
        refs = [("doc_0026", "s2"), ("doc_0030", "s1"), ("doc_0040", "s1")]
        for count in range(4):
            row = numbered_gold_row()
            for slot, (document, section) in enumerate(refs[:count], 1):
                row[f"hard_negative_{slot}_document_id"] = document
                row[f"hard_negative_{slot}_section_ref"] = section
            with self.subTest(count=count):
                case = self.parse(row)
                self.assertEqual(set(case.hard_negatives), {chunk_id(doc, ref) for doc, ref in refs[:count]})
                self.assertEqual(case.metadata["answer_hint"], "HINT_ONLY_NOT_RETRIEVAL")
                self.assertNotIn("HINT_ONLY_NOT_RETRIEVAL", case.text)

    def test_half_filled_numbered_pair_rejected_for_each_slot(self):
        for slot in range(1, 4):
            for field, value in (("document_id", "doc_0030"), ("section_ref", "s1")):
                with self.subTest(slot=slot, field=field):
                    row = numbered_gold_row(**{f"hard_negative_{slot}_{field}": value})
                    with self.assertRaisesRegex(TeamDatasetError, "incomplete_hard_negative_pair"):
                        self.parse(row)

    def test_blank_pairs_and_noncontiguous_slots(self):
        row = numbered_gold_row(hard_negative_1_document_id=" ", hard_negative_1_section_ref=" ",
                                hard_negative_3_document_id=" doc_0030 ", hard_negative_3_section_ref=" s1 ")
        self.assertEqual(self.parse(row).hard_negatives, (chunk_id("doc_0030", "s1"),))

    def test_numbered_duplicate_refs_rejected(self):
        row = numbered_gold_row(hard_negative_1_document_id="doc_0030", hard_negative_1_section_ref="s1",
                                hard_negative_3_document_id="doc_0030", hard_negative_3_section_ref=" s1 ")
        with self.assertRaisesRegex(TeamDatasetError, "duplicate_ref"):
            self.parse(row)

    def test_numbered_positive_overlap_rejected(self):
        row = numbered_gold_row(hard_negative_1_document_id="doc_0026", hard_negative_1_section_ref="s1")
        with self.assertRaisesRegex(TeamDatasetError, "positive_hard_negative_overlap"):
            self.parse(row)

    def test_numbered_refs_must_exist(self):
        for document, section in (("doc_0030", "missing"), ("doc_9999", "s1")):
            with self.subTest(document=document):
                row = numbered_gold_row(hard_negative_2_document_id=document, hard_negative_2_section_ref=section)
                with self.assertRaisesRegex(TeamDatasetError, "unresolved_refs"):
                    self.parse(row)

    def test_numbered_document_id_format(self):
        row = numbered_gold_row(hard_negative_1_document_id="doc_30", hard_negative_1_section_ref="s1")
        with self.assertRaisesRegex(TeamDatasetError, "invalid_document_id"):
            self.parse(row)

    def test_aggregate_and_numbered_are_ambiguous(self):
        for aggregate in ("hard_negative_refs", "hard_negative_section_refs"):
            row = numbered_gold_row(hard_negative_1_document_id="doc_0030", hard_negative_1_section_ref="s1")
            row[aggregate] = "doc_0040:s1"
            with self.subTest(aggregate=aggregate), self.assertRaisesRegex(TeamDatasetError, "ambiguous_hard_negative_columns"):
                TeamGoldAdapter().from_source(".csv", csv_bytes([row], delimiter=";"), self.corpus)

    def test_aggregate_format_with_blank_numbered_columns_remains_supported(self):
        row = numbered_gold_row()
        row["hard_negative_refs"] = "doc_0030:s1"
        case = TeamGoldAdapter().from_source(".csv", csv_bytes([row], delimiter=";"), self.corpus)[0]
        self.assertEqual(case.hard_negatives, (chunk_id("doc_0030", "s1"),))

    def test_numbered_format_with_blank_aggregate_remains_supported(self):
        row = numbered_gold_row(hard_negative_1_document_id="doc_0030", hard_negative_1_section_ref="s1")
        row["hard_negative_refs"] = "  "
        case = TeamGoldAdapter().from_source(".csv", csv_bytes([row], delimiter=";"), self.corpus)[0]
        self.assertEqual(case.hard_negatives, (chunk_id("doc_0030", "s1"),))


if __name__ == "__main__":
    unittest.main()
