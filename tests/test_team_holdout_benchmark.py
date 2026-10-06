"""Synthetic HOLDOUT fixtures; no private inputs or provider calls."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_base.team_benchmark import TeamDatasetError, canonical_json, chunk_id, snapshot_hash
from knowledge_base.team_holdout_benchmark import prepare_team_holdout_benchmark


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def question(qid="q1", status="approved", **updates):
    row = {"query_id": qid, "query": "Синтетический вопрос?", "difficulty": "easy",
           "answer_hint": "Синтетическая подсказка", "document_id": "doc_0999",
           "positive_section_refs": ["s1", "s2"],
           "hard_negatives": [{"document_id": "doc_0998", "section_ref": "s1"}],
           "source_file": "synthetic.xlsx", "source_row": 4, "status": status}
    row.update(updates)
    return row


class HoldoutBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        for doc, refs in (("doc_0999", ["s1", "s2", "s3"]), ("doc_0998", ["s1"])):
            (self.corpus / f"synthetic({doc})_embeddable.csv").write_bytes(csv_bytes([
                {"section_ref": ref, "chapter": "Глава", "article_title": "Статья", "text": "Синтетический текст"}
                for ref in refs]))
            (self.corpus / f"synthetic({doc})_embedding_map.csv").write_bytes(csv_bytes([
                {"section_ref": ref, "chapter": "Глава", "article_title": "Статья", "embedded": "да", "point_id": "", "covered_by": ""}
                for ref in refs]))
        self.gold = self.root / "normalized" / "gold.jsonl"
        self.gold.parent.mkdir()
        self.rows = [question(), question("q2", "provisional", hard_negatives=[])]
        self.write_rows()
        self.out = self.root / "benchmark"

    def write_rows(self):
        self.gold.write_bytes(b"".join(canonical_json(row) for row in self.rows))

    def prepare(self, mode="approved-only"):
        return prepare_team_holdout_benchmark(self.corpus, self.gold, self.out, mode=mode)

    def test_default_approved_only_and_manifest_hashes(self):
        manifest = self.prepare()
        self.assertEqual(manifest["mode"], "approved-only")
        self.assertEqual(manifest["source_question_count"], 2)
        self.assertEqual(manifest["selected_question_count"], 1)
        self.assertEqual((manifest["approved_count"], manifest["provisional_count"]), (1, 1))
        self.assertEqual((manifest["selected_approved_count"], manifest["selected_provisional_count"]), (1, 0))
        self.assertEqual((manifest["document_count"], manifest["chunk_count"], manifest["unresolved_refs_count"]), (2, 4, 0))
        self.assertEqual(set(p.name for p in self.out.iterdir()), {"corpus.json", "gold.json", "manifest.json"})
        for name, field in (("corpus.json", "corpus_snapshot_hash"), ("gold.json", "gold_snapshot_hash")):
            self.assertEqual(manifest[field], snapshot_hash((self.out / name).read_bytes()))
        for source in manifest["source_files"]:
            path = self.gold if source["role"] == "normalized_gold" else self.corpus / source["source_file"]
            self.assertEqual(source["sha256"], snapshot_hash(path.read_bytes()))

    def test_all_multiple_positives_cross_document_negatives_and_metadata(self):
        manifest = self.prepare("all")
        rows = json.loads((self.out / "gold.json").read_bytes())["queries"]
        self.assertEqual(manifest["selected_question_count"], 2)
        self.assertEqual((manifest["selected_approved_count"], manifest["selected_provisional_count"]), (1, 1))
        self.assertEqual(set(rows[0]["positive_chunk_ids"]), {chunk_id("doc_0999", "s1"), chunk_id("doc_0999", "s2")})
        self.assertEqual(rows[0]["hard_negative_chunk_ids"], [chunk_id("doc_0998", "s1")])
        self.assertEqual(rows[1]["hard_negative_chunk_ids"], [])
        self.assertEqual(rows[1]["metadata"]["status"], "provisional")
        self.assertEqual(rows[0]["metadata"]["source_file"], "synthetic.xlsx")
        self.assertEqual(rows[0]["metadata"]["source_row"], 4)
        # Existing runner's format-only loader, without running retrieval.
        from knowledge_base.retrieval_benchmark import load_inputs
        inputs = load_inputs(self.out / "corpus.json", self.out / "gold.json")
        self.assertEqual(len(inputs.queries), 2)

    def test_unresolved_positive_or_negative_in_excluded_row_rejected(self):
        for update in ({"positive_section_refs": ["absent"]},
                       {"hard_negatives": [{"document_id": "doc_0998", "section_ref": "absent"}]}):
            with self.subTest(update=update):
                self.rows = [question(), question("q2", "provisional", **update)]
                self.write_rows()
                with self.assertRaises(TeamDatasetError) as caught:
                    self.prepare()
                self.assertEqual(caught.exception.code, "unresolved_refs")
                self.assertEqual(caught.exception.unresolved_refs_count, 1)
                self.assertFalse(self.out.exists())

    def test_invalid_schema_status_and_provenance_rejected(self):
        invalid = [question(status="draft"), question(status=[]), question(source_row=True),
                   question(source_row=0), question(source_file=""), question(document_id="bad"),
                   question(positive_section_refs="s1"), question(positive_section_refs=[]),
                   question(positive_section_refs=[4]), question(hard_negatives="s3"),
                   question(hard_negatives=[{"section_ref": "s3"}]), question(answer_hint=None),
                   question(query=""), question(difficulty="unknown"), question(positive_refs=["s1"])]
        missing = question()
        del missing["status"]
        invalid.extend([missing, [], "not a row"])
        for row in invalid:
            with self.subTest(row=row):
                self.rows = [question(), row]  # invalid rows cannot hide behind status filtering
                self.write_rows()
                with self.assertRaises(TeamDatasetError):
                    self.prepare()
                self.assertFalse(self.out.exists())

    def test_duplicate_ids_refs_and_positive_negative_overlap_use_existing_checks(self):
        for rows, code in (([question(), question()], "duplicate_query_id"),
                           ([question(positive_section_refs=["s1", "s1"])], "duplicate_ref"),
                           ([question(hard_negatives=[{"document_id": "doc_0999", "section_ref": "s1"}])], "positive_hard_negative_overlap")):
            with self.subTest(code=code):
                self.rows = rows
                self.write_rows()
                with self.assertRaises(TeamDatasetError) as caught:
                    self.prepare()
                self.assertEqual(caught.exception.code, code)

    def test_determinism_no_input_mutation_and_row_order_independent_snapshots(self):
        before = {p: p.read_bytes() for p in [self.gold, *self.corpus.iterdir()]}
        original = copy.deepcopy(self.rows)
        first = self.prepare("all")
        outputs = {p.name: p.read_bytes() for p in self.out.iterdir()}
        second = self.prepare("all")
        self.assertEqual(first, second)
        self.assertEqual(outputs, {p.name: p.read_bytes() for p in self.out.iterdir()})
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(self.rows, original)
        self.rows.reverse()
        self.write_rows()
        reordered = self.prepare("all")
        self.assertEqual(first["gold_snapshot_hash"], reordered["gold_snapshot_hash"])
        self.assertEqual(first["corpus_snapshot_hash"], reordered["corpus_snapshot_hash"])

    def test_invalid_sources_preserve_existing_outputs(self):
        self.prepare()
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        self.rows[0]["positive_section_refs"] = ["missing"]
        self.write_rows()
        with self.assertRaises(TeamDatasetError):
            self.prepare()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_map_required_for_each_document_and_map_coverage_validated(self):
        path = self.corpus / "synthetic(doc_0999)_embedding_map.csv"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(TeamDatasetError, "one_map_per_document"):
            self.prepare()
        path.write_bytes(original.replace(b"s2", b"missing"))
        with self.assertRaisesRegex(TeamDatasetError, "coverage_mismatch"):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_invalid_json_mode_empty_selection_and_input_overlap(self):
        with self.assertRaisesRegex(TeamDatasetError, "invalid_holdout_mode"):
            self.prepare("unknown")
        self.gold.write_bytes(b"{broken}")
        with self.assertRaisesRegex(TeamDatasetError, "invalid_holdout_jsonl"):
            self.prepare()
        self.rows = [question(status="provisional")]
        self.write_rows()
        with self.assertRaisesRegex(TeamDatasetError, "empty_holdout_selection"):
            self.prepare()
        for out in (self.corpus, self.corpus / "outputs", self.gold.parent):
            with self.assertRaisesRegex(TeamDatasetError, "overlaps_inputs"):
                prepare_team_holdout_benchmark(self.corpus, self.gold, out)


if __name__ == "__main__":
    unittest.main()
