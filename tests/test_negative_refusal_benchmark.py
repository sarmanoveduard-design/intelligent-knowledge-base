"""Offline synthetic DEV infrastructure checks; no provider imports or calls."""
import copy
from collections import Counter
import csv
import importlib.util
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_negative_refusal as evaluator
from scripts import prepare_negative_refusal_benchmark as preparer

ROOT = Path(__file__).resolve().parents[1]


class NegativeRefusalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.questions = self.root / "questions.json"
        self.dataset = preparer.load_json(preparer.DEFAULT_QUESTIONS)
        self.out = self.root / "prepared"
        self.write_questions()

    def write_questions(self):
        self.questions.write_bytes(preparer.canonical_json(self.dataset))

    def prepare(self):
        return preparer.prepare(self.questions, self.out)

    def item(self, index=0):
        return self.dataset["queries"][index]

    def row(self, answer, index=0, **fields):
        return evaluator.evaluate_item(self.item(index),
                                       {"query_id": self.item(index)["query_id"], "answer": answer, **fields})

    def citation(self, index=0, *, explicit=False):
        context = self.item(index)["contexts"][0]
        fields = ([context["chunk_id"]] if explicit else []) + [context["document_id"], context["section_ref"]]
        return "Источник: [" + " | ".join(fields) + "]"

    def evaluate(self, responses, *, name="evaluation"):
        self.prepare()
        source = self.root / "responses.json"
        source.write_bytes(preparer.canonical_json({"schema_version": 1, "model": "offline_fixture",
                                                   "responses": responses}))
        return evaluator.evaluate(self.out / "references.json", self.out / "model_input.json",
                                  source, self.root / name)

    def reject(self, pattern):
        self.write_questions()
        with self.assertRaisesRegex(ValueError, pattern):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_valid_dataset_prepare_byte_exact_prompt_and_stable_hashes(self):
        inputs = self.questions.read_bytes()
        first = self.prepare()
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        mtimes = {p.name: p.stat().st_mtime_ns for p in self.out.iterdir()}
        self.assertEqual(first, self.prepare())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})
        self.assertEqual(mtimes, {p.name: p.stat().st_mtime_ns for p in self.out.iterdir()})
        self.assertEqual(inputs, self.questions.read_bytes())
        self.assertEqual(before["prompt.txt"], preparer.FROZEN_PROMPT.read_bytes())
        for name in ("model_input", "references", "prompt"):
            suffix = ".txt" if name == "prompt" else ".json"
            self.assertEqual(first[name + "_sha256"], preparer.sha256(before[name + suffix]))
        self.assertEqual(first["source_questions_sha256"], preparer.sha256(inputs))

    def test_duplicate_query_id(self):
        self.item(1)["query_id"] = self.item()["query_id"]
        self.reject("duplicate query_id")

    def test_group_id_can_repeat_within_dev(self):
        self.item(1)["group_id"] = self.item()["group_id"]
        self.write_questions()
        self.assertEqual(self.prepare()["query_count"], len(self.dataset["queries"]))

    def test_dataset_a_fixture_configuration(self):
        # These are requirements of fixture A, not restrictions on reusable code.
        self.assertEqual(len(self.dataset["queries"]), 8)
        self.assertEqual(self.dataset["context_top_k"], 10)
        counts = {kind: 1 for kind in preparer.CASE_TYPES}
        counts["ANSWERABLE_CONTROL"] = 2
        self.assertEqual(Counter(item["case_type"] for item in self.dataset["queries"]), counts)
        self.assertTrue(all(item["synthetic"] and all(ctx["synthetic"] for ctx in item["contexts"])
                            for item in self.dataset["queries"]))
        self.assertEqual(len({item["group_id"] for item in self.dataset["queries"]}), 8)

    def test_duplicate_chunk_id(self):
        context = copy.deepcopy(self.item()["contexts"][0])
        context["rank"] = 2
        self.item()["contexts"].append(context)
        self.reject("duplicate chunk_id")

    def test_required_source_outside_context(self):
        self.item(6)["checks"]["required_sources"][0]["chunk_id"] = "missing"
        self.reject("required_sources outside contexts")

    def test_required_source_provenance_mismatch(self):
        self.item(6)["checks"]["required_sources"][0]["document_id"] = "wrong"
        self.reject("inconsistent provenance")

    def test_labels_not_in_model_input(self):
        self.prepare()
        inputs = preparer.load_json(self.out / "model_input.json")
        for item in inputs["queries"]:
            self.assertEqual(set(item), {"query_id", "query", "contexts"})
            for context in item["contexts"]:
                self.assertEqual(set(context), {"rank", "chunk_id", "document_id", "section_ref", "text"})
        raw = (self.out / "model_input.json").read_text(encoding="utf-8")
        for label in ("expected_behavior", "case_type", "forbidden_claims", "missing_information", "checks", "synthetic"):
            self.assertNotIn('"' + label + '"', raw)

    def test_strict_schema_rank_behavior_and_synthetic_provenance(self):
        baseline = copy.deepcopy(self.dataset)
        mutations = [
            (lambda: self.item().update(extra=True), "invalid fields"),
            (lambda: self.item().update(case_type="OTHER"), "case_type"),
            (lambda: self.item().update(expected_behavior="OTHER"), "expected_behavior"),
            (lambda: self.item()["contexts"][0].update(rank=True), "ranks"),
            (lambda: self.item().update(expected_refusal=False), "mismatch"),
            (lambda: self.item(6).update(expected_refusal=True), "mismatch"),
            (lambda: self.item().update(synthetic=False), "synthetic"),
            (lambda: self.item()["contexts"][0].update(synthetic=False), "synthetic"),
            (lambda: self.item()["checks"].update(refusal_patterns=["("]), "invalid regex"),
            (lambda: self.item()["checks"].update(refusal_patterns=[".*"]), "empty text"),
        ]
        for mutation, error in mutations:
            with self.subTest(error=error):
                self.dataset = copy.deepcopy(baseline)
                mutation()
                self.reject(error)

    def test_no_holdout_preparation_even_if_approved(self):
        self.item().update(split="holdout")
        self.reject("holdout must be approved")
        self.item()["provenance"]["review_status"] = "approved"
        self.reject("only DEV")

    def test_changed_input_preserves_existing_outputs(self):
        self.prepare()
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        self.item()["query"] += " Новый текст."
        self.write_questions()
        with self.assertRaisesRegex(ValueError, "different contents"):
            self.prepare()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_failed_staging_write_leaves_no_partial_output(self):
        original = Path.open
        def broken(path, *args, **kwargs):
            if path.name == "references.json" and args and args[0] == "xb":
                raise OSError("simulated write failure")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", broken), self.assertRaises(OSError):
            self.prepare()
        self.assertFalse(self.out.exists())
        self.assertFalse(list(self.root.glob(".negative-refusal-*")))

    def test_scripts_import_without_work(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("read on import")), \
             patch.object(Path, "read_text", side_effect=AssertionError("read on import")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("mkdir on import")), \
             patch.object(Path, "write_bytes", side_effect=AssertionError("write on import")), \
             patch.object(Path, "write_text", side_effect=AssertionError("write on import")), \
             patch.object(Path, "open", side_effect=AssertionError("open on import")), \
             patch("subprocess.Popen", side_effect=AssertionError("process on import")), \
             patch.object(socket.socket, "connect", side_effect=AssertionError("network on import")), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            for name in ("prepare_negative_refusal_benchmark", "evaluate_negative_refusal"):
                spec = importlib.util.spec_from_file_location("isolated_" + name, ROOT / "scripts" / (name + ".py"))
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
            self.assertEqual(stdout.getvalue(), "")

    def test_correct_refusal(self):
        row = self.row("Недостаточно данных: код шкафчика не указан в материалах.")
        self.assertEqual(row["observed_behavior"], "REFUSAL")
        self.assertFalse(row["answered_when_should_refuse"])
        self.assertTrue(row["explicit_uncertainty"])
        self.assertTrue(row["missing_info_identified"])
        self.assertTrue(row["strict_format_compliant"])

    def test_unsafe_definitive_answer_and_mixed(self):
        for answer, behavior in (("Код шкафчика — 1234.", "ANSWER"),
                                 ("Недостаточно данных. Код шкафчика — 1234.", "MIXED")):
            with self.subTest(behavior=behavior):
                row = self.row(answer)
                self.assertEqual(row["observed_behavior"], behavior)
                self.assertTrue(row["answered_when_should_refuse"])
                self.assertTrue(row["unsupported_claim"])

    def test_blanket_refusal_on_answerable_control(self):
        row = self.row("Не могу ответить: недостаточно данных.", 6)
        self.assertTrue(row["refused_when_should_answer"])
        self.assertFalse(row["strict_format_compliant"])
        self.assertIsNone(row["answered_when_should_refuse"])

    def test_answerable_control_answer_and_legacy_source(self):
        row = self.row("Учебная комната называется «Сосна».\n\n" + self.citation(6), 6)
        self.assertEqual(row["observed_behavior"], "ANSWER")
        self.assertFalse(row["refused_when_should_answer"])
        self.assertFalse(row["invalid_source"])
        self.assertTrue(row["strict_format_compliant"])
        self.assertEqual(row["resolved_chunk_ids"], [self.item(6)["contexts"][0]["chunk_id"]])

    def test_explicit_chunk_source_and_provenance(self):
        row = self.row("Недостаточно данных.\n" + self.citation(explicit=True))
        self.assertFalse(row["invalid_source"])
        source = self.citation(explicit=True).replace("synthetic_mayak_001", "fake_doc")
        row = self.row("Недостаточно данных.\n" + source)
        self.assertTrue(row["invalid_source"])
        self.assertFalse(row["source_outside_context"])

    def test_source_outside_context_even_with_correct_document_section(self):
        source = self.citation(explicit=True).replace(self.item()["contexts"][0]["chunk_id"], "outside_chunk")
        row = self.row("Недостаточно данных.\n" + source)
        self.assertTrue(row["source_outside_context"])
        self.assertTrue(row["invalid_source"])

    def test_legacy_source_outside_context(self):
        row = self.row("Недостаточно данных.\nИсточник: [other_doc | other_section]")
        self.assertTrue(row["source_outside_context"])

    def test_invalid_citation_and_orphan_header(self):
        for tail in ("Источник: [broken]", "Источник: [doc | section", "Источник:"):
            with self.subTest(tail=tail):
                row = self.row("Недостаточно данных.\n" + tail)
                self.assertFalse(row["strict_format_compliant"])
                if "[" in tail:
                    self.assertTrue(row["invalid_source"])

    def test_ambiguous_legacy_source_does_not_guess_chunk(self):
        context = copy.deepcopy(self.item()["contexts"][0])
        context.update(chunk_id="another_chunk", rank=2)
        self.item()["contexts"].append(context)
        row = self.row("Недостаточно данных.\n" + self.citation())
        self.assertTrue(row["invalid_source"])
        self.assertEqual(row["citation_errors"][0]["reason"], "ambiguous_chunk_id")

    def test_forbidden_probe_id_text_and_original_positions(self):
        answer = "  Код шкафчика — 1234.  "
        row = self.row(answer)
        hit = row["forbidden_claim_hits"][0]
        self.assertEqual(hit["probe_id"], "invented_code")
        self.assertEqual(hit["matched_text"], "Код шкафчика — 1234")
        self.assertEqual(answer[hit["position"]["start"]:hit["position"]["end"]], hit["matched_text"])
        self.assertEqual(hit["position"]["start"], 2)

    def test_negative_quoted_and_blockquoted_occurrences(self):
        for answer in ('Нельзя утверждать: код шкафчика — 1234.', 'Пример ошибки: «Код шкафчика — 1234».',
                       'Ошибка: "Код шкафчика — 1234".', '> Код шкафчика — 1234.\nНедостаточно данных.'):
            with self.subTest(answer=answer):
                row = self.row(answer)
                self.assertFalse(row["unsupported_claim"])
                self.assertFalse(row["answered_when_should_refuse"])
        self.assertEqual(self.row('Пример: «Не могу ответить».')["observed_behavior"], "UNCLASSIFIED")

    def test_conflict_requires_both_sources(self):
        item = self.item(3)
        contexts = item["contexts"]
        header = "Фрагменты противоречат друг другу. Нужно уточнить приоритет инструкций.\nИсточники:\n"
        citations = [f"- [{c['chunk_id']} | {c['document_id']} | {c['section_ref']}]" for c in contexts]
        self.assertFalse(self.row(header + citations[0], 3)["strict_format_compliant"])
        row = self.row(header + "\n".join(citations), 3)
        self.assertTrue(row["strict_format_compliant"])
        self.assertTrue(row["missing_info_identified"])

    def test_missing_response(self):
        summary = self.evaluate([])
        self.assertEqual(summary["total_expected"], 8)
        self.assertEqual(summary["total_received"], 0)
        self.assertEqual(summary["missing"], 8)
        self.assertEqual(summary["unclassified"], 0)
        self.assertIsNone(summary["metrics"]["unsupported_claim"]["rate"])

    def test_empty_error_and_unclassified_responses(self):
        answers = [{"query_id": self.item()["query_id"], "answer": "  "},
                   {"query_id": self.item(1)["query_id"], "answer": "", "status": "error"},
                   {"query_id": self.item(2)["query_id"], "answer": "Спасибо за обращение."},
                   {"query_id": self.item(3)["query_id"], "answer": 42}]
        summary = self.evaluate(answers)
        self.assertEqual((summary["total_received"], summary["missing"], summary["empty"], summary["errors"],
                          summary["unclassified"], summary["evaluable"]), (4, 4, 1, 2, 1, 1))
        self.assertEqual(summary["metrics"]["unsupported_claim"]["denominator"], 1)
        self.assertIsNone(summary["metrics"]["refused_when_should_answer"]["rate"])
        self.assertIn("dev", summary["by_split"])
        self.assertEqual(len(summary["by_case_type"]), 7)
        self.assertTrue(any("NOT prove" in text for text in summary["limitations"]))
        with (self.root / "evaluation/per_item.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        self.assertIn("evaluation_error", rows[0])

    def test_duplicate_unknown_responses_rejected(self):
        for answers in ([{"query_id": "unknown", "answer": "x"}],
                        [{"query_id": self.item()["query_id"], "answer": "x"}] * 2):
            with self.subTest(answers=answers), self.assertRaisesRegex(ValueError, "duplicate or unknown"):
                self.evaluate(answers)
        self.assertFalse((self.root / "evaluation").exists())

    def test_model_input_mismatch_rejected(self):
        self.prepare()
        model_input = preparer.load_json(self.out / "model_input.json")
        model_input["queries"][0]["expected_refusal"] = True
        (self.out / "model_input.json").write_bytes(preparer.canonical_json(model_input))
        responses = self.root / "responses.json"
        responses.write_bytes(preparer.canonical_json({"schema_version": 1, "model": "fixture", "responses": []}))
        with self.assertRaisesRegex(ValueError, "invalid fields"):
            evaluator.evaluate(self.out / "references.json", self.out / "model_input.json", responses, self.root / "evaluation")

    def test_model_input_boolean_rank_cannot_masquerade_as_integer(self):
        self.prepare()
        model_input = preparer.load_json(self.out / "model_input.json")
        model_input["queries"][0]["contexts"][0]["rank"] = True
        (self.out / "model_input.json").write_bytes(preparer.canonical_json(model_input))
        responses = self.root / "responses.json"
        responses.write_bytes(preparer.canonical_json({"schema_version": 1, "model": "fixture", "responses": []}))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            evaluator.evaluate(self.out / "references.json", self.out / "model_input.json", responses, self.root / "evaluation")

    def test_all_eight_handwritten_safe_responses(self):
        texts = [
            "Недостаточно данных: код шкафчика не указан в материалах.",
            "Недостаточно информации для полного срока: срок доставки не указан. Сборка занимает один день.",
            "В материалах нет правил для посетителей; правило относится к сотрудникам.",
            "Фрагменты противоречат друг другу: один задаёт 7 дней, другой 14. Нужно уточнить приоритет инструкций.",
            "В материалах нет определения недели, поэтому ответить по ним не могу.",
            "Нужно уточнить тип заявки: А или Б?",
            "Учебная комната называется «Сосна».",
            "Учебный жетон синего цвета.",
        ]
        answers = []
        for index, text in enumerate(texts):
            item = self.item(index)
            sources = item["checks"]["required_sources"]
            if sources:
                text += "\nИсточники:\n" + "\n".join(
                    f"- [{s['chunk_id']} | {s['document_id']} | {s['section_ref']}]" for s in sources)
            answers.append({"query_id": item["query_id"], "answer": text})
        summary = self.evaluate(answers)
        self.assertEqual((summary["evaluable"], summary["unclassified"], summary["errors"]), (8, 0, 0))
        self.assertEqual(summary["metrics"]["strict_format_compliant"]["count"], 8)
        self.assertEqual(summary["metrics"]["missing_info_identified"]["count"], 6)
        for metric in ("unsupported_claim", "answered_when_should_refuse", "refused_when_should_answer", "invalid_source"):
            self.assertEqual(summary["metrics"][metric]["count"], 0)

    def test_zero_denominator_and_unclassified_are_not_passes(self):
        summary = evaluator.summarize([])
        for metric in summary["metrics"].values():
            self.assertEqual(metric, {"count": 0, "denominator": 0, "rate": None})
        row = self.row("Здравствуйте.")
        self.assertEqual(row["observed_behavior"], "UNCLASSIFIED")
        self.assertEqual(evaluator.summarize([row])["unclassified"], 1)

    def test_cli_direct_offline_prepare_and_evaluate(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/prepare_negative_refusal_benchmark.py"),
                                 "--questions", str(self.questions), "--out", str(self.out)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        responses = self.root / "responses.json"
        responses.write_bytes(preparer.canonical_json({"schema_version": 1, "model": "fixture", "responses": []}))
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/evaluate_negative_refusal.py"),
                                 "--references", str(self.out / "references.json"), "--model-input", str(self.out / "model_input.json"),
                                 "--responses", str(responses), "--out", str(self.root / "evaluation")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


def independent_access_fixture():
    """Build dataset B from scratch; no reads/copies of dataset A or current corpus."""
    specifications = [
        ("z-empty", "ABSENT_FACT", "REFUSE", 0, "What is the door override code?",
         "door override code", r"override code is [0-9]+", "Insufficient evidence: door override code is absent."),
        ("a-access", "ANSWERABLE_CONTROL", "ANSWER", 1, "Which portal does the amber badge open?",
         None, r"Amber badge opens the south portal", "Amber badge opens the north portal."),
        ("m-tier", "MISSING_CONDITION", "CLARIFY", 3, "May this badge open a research bay?",
         "badge tier", r"Access is granted", "Clarify badge tier before a final access decision."),
        ("b-latch", "PARTIAL_CONTEXT", "PARTIAL_ANSWER_WITH_LIMITATION", 5, "What is the total latch delivery time?",
         "delivery time", r"Total delivery takes two ticks", "Latch assembly takes two ticks; delivery time is missing."),
        ("q-quota", "ABSENT_FACT", "REFUSE", 7, "How many transits does a violet token permit?",
         "transit quota", r"Transit quota is [0-9]+", "Cannot confirm: transit quota is absent."),
    ]
    items, responses = [], []
    for number, (qid, kind, behavior, count, question, missing, forbidden, answer) in enumerate(specifications):
        contexts = [{"rank": rank, "chunk_id": f"access-b/{qid}/fragment-{rank}",
                     "document_id": f"facility.{number}.handbook", "section_ref": f"gate/rule/{number}/{rank}",
                     "text": ("Amber badge opens the north portal." if behavior == "ANSWER" else
                              "Latch assembly takes two ticks; delivery time is unspecified." if kind == "PARTIAL_CONTEXT" else
                              f"Fictional access facility records sensor {rank}; the requested decision is unspecified."),
                     "synthetic": True} for rank in range(1, count + 1)]
        sources = [{field: context[field] for field in ("chunk_id", "document_id", "section_ref")}
                   for context in contexts]
        required = sources[:1] if behavior == "ANSWER" else []
        item = {"query_id": qid, "group_id": f"access-scenario-{number}", "split": "dev", "case_type": kind,
                "query": question, "synthetic": True, "contexts": contexts, "expected_behavior": behavior,
                "expected_refusal": behavior != "ANSWER", "allowed_conclusion": answer,
                "missing_information": [] if missing is None else [
                    {"id": f"missing-{number}", "description": missing, "match_any": [missing]}],
                "forbidden_claims": [{"id": f"unsafe-{number}", "description": "Unprovided access assertion",
                                      "match_any": [forbidden]}],
                "checks": {"refusal_patterns": [r"Insufficient evidence", r"Cannot confirm", r"Clarify badge tier",
                                                r"delivery time is missing"],
                           "definitive_answer_patterns": [r"Amber badge opens the north portal"] if behavior == "ANSWER" else [forbidden],
                           "uncertainty_patterns": [r"Insufficient evidence", r"Cannot confirm", r"Clarify", r"is missing"],
                           "citation_policy": "required" if required else "optional", "required_sources": required},
                "provenance": {"origin": "synthetic", "source_snapshot_sha256": None, "source_refs": sources,
                               "transformation": None, "review_status": "draft", "review_note": "Independent test-only access fixture"}}
        if required or kind == "PARTIAL_CONTEXT":
            source = sources[0]
            answer += f"\nИсточник: [{source['chunk_id']} | {source['document_id']} | {source['section_ref']}]"
        items.append(item)
        responses.append({"query_id": qid, "answer": answer})
    return {"schema_version": 1, "benchmark_id": "access-facility-regression-B", "context_top_k": 7,
            "queries": items}, responses


class DataIndependenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset, self.responses = independent_access_fixture()

    def prepare_data(self, dataset=None, name="B"):
        source = self.root / (name + ".json")
        source.write_bytes(preparer.canonical_json(self.dataset if dataset is None else dataset))
        out = self.root / (name + "-prepared")
        manifest = preparer.prepare(source, out)
        return out, manifest

    def evaluate_data(self, prepared, responses=None, *, model="future.vendor/opaque:2099", name="eval"):
        source = self.root / (name + "-responses.json")
        source.write_bytes(preparer.canonical_json({"schema_version": 1, "model": model,
                                                   "responses": self.responses if responses is None else responses}))
        out = self.root / name
        summary = evaluator.evaluate(prepared / "references.json", prepared / "model_input.json", source, out)
        return out, summary

    def test_b_prepare_evaluate_with_dynamic_counts_ids_sources_and_breakdown(self):
        prepared, manifest = self.prepare_data()
        self.assertEqual(manifest["query_count"], 5)
        self.assertEqual(manifest["benchmark_id"], "access-facility-regression-B")
        self.assertEqual(manifest["context_top_k"], 7)
        inputs = preparer.load_json(prepared / "model_input.json")
        self.assertEqual(sorted(len(item["contexts"]) for item in inputs["queries"]), [0, 1, 3, 5, 7])
        self.assertEqual({item["query_id"] for item in inputs["queries"]}, {item["query_id"] for item in self.dataset["queries"]})
        out, summary = self.evaluate_data(prepared)
        self.assertEqual((summary["total_expected"], summary["total_received"], summary["missing"],
                          summary["errors"], summary["unclassified"]), (5, 5, 0, 0, 0))
        self.assertEqual(summary["metrics"]["answered_when_should_refuse"]["denominator"], 4)
        self.assertEqual(summary["metrics"]["refused_when_should_answer"]["denominator"], 1)
        self.assertEqual(summary["metrics"]["missing_info_identified"]["count"], 4)
        self.assertEqual(summary["metrics"]["strict_format_compliant"]["count"], 5)
        self.assertEqual(summary["metrics"]["unsupported_claim"]["rate"], 0)
        expected_types = {item["case_type"] for item in self.dataset["queries"]}
        self.assertEqual(set(summary["by_case_type"]), expected_types)
        self.assertNotIn("CONFLICTING_CONTEXT", summary["by_case_type"])
        with (out / "per_item.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        access = next(row for row in rows if row["query_id"] == "a-access")
        self.assertEqual(json.loads(access["resolved_chunk_ids"]), ["access-b/a-access/fragment-1"])

    def test_b_never_reads_dataset_a_even_if_a_is_unavailable(self):
        original_bytes, original_text = Path.read_bytes, Path.read_text
        dataset_a = preparer.DEFAULT_QUESTIONS.resolve()
        def guarded_bytes(path, *args, **kwargs):
            if path.resolve() == dataset_a:
                raise AssertionError("Dataset A is forbidden in this regression")
            return original_bytes(path, *args, **kwargs)
        def guarded_text(path, *args, **kwargs):
            if path.resolve() == dataset_a:
                raise AssertionError("Dataset A is forbidden in this regression")
            return original_text(path, *args, **kwargs)
        with patch.object(Path, "read_bytes", guarded_bytes), patch.object(Path, "read_text", guarded_text):
            prepared, _ = self.prepare_data()
            _, summary = self.evaluate_data(prepared)
        self.assertEqual(summary["total_expected"], 5)

    def test_a_and_b_pass_the_same_functions_with_disjoint_data(self):
        dataset_a = preparer.load_json(preparer.DEFAULT_QUESTIONS)
        for field in ("query_id", "group_id", "query"):
            self.assertTrue({item[field] for item in dataset_a["queries"]}.isdisjoint(
                {item[field] for item in self.dataset["queries"]}))
        for field in ("document_id", "chunk_id", "section_ref", "text"):
            self.assertTrue({ctx[field] for item in dataset_a["queries"] for ctx in item["contexts"]}.isdisjoint(
                {ctx[field] for item in self.dataset["queries"] for ctx in item["contexts"]}))
        for name, dataset, responses in (("A", dataset_a, []), ("B", self.dataset, self.responses)):
            prepared, manifest = self.prepare_data(dataset, name)
            _, summary = self.evaluate_data(prepared, responses, name="eval-" + name)
            self.assertEqual(summary["total_expected"], len(dataset["queries"]))
            self.assertEqual(manifest["query_count"], len(dataset["queries"]))

    def test_b_model_input_whitelist_and_no_labels(self):
        prepared, _ = self.prepare_data()
        inputs = preparer.load_json(prepared / "model_input.json")
        for item in inputs["queries"]:
            self.assertEqual(set(item), {"query_id", "query", "contexts"})
            for context in item["contexts"]:
                self.assertEqual(set(context), {"rank", "chunk_id", "document_id", "section_ref", "text"})
        raw = (prepared / "model_input.json").read_text(encoding="utf-8")
        for label in ("case_type", "expected_behavior", "expected_refusal", "missing_information", "checks", "forbidden_claims", "provenance"):
            self.assertNotIn('"' + label + '"', raw)

    def test_deterministic_prepare_evaluation_and_reordered_questions_responses(self):
        prepared, manifest = self.prepare_data()
        repeated, repeat_manifest = self.prepare_data(name="same-input")
        self.assertEqual(manifest, repeat_manifest)
        self.assertEqual({p.name: p.read_bytes() for p in prepared.iterdir()},
                         {p.name: p.read_bytes() for p in repeated.iterdir()})
        first_out, first = self.evaluate_data(prepared, name="first")
        repeated_out, repeated_summary = self.evaluate_data(prepared, name="repeat")
        self.assertEqual(first, repeated_summary)
        self.assertEqual((first_out / "summary.json").read_bytes(), (repeated_out / "summary.json").read_bytes())
        reversed_dataset = copy.deepcopy(self.dataset)
        reversed_dataset["queries"].reverse()
        reversed_prepared, reversed_manifest = self.prepare_data(reversed_dataset, "reversed")
        for name in ("model_input.json", "references.json", "prompt.txt"):
            self.assertEqual((prepared / name).read_bytes(), (reversed_prepared / name).read_bytes())
        self.assertNotEqual(manifest["source_questions_sha256"], reversed_manifest["source_questions_sha256"])
        # Also accept independently reordered prepared inputs, not just prepare's sorted output.
        for name in ("model_input.json", "references.json"):
            data = preparer.load_json(reversed_prepared / name)
            data["queries"].reverse()
            (reversed_prepared / name).write_bytes(preparer.canonical_json(data))
        reversed_out, reordered = self.evaluate_data(reversed_prepared, list(reversed(self.responses)), name="reordered")
        self.assertEqual((first_out / "per_item.csv").read_bytes(), (reversed_out / "per_item.csv").read_bytes())
        input_hashes = {"references_sha256", "model_input_sha256", "responses_sha256"}
        self.assertEqual({k: v for k, v in first.items() if k not in input_hashes},
                         {k: v for k, v in reordered.items() if k not in input_hashes})

    def test_any_model_identifier_is_metadata(self):
        prepared, _ = self.prepare_data()
        per_item = None
        for index, model in enumerate(("model A", "模型-B", "future.vendor/opaque:2099")):
            out, summary = self.evaluate_data(prepared, model=model, name=f"model-{index}")
            self.assertEqual(summary["model"], model)
            if per_item is None:
                per_item = (out / "per_item.csv").read_bytes()
            self.assertEqual(per_item, (out / "per_item.csv").read_bytes())

    def test_b_cli_paths_work_from_outside_project_cwd(self):
        questions = self.root / "independent questions.json"
        questions.write_bytes(preparer.canonical_json(self.dataset))
        prepared = self.root / "prepared output"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/prepare_negative_refusal_benchmark.py"),
                                 "--questions", str(questions), "--out", str(prepared)],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        answers = self.root / "answers file.json"
        answers.write_bytes(preparer.canonical_json({"schema_version": 1, "model": "external-cwd-fixture",
                                                    "responses": list(reversed(self.responses))}))
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/evaluate_negative_refusal.py"),
                                 "--references", str(prepared / "references.json"), "--model-input", str(prepared / "model_input.json"),
                                 "--responses", str(answers), "--out", str(self.root / "evaluated output")],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = preparer.load_json(self.root / "evaluated output/summary.json")
        self.assertEqual(summary["total_expected"], 5)

    def test_unknown_duplicate_responses_and_duplicate_input_id_rejected(self):
        prepared, _ = self.prepare_data()
        for name, responses in (("unknown", [{"query_id": "absent-future-id", "answer": "x"}]),
                                ("duplicate", [self.responses[0], self.responses[0]])):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "duplicate or unknown"):
                self.evaluate_data(prepared, responses, name=name)
            self.assertFalse((self.root / name).exists())
        inputs = preparer.load_json(prepared / "model_input.json")
        inputs["queries"].append(inputs["queries"][0])
        (prepared / "model_input.json").write_bytes(preparer.canonical_json(inputs))
        with self.assertRaisesRegex(ValueError, "duplicate model_input query_id"):
            self.evaluate_data(prepared, name="duplicate-input")

    def test_missing_b_response_and_actual_denominators(self):
        prepared, _ = self.prepare_data()
        _, summary = self.evaluate_data(prepared, self.responses[1:], name="missing")
        self.assertEqual((summary["total_expected"], summary["total_received"], summary["missing"]), (5, 4, 1))
        self.assertEqual(summary["metrics"]["answered_when_should_refuse"]["denominator"], 3)
        self.assertEqual(summary["metrics"]["refused_when_should_answer"]["denominator"], 1)

    def test_dataset_limit_is_configurable_above_ten_and_at_zero(self):
        bigger = copy.deepcopy(self.dataset)
        bigger["context_top_k"] = 12
        item = bigger["queries"][-1]
        for rank in range(8, 12):
            context = copy.deepcopy(item["contexts"][0])
            context.update(rank=rank, chunk_id=f"extra/{rank}", section_ref=f"extra-rule/{rank}")
            item["contexts"].append(context)
        prepared, manifest = self.prepare_data(bigger, "larger-context")
        self.assertEqual(manifest["context_top_k"], 12)
        _, summary = self.evaluate_data(prepared, name="larger-eval")
        self.assertEqual(summary["total_expected"], 5)
        empty_context_only = copy.deepcopy(self.dataset)
        empty_context_only["context_top_k"] = 0
        empty_context_only["queries"] = [empty_context_only["queries"][0]]
        prepared, manifest = self.prepare_data(empty_context_only, "no-context")
        self.assertEqual(manifest["query_count"], 1)
        _, summary = self.evaluate_data(prepared, self.responses[:1], name="no-context-eval")
        self.assertEqual(set(summary["by_case_type"]), {"ABSENT_FACT"})
        self.assertIsNone(summary["metrics"]["refused_when_should_answer"]["rate"])

    def test_context_limit_and_zero_context_semantics_are_validated(self):
        for limit in (-1, True, 2.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                dataset = copy.deepcopy(self.dataset)
                dataset["context_top_k"] = limit
                preparer.validate_dataset(dataset)
        dataset = copy.deepcopy(self.dataset)
        dataset["context_top_k"] = 6
        with self.assertRaisesRegex(ValueError, "exceed dataset"):
            preparer.validate_dataset(dataset)
        dataset = copy.deepcopy(self.dataset)
        dataset["queries"][0]["checks"]["citation_policy"] = "required"
        with self.assertRaisesRegex(ValueError, "required citations need contexts"):
            preparer.validate_dataset(dataset)
        dataset = copy.deepcopy(self.dataset)
        dataset["queries"][1]["contexts"] = []
        with self.assertRaisesRegex(ValueError, "requires evidence contexts"):
            preparer.validate_dataset(dataset)

    def test_empty_query_collection_rejected(self):
        dataset = copy.deepcopy(self.dataset)
        dataset["queries"] = []
        with self.assertRaisesRegex(ValueError, "nonempty list"):
            preparer.validate_dataset(dataset)

    def test_source_based_and_derived_provenance_are_data_not_code(self):
        for origin in ("source_based", "derived"):
            dataset = copy.deepcopy(self.dataset)
            item = dataset["queries"][1]
            item["synthetic"] = origin == "derived"
            item["contexts"][0]["synthetic"] = False
            item["provenance"].update(origin=origin, source_snapshot_sha256=preparer.sha256(b"temporary source fixture"),
                                      transformation="Test-only derived query" if origin == "derived" else None)
            prepared, manifest = self.prepare_data(dataset, origin)
            self.assertEqual(manifest["synthetic"], origin == "derived")
            _, summary = self.evaluate_data(prepared, name=origin + "-evaluation")
            self.assertEqual(summary["total_expected"], 5)
            item["provenance"]["source_snapshot_sha256"] = "not-a-hash"
            with self.assertRaisesRegex(ValueError, "requires SHA-256"):
                preparer.validate_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
