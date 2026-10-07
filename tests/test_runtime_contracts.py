"""Offline contract tests; all adapters below are synthetic test doubles."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import subprocess
import sys
import unittest
from typing import Sequence, get_type_hints

from knowledge_base.runtime import config, models, protocols
from knowledge_base.runtime.config import RuntimeConfig, ScoreThreshold
from knowledge_base.runtime.models import (
    AccessContext, AnswerState, AskRequest, AskResult, ChunkIdentity,
    CitationReference, CitationValidationResult, ContextEntry, ContextManifest,
    DocumentIdentity, DocumentVersionIdentity, DocumentVersionStatus,
    EscalationStatus, EvidenceChunk, ExpertEscalation, GenerationDraft,
    GenerationLimits, GenerationRequest, IndexIdentity, IndexedEvidence,
    IndexManifest, ProcessingSnapshotIdentity, ProcessingState, RetrievalFilters,
    RuntimeDocument, RuntimeDocumentVersion, RuntimeEvidence, SemanticSupport,
    SourceCoordinates, StructuralValidity, SufficiencyDecision, SufficiencyStatus,
    compose_retrieval_filters,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/knowledge_base/runtime"


def example_corpus(label: str, count: int):
    organization = "org/" + label
    document = DocumentIdentity("document/" + label, organization)
    version = DocumentVersionIdentity(document.document_id, "revision/" + label)
    if label == "company":
        question = "Where does the fictional company accept supply requests?"
        texts = ["Supply requests go to the Cedar company desk."]
    else:
        question = "How does the fictional laboratory label sample containers?"
        texts = ["Glass containers carry a blue label.", "Polymer containers carry a silver label.",
                 "The laboratory records the label before storage."]
    evidence = tuple(RuntimeEvidence(
        EvidenceChunk(
            ChunkIdentity(document.document_id, version.version_id, "snapshot/" + label, "chunk/" + label + "/" + str(i)),
            document, version, texts[i], SourceCoordinates(section_ref="part/" + str(i)),
            local_chunk_index=i, metadata={"label": label},
        ), retrieval_score=0.4 + i / 10, retrieval_score_type="cosine", retrieval_scorer_identity="synthetic-provider/synthetic-space",
    ) for i in range(count))
    request = AskRequest("request/" + label, question, AccessContext(organization, "reader/" + label))
    return request, evidence


def index_identity():
    return IndexIdentity("synthetic-provider", "synthetic-space", 2, "parser/v1", "chunking/v1", "plain/v1")


class FakeDocuments:
    def __init__(self):
        self.documents = {}
        self.versions = {}

    def save_document(self, document: RuntimeDocument) -> None:
        self.documents[document.identity.document_id] = document

    def get_document(self, identity: DocumentIdentity) -> RuntimeDocument | None:
        document = self.documents.get(identity.document_id)
        return document if document and document.identity == identity else None

    def save_version(self, version: RuntimeDocumentVersion) -> None:
        if version.identity.document_id not in self.documents:
            raise ValueError("unknown document")
        previous = self.versions.get(version.identity)
        if version.status == DocumentVersionStatus.ACTIVE and (previous is None or previous.status != version.status):
            raise ValueError("publication requires activate")
        self.versions[version.identity] = version

    def get_version(self, identity: DocumentVersionIdentity) -> RuntimeDocumentVersion | None:
        return self.versions.get(identity)

    def list_visible_versions(
        self, access: AccessContext, *, filters: RetrievalFilters,
    ) -> tuple[RuntimeDocumentVersion, ...]:
        filters = compose_retrieval_filters(access, filters)
        result = []
        for version in self.versions.values():
            document = self.documents[version.identity.document_id]
            if (document.identity.organization_id == access.organization_id
                    and document.required_scopes <= access.scopes
                    and version.status == DocumentVersionStatus.ACTIVE
                    and version.approved and version.processing_state == ProcessingState.INDEXED
                    and (filters.allowed_document_ids is None or version.identity.document_id in filters.allowed_document_ids)
                    and (filters.allowed_version_ids is None or version.identity.version_id in filters.allowed_version_ids)):
                result.append(version)
        return tuple(result)

    def activate_version(
        self, identity: DocumentVersionIdentity, *, expected_current_version_id: str | None,
    ) -> None:
        target = self.versions[identity]
        current = next((v for v in self.versions.values()
                        if v.identity.document_id == identity.document_id
                        and v.status == DocumentVersionStatus.ACTIVE), None)
        if (current.identity.version_id if current else None) != expected_current_version_id:
            raise ValueError("concurrent publication")
        if (target.status not in (DocumentVersionStatus.DRAFT, DocumentVersionStatus.ACTIVE)
                or target.processing_state not in (ProcessingState.CHUNKED, ProcessingState.INDEXED)):
            raise ValueError("version cannot be published")
        if current is not None:
            self.versions[current.identity] = replace(current, status=DocumentVersionStatus.SUPERSEDED)
        self.versions[identity] = replace(target, status=DocumentVersionStatus.ACTIVE, approved=True)

    def archive_version(self, identity: DocumentVersionIdentity) -> None:
        self.versions[identity] = replace(self.versions[identity], status=DocumentVersionStatus.ARCHIVED)


class FakeChunks:
    def __init__(self):
        self.snapshots = {}

    def save_snapshot(self, snapshot: ProcessingSnapshotIdentity, chunks: Sequence[EvidenceChunk]) -> None:
        chunks = tuple(chunks)
        if any(chunk.identity.snapshot != snapshot for chunk in chunks):
            raise ValueError("snapshot mismatch")
        if len({chunk.identity.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("duplicate chunk")
        if snapshot in self.snapshots and self.snapshots[snapshot] != chunks:
            raise ValueError("immutable snapshot")
        self.snapshots[snapshot] = chunks

    def list_snapshot(self, snapshot: ProcessingSnapshotIdentity) -> tuple[EvidenceChunk, ...]:
        return self.snapshots.get(snapshot, ())

    def get_chunk(self, identity: ChunkIdentity) -> EvidenceChunk | None:
        return next((chunk for chunk in self.list_snapshot(identity.snapshot) if chunk.identity == identity), None)

    def delete_snapshot(self, snapshot: ProcessingSnapshotIdentity) -> None:
        self.snapshots.pop(snapshot, None)


class FakeVectorIndex:
    """Records search arguments only; it does not execute retrieval."""
    def __init__(self):
        self._identity = index_identity()
        self.manifests = {}
        self.last_search = None

    @property
    def identity(self) -> IndexIdentity:
        return self._identity

    def upsert(self, manifest: IndexManifest, chunks: Sequence[IndexedEvidence]) -> None:
        if not self.identity.is_compatible_with(manifest.identity):
            raise ValueError("incompatible index identity")
        if len(chunks) != manifest.chunk_count:
            raise ValueError("manifest count mismatch")
        if len({item.chunk.identity.chunk_id for item in chunks}) != len(chunks):
            raise ValueError("duplicate chunk")
        if any(item.chunk.identity.snapshot != manifest.snapshot
               or len(item.vector) != self.identity.vector_dimension for item in chunks):
            raise ValueError("invalid indexed chunk")
        self.manifests[manifest.snapshot] = manifest

    def search(
        self, query_vector: Sequence[float], *, access: AccessContext,
        filters: RetrievalFilters, top_k: int, expected_identity: IndexIdentity,
    ) -> tuple[RuntimeEvidence, ...]:
        if not self.identity.is_compatible_with(expected_identity):
            raise ValueError("incompatible index identity")
        if len(query_vector) != self.identity.vector_dimension or top_k < 1:
            raise ValueError("invalid search shape")
        self.last_search = compose_retrieval_filters(access, filters)
        return ()

    def get_manifest(self, snapshot: ProcessingSnapshotIdentity) -> IndexManifest | None:
        return self.manifests.get(snapshot)

    def delete_snapshot(self, snapshot: ProcessingSnapshotIdentity) -> None:
        self.manifests.pop(snapshot, None)

    def deactivate_version(self, identity: DocumentVersionIdentity) -> None:
        for snapshot, manifest in tuple(self.manifests.items()):
            if snapshot.version == identity:
                self.manifests[snapshot] = replace(manifest, ready=False)


class FakeGeneration:
    def generate(self, request: GenerationRequest) -> GenerationDraft:
        return GenerationDraft("Synthetic draft", provider_metadata={"provider": "test-double"})


class FakeEscalations:
    def __init__(self):
        self.records = {}

    def save(self, escalation: ExpertEscalation) -> None:
        self.records[escalation.escalation_id] = escalation

    def get(self, escalation_id: str) -> ExpertEscalation | None:
        return self.records.get(escalation_id)


class RuntimeContractsTests(unittest.TestCase):
    def test_import_all_runtime_modules_without_io_network_processes_or_providers(self):
        code = r'''
import builtins, dataclasses, datetime, enum, importlib, importlib.abc, math, pathlib, socket, sqlite3, subprocess, sys, types, typing
from unittest.mock import patch
import knowledge_base
class RejectExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("openai", "ollama", "torch", "transformers", "requests", "httpx",
                   "benchmarks", "scripts", "knowledge_base.benchmark", "knowledge_base.team",
                   "knowledge_base.candidate_retrieval", "knowledge_base.bge_reranker",
                   "knowledge_base.retriever", "knowledge_base.vector_store")
        if fullname.startswith(blocked):
            raise AssertionError("external import: " + fullname)
sys.meta_path.insert(0, RejectExternal())
def forbidden(*args, **kwargs):
    raise AssertionError("import performed I/O or execution")
with patch.object(builtins, "open", forbidden), patch.object(pathlib.Path, "open", forbidden), \
     patch.object(pathlib.Path, "read_text", forbidden), patch.object(pathlib.Path, "read_bytes", forbidden), \
     patch.object(pathlib.Path, "write_text", forbidden), patch.object(pathlib.Path, "write_bytes", forbidden), \
     patch.object(pathlib.Path, "mkdir", forbidden), patch.object(socket, "socket", forbidden), \
     patch.object(subprocess, "Popen", forbidden), patch.object(sqlite3, "connect", forbidden):
    for name in ("knowledge_base.runtime", "knowledge_base.runtime.models", "knowledge_base.runtime.config", "knowledge_base.runtime.protocols",
                 "knowledge_base.runtime.storage", "knowledge_base.runtime.ingestion",
                 "knowledge_base.runtime.vector_index", "knowledge_base.runtime.indexing", "knowledge_base.runtime.retrieval"):
        importlib.import_module(name)
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_import_graph_uses_only_explicit_production_dependencies(self):
        for path in RUNTIME.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], sys.stdlib_module_names, path.name)
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    if node.module.startswith("knowledge_base."):
                        self.assertIn(node.module, {"knowledge_base.document_intake", "knowledge_base.document_registry",
                            "knowledge_base.document_versions", "knowledge_base.source_blocks", "knowledge_base.chunker",
                            "knowledge_base.embeddings"}, path.name)
                    else:
                        self.assertIn(node.module.split(".")[0], sys.stdlib_module_names, path.name)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 1)
                    self.assertIn(node.module, {"models", "config", "protocols", "storage", "ingestion", "vector_index", "indexing", "retrieval"})

    def test_runtime_contains_no_fixed_provider_or_domain_literals(self):
        prohibited = {"MAIN119", "TEAM HOLDOUT", "MOST", "legal", "medical", "company", "laboratory",
                      "bge-m3", "BAAI/bge-reranker-v2-m3", "Qwen", "openai", "ollama"}
        for path in RUNTIME.glob("*.py"):
            literals = {node.value for node in ast.walk(ast.parse(path.read_text(encoding="utf8")))
                        if isinstance(node, ast.Constant) and isinstance(node.value, str)}
            self.assertFalse(literals & prohibited, path.name)

    def test_arbitrary_question_and_runtime_request_id(self):
        request = AskRequest("arbitrary/request", "¿Cómo funciona el dispositivo Ω?", AccessContext("tenant Ω", "principal Z"))
        self.assertIn("Ω", request.question)
        self.assertNotIn("query_id", {field.name for field in fields(request)})
        with self.assertRaises(TypeError):
            AskRequest("x", "q", request.access, query_id="label")

    def test_opaque_identities_and_local_zero_do_not_collide(self):
        _, a = example_corpus("company", 1)
        _, b = example_corpus("laboratory", 3)
        self.assertEqual(a[0].chunk.local_chunk_index, b[0].chunk.local_chunk_index)
        self.assertNotEqual(a[0].chunk.identity, b[0].chunk.identity)
        self.assertNotEqual(a[0].chunk.identity.chunk_id, b[0].chunk.identity.chunk_id)
        self.assertEqual(len({a[0].chunk.identity, b[0].chunk.identity}), 2)
        identity = ChunkIdentity("документ Ω", "version.next", "processing#other", "fragment/自由")
        self.assertEqual(identity.snapshot.version.document_id, "документ Ω")

    def test_version_and_snapshot_are_part_of_identity(self):
        identity = ChunkIdentity("doc", "v1", "snapshot1", "chunk")
        self.assertNotEqual(identity, replace(identity, version_id="v2"))
        self.assertNotEqual(identity, replace(identity, processing_snapshot_id="snapshot2"))

    def test_identity_mismatch_is_rejected(self):
        _, items = example_corpus("company", 1)
        with self.assertRaises(ValueError):
            replace(items[0].chunk, version=DocumentVersionIdentity("other", "revision/company"))
        with self.assertRaises(ValueError):
            replace(items[0].chunk, identity=replace(items[0].chunk.identity, version_id="other"))

    def test_blank_ids_and_questions_are_rejected(self):
        for factory in (lambda: DocumentIdentity(" ", "org"), lambda: DocumentVersionIdentity("doc", ""),
                        lambda: ChunkIdentity("d", "v", "", "c"),
                        lambda: AskRequest("r", " ", AccessContext("o", "p"))):
            with self.assertRaises(ValueError):
                factory()

    def test_metadata_and_sequences_are_defensively_immutable(self):
        _, items = example_corpus("company", 1)
        source = {"nested": {"values": [1, 2]}}
        chunk = replace(items[0].chunk, metadata=source)
        source["nested"]["values"].append(3)
        self.assertEqual(chunk.metadata["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            chunk.metadata["nested"]["new"] = "value"
        with self.assertRaises(FrozenInstanceError):
            chunk.text = "changed"
        scopes = {"read"}
        access = AccessContext("o", "p", scopes)
        scopes.add("admin")
        self.assertEqual(access.scopes, frozenset({"read"}))

    def test_filter_composition_intersects_document_and_version_grants(self):
        access = AccessContext("tenant", "reader", {"read", "restricted"})
        authorized = RetrievalFilters({"a", "b"}, {"v1", "v2"}, "tenant", {"read"})
        requested = RetrievalFilters({"b", "foreign"}, {"v2", "other"}, required_scopes={"restricted"})
        result = compose_retrieval_filters(access, authorized, requested)
        self.assertEqual(result.allowed_document_ids, frozenset({"b"}))
        self.assertEqual(result.allowed_version_ids, frozenset({"v2"}))
        self.assertEqual(result.required_scopes, frozenset({"read", "restricted"}))
        self.assertEqual(access.scopes, frozenset({"read", "restricted"}))

    def test_unrestricted_request_cannot_expand_authorized_filters(self):
        access = AccessContext("tenant", "reader")
        authorized = RetrievalFilters({"a"}, {"v1"})
        result = compose_retrieval_filters(access, authorized, RetrievalFilters())
        self.assertEqual(result.allowed_document_ids, frozenset({"a"}))
        self.assertEqual(result.allowed_version_ids, frozenset({"v1"}))

    def test_empty_filters_mean_no_matches_and_none_means_no_extra_restriction(self):
        self.assertEqual(RetrievalFilters(set()).narrow(RetrievalFilters()).allowed_document_ids, frozenset())
        self.assertIsNone(RetrievalFilters().narrow(RetrievalFilters()).allowed_document_ids)
        self.assertEqual(RetrievalFilters({"a"}).narrow(RetrievalFilters({"b"})).allowed_document_ids, frozenset())

    def test_foreign_organization_and_unavailable_scopes_are_rejected(self):
        access = AccessContext("tenant", "reader", {"read"})
        for filters in (RetrievalFilters(organization_id="foreign"), RetrievalFilters(required_scopes={"admin"})):
            with self.assertRaises(ValueError):
                compose_retrieval_filters(access, RetrievalFilters(), filters)
        with self.assertRaises(ValueError):
            compose_retrieval_filters(access, RetrievalFilters(organization_id="foreign"))

    def test_config_accepts_final_top_k_one_three_seven(self):
        for k in (1, 3, 7):
            config = RuntimeConfig(candidate_top_k=k, final_top_k=k)
            self.assertEqual(config.final_top_k, k)
            self.assertFalse(config.partial_answers_enabled)

    def test_config_rejects_inconsistent_limits(self):
        for kwargs in ({"candidate_top_k": 2, "final_top_k": 3}, {"context_max_chars": 0},
                       {"minimum_evidence_count": 0}, {"final_top_k": 0},
                       {"final_top_k": 1, "minimum_evidence_count": 2},
                       {"candidate_top_k": True}, {"partial_answers_enabled": 1}):
            with self.assertRaises(ValueError):
                RuntimeConfig(**kwargs)

    def test_absent_threshold_is_unconfigured_not_a_sufficiency_decision(self):
        config = RuntimeConfig()
        self.assertIsNone(config.dense_threshold)
        self.assertIsNone(config.reranker_threshold)
        self.assertFalse(hasattr(config, "is_sufficient"))
        with self.assertRaises(ValueError):
            RuntimeConfig(dense_threshold=0.5)

    def test_thresholds_require_finite_value_type_and_scorer_identity(self):
        threshold = ScoreThreshold(-2.5, "raw_logit", "arbitrary-scorer/revision-9")
        self.assertEqual(RuntimeConfig(reranker_threshold=threshold).reranker_threshold, threshold)
        for args in ((float("nan"), "cosine", "s"), (float("inf"), "cosine", "s"),
                     (True, "cosine", "s"), (0.2, "", "s"), (0.2, "cosine", "")):
            with self.assertRaises(ValueError):
                ScoreThreshold(*args)

    def test_dense_and_reranker_scores_remain_separate(self):
        _, evidence = example_corpus("company", 1)
        scored = replace(evidence[0], reranker_score=-3.0, reranker_score_type="raw_logit",
                         reranker_scorer_identity="synthetic-ranker/revision")
        self.assertEqual(scored.retrieval_score, evidence[0].retrieval_score)
        self.assertEqual(scored.retrieval_score_type, "cosine")
        self.assertEqual(scored.reranker_score, -3.0)
        self.assertEqual(scored.reranker_score_type, "raw_logit")
        self.assertEqual(scored.reranker_scorer_identity, "synthetic-ranker/revision")
        self.assertEqual(scored.retrieval_scorer_identity, evidence[0].retrieval_scorer_identity)

    def test_score_types_and_finite_scores_are_required(self):
        _, evidence = example_corpus("company", 1)
        for kwargs in ({"retrieval_score_type": ""}, {"retrieval_scorer_identity": ""},
                       {"retrieval_score": float("nan")},
                       {"reranker_score": 1.0}, {"reranker_score_type": "logit"},
                       {"reranker_score": float("inf"), "reranker_score_type": "logit", "reranker_scorer_identity": "scorer"},
                       {"reranker_score": 1.0, "reranker_score_type": "logit", "reranker_scorer_identity": ""}):
            with self.assertRaises(ValueError):
                replace(evidence[0], **kwargs)

    def test_context_manifest_contains_only_selected_ordered_exact_evidence(self):
        _, pool = example_corpus("laboratory", 3)
        manifest = ContextManifest([ContextEntry("source-z", pool[2]), ContextEntry("source-a", pool[0])])
        self.assertEqual(manifest.chunk_ids, (pool[2].chunk.identity.chunk_id, pool[0].chunk.identity.chunk_id))
        self.assertNotIn(pool[1].chunk.identity.chunk_id, manifest.chunk_ids)
        self.assertEqual(manifest.entries[0].evidence.chunk.text, pool[2].chunk.text)
        self.assertEqual(manifest.entries[0].evidence.chunk.coordinates, pool[2].chunk.coordinates)

    def test_manifest_distinguishes_versions_and_rejects_duplicate_handles_or_chunk_ids(self):
        _, pool = example_corpus("company", 1)
        old = pool[0]
        new = RuntimeEvidence(replace(old.chunk,
            identity=replace(old.chunk.identity, version_id="v2", chunk_id="chunk/v2"),
            version=replace(old.chunk.version, version_id="v2")), 0.9, "cosine", "synthetic-provider/synthetic-space")
        manifest = ContextManifest((ContextEntry("old", old), ContextEntry("new", new)))
        self.assertNotEqual(manifest.entries[0].evidence.chunk.version, manifest.entries[1].evidence.chunk.version)
        for entries in ((ContextEntry("same", old), ContextEntry("same", new)),
                        (ContextEntry("a", old), ContextEntry("b", old))):
            with self.assertRaises(ValueError):
                ContextManifest(entries)

    def test_citation_represents_exact_source_handle_and_chunk_identity(self):
        _, evidence = example_corpus("company", 1)
        reference = CitationReference("source-1", evidence[0].chunk.identity)
        self.assertEqual(reference.chunk_identity.version_id, "revision/company")
        draft = GenerationDraft("A draft.", [reference])
        self.assertEqual(draft.citations, (reference,))

    def test_structural_and_semantic_validation_are_independent(self):
        result = CitationValidationResult(StructuralValidity.PASS)
        self.assertEqual(result.semantic_support, SemanticSupport.NOT_CHECKED)
        self.assertEqual(replace(result, semantic_support=SemanticSupport.UNSUPPORTED).structural_validity, StructuralValidity.PASS)
        with self.assertRaises(ValueError):
            replace(result, errors=("invalid source",))

    def test_generation_request_has_no_labels_or_arbitrary_metadata_channel(self):
        forbidden = {"expected_behavior", "expected_answer", "answer_hint", "forbidden_claims", "gold", "probes", "query_id"}
        self.assertFalse(forbidden & {field.name for field in fields(GenerationRequest)})
        self.assertNotIn("metadata", {field.name for field in fields(GenerationRequest)})
        request = GenerationRequest("Any question", ContextManifest(), "prompt/custom", "schema/custom")
        for name in forbidden:
            with self.assertRaises(TypeError):
                GenerationRequest(request.question, request.context_manifest, request.prompt_version,
                                  request.output_schema_version, **{name: "label"})

    def test_generation_limits_validate_positive_values(self):
        for kwargs in ({"max_output_tokens": 0}, {"timeout_seconds": -1}, {"timeout_seconds": float("nan")}):
            with self.assertRaises(ValueError):
                GenerationLimits(**kwargs)

    def test_all_answer_and_lifecycle_states_are_present(self):
        self.assertEqual(set(AnswerState.__members__), {"ANSWER", "PARTIAL_ANSWER", "CLARIFY", "REFUSE_INSUFFICIENT_CONTEXT", "CONFLICT", "ESCALATE_EXPERT", "ERROR"})
        self.assertEqual(set(DocumentVersionStatus.__members__), {"DRAFT", "ACTIVE", "SUPERSEDED", "ARCHIVED"})
        self.assertEqual(set(ProcessingState.__members__), {"REGISTERED", "PARSED", "CHUNKED", "INDEXED", "FAILED"})
        self.assertEqual(set(SufficiencyStatus.__members__), {"SUFFICIENT", "INSUFFICIENT", "CLARIFICATION_REQUIRED", "CONFLICT", "ERROR"})

    def test_sufficiency_decision_needs_only_runtime_evidence_and_reasons(self):
        decision = SufficiencyDecision(SufficiencyStatus.INSUFFICIENT, "empty_context", "No evidence available", 0)
        self.assertEqual(decision.evidence_count, 0)
        self.assertFalse(hasattr(decision, "expected_behavior"))
        with self.assertRaises(ValueError):
            replace(decision, evidence_count=-1)

    def test_ask_results_support_refusal_answer_and_escalation(self):
        refusal = AskResult("r", AnswerState.REFUSE_INSUFFICIENT_CONTEXT, "empty_context")
        self.assertIsNone(refusal.answer)
        answer = replace(refusal, state=AnswerState.ANSWER, answer="A supported draft")
        self.assertIsNotNone(answer.answer)
        with self.assertRaises(ValueError):
            replace(refusal, state=AnswerState.ANSWER)
        with self.assertRaises(ValueError):
            replace(refusal, state=AnswerState.ESCALATE_EXPERT)
        self.assertEqual(replace(refusal, state=AnswerState.ESCALATE_EXPERT, escalation_id="e").escalation_id, "e")

    def test_expert_escalation_is_only_a_record_with_aware_timestamp(self):
        request, evidence = example_corpus("company", 1)
        escalation = ExpertEscalation("e", request.request_id, datetime.now(timezone.utc), request.question,
                                     request.access, ContextManifest(), "missing evidence")
        self.assertEqual(escalation.status, EscalationStatus.PENDING)
        self.assertIsNone(escalation.draft)
        self.assertEqual(set(EscalationStatus.__members__), {"PENDING", "RESOLVED", "CANCELLED"})
        with self.assertRaises(ValueError):
            replace(escalation, created_at=datetime(2026, 1, 1))
        repository = FakeEscalations()
        repository.save(escalation)
        self.assertEqual(repository.get("e"), escalation)

    def test_same_contracts_work_for_two_disjoint_synthetic_contexts(self):
        pairs = [example_corpus("company", 1), example_corpus("laboratory", 3)]
        all_ids = []
        for request, evidence in pairs:
            manifest = ContextManifest(tuple(ContextEntry("source/" + str(i), item) for i, item in enumerate(evidence)))
            generation = GenerationRequest(request.question, manifest, "generic/v1", "draft/v1")
            draft = FakeGeneration().generate(generation)
            self.assertIsInstance(draft, GenerationDraft)
            self.assertEqual(len(manifest.entries), len(evidence))
            all_ids.append(set(manifest.chunk_ids))
        self.assertFalse(all_ids[0] & all_ids[1])
        self.assertNotEqual(pairs[0][0].question, pairs[1][0].question)
        self.assertNotEqual(pairs[0][1][0].chunk.version, pairs[1][1][0].chunk.version)
        self.assertNotEqual(pairs[0][1][0].chunk.text, pairs[1][1][0].chunk.text)

    def test_index_identity_distinguishes_embedding_and_processing_spaces(self):
        identity = index_identity()
        self.assertTrue(identity.is_compatible_with(replace(identity)))
        for kwargs in ({"embedding_provider": "another-provider"}, {"embedding_model": "another-model"},
                       {"vector_dimension": 3}, {"preprocessing_version": "parser/v2"},
                       {"chunking_version": "chunking/v2"}, {"representation_version": "representation/v2"},
                       {"embedding_revision": "revision/other"}):
            self.assertFalse(identity.is_compatible_with(replace(identity, **kwargs)))

    def test_manifest_readiness_and_snapshot_identity_are_explicit(self):
        _, evidence = example_corpus("company", 1)
        manifest = IndexManifest(index_identity(), evidence[0].chunk.identity.snapshot, 1)
        self.assertFalse(manifest.ready)
        self.assertNotEqual(manifest.snapshot, replace(manifest.snapshot, processing_snapshot_id="another"))
        with self.assertRaises(ValueError):
            replace(manifest, chunk_count=-1)

    def test_indexed_vectors_are_immutable_finite_and_nonempty(self):
        _, evidence = example_corpus("company", 1)
        vector = [1.0, 0.0]
        indexed = IndexedEvidence(evidence[0].chunk, vector)
        vector.append(3)
        self.assertEqual(indexed.vector, (1.0, 0.0))
        for value in ([], [float("nan")], [float("inf")], [True]):
            with self.assertRaises(ValueError):
                IndexedEvidence(evidence[0].chunk, value)

    def test_repository_fakes_cover_publication_superseding_archival_and_snapshots(self):
        request, evidence = example_corpus("company", 1)
        chunk = evidence[0].chunk
        documents = FakeDocuments()
        documents.save_document(RuntimeDocument(chunk.document, "Synthetic document"))
        self.assertEqual(documents.get_document(chunk.document).identity, chunk.document)
        version = RuntimeDocumentVersion(chunk.version, "first")
        documents.save_version(version)
        filters = RetrievalFilters()
        self.assertEqual(documents.list_visible_versions(request.access, filters=filters), ())
        version = replace(version, approved=True, processing_state=ProcessingState.INDEXED)
        documents.save_version(version)
        documents.activate_version(version.identity, expected_current_version_id=None)
        next_version = replace(version, identity=replace(version.identity, version_id="next"))
        documents.save_version(next_version)
        documents.activate_version(next_version.identity, expected_current_version_id=version.identity.version_id)
        self.assertEqual(documents.get_version(version.identity).status, DocumentVersionStatus.SUPERSEDED)
        self.assertEqual(documents.list_visible_versions(request.access, filters=filters)[0].identity, next_version.identity)
        documents.archive_version(next_version.identity)
        self.assertEqual(documents.list_visible_versions(request.access, filters=filters), ())
        chunks = FakeChunks()
        chunks.save_snapshot(chunk.identity.snapshot, [chunk])
        chunks.save_snapshot(chunk.identity.snapshot, [chunk])
        self.assertEqual(chunks.get_chunk(chunk.identity), chunk)
        chunks.delete_snapshot(chunk.identity.snapshot)
        self.assertEqual(chunks.list_snapshot(chunk.identity.snapshot), ())

    def test_vector_index_contract_accepts_filters_and_manifest_operations(self):
        request, evidence = example_corpus("company", 1)
        chunk = evidence[0].chunk
        index = FakeVectorIndex()
        manifest = IndexManifest(index.identity, chunk.identity.snapshot, 1, ready=True)
        index.upsert(manifest, [IndexedEvidence(chunk, (1.0, 0.0))])
        filters = RetrievalFilters({chunk.document.document_id})
        self.assertEqual(index.search([1.0, 0.0], access=request.access, filters=filters,
                                      top_k=1, expected_identity=index.identity), ())
        self.assertEqual(index.last_search.allowed_document_ids, filters.allowed_document_ids)
        self.assertEqual(index.get_manifest(manifest.snapshot), manifest)
        index.deactivate_version(chunk.version)
        self.assertFalse(index.get_manifest(manifest.snapshot).ready)
        index.delete_snapshot(manifest.snapshot)
        self.assertIsNone(index.get_manifest(manifest.snapshot))

    def test_protocol_fake_shapes_and_resolvable_annotations(self):
        # Runtime Protocol checks only shape; adapter semantics need later tests.
        for protocol, fake in ((protocols.DocumentRepository, FakeDocuments()),
                               (protocols.ChunkRepository, FakeChunks()),
                               (protocols.VectorIndex, FakeVectorIndex()),
                               (protocols.GenerationProvider, FakeGeneration()),
                               (protocols.EscalationRepository, FakeEscalations())):
            self.assertIsInstance(fake, protocol)
            for name, member in vars(protocol).items():
                if inspect.isfunction(member) and not name.startswith("_"):
                    expected = inspect.signature(member)
                    actual = inspect.signature(getattr(type(fake), name))
                    self.assertEqual(list(expected.parameters), list(actual.parameters))
                    for parameter in expected.parameters:
                        self.assertEqual(expected.parameters[parameter].kind, actual.parameters[parameter].kind)
                    self.assertEqual(get_type_hints(member), get_type_hints(getattr(type(fake), name)))
        for name in ("RuntimeReranker", "ContextAssembler", "SufficiencyGate", "CitationValidator", "FinalAnswerPolicy"):
            protocol = getattr(protocols, name)
            self.assertTrue(protocol._is_protocol)
            for member in vars(protocol).values():
                if inspect.isfunction(member):
                    get_type_hints(member)

    def test_reranker_protocol_is_neutral_and_preserves_separate_scores(self):
        class FakeReranker:
            name = "synthetic-ranker"
            score_type = "synthetic-score"
            scorer_identity = "synthetic-ranker/revision"

            def rerank(self, question, candidates):
                return tuple(replace(item, reranker_score=float(i), reranker_score_type=self.score_type,
                                     reranker_scorer_identity=self.scorer_identity)
                             for i, item in enumerate(reversed(candidates)))
        _, candidates = example_corpus("laboratory", 3)
        fake = FakeReranker()
        self.assertIsInstance(fake, protocols.RuntimeReranker)
        result = fake.rerank("arbitrary", candidates)
        original = {item.chunk.identity: item.retrieval_score for item in candidates}
        self.assertEqual({item.chunk.identity for item in result}, set(original))
        for item in result:
            self.assertEqual(item.retrieval_score, original[item.chunk.identity])
            self.assertEqual(item.reranker_score_type, fake.score_type)


if __name__ == "__main__":
    unittest.main()