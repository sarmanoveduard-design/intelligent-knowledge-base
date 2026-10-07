"""Offline whole-context and technical sufficiency tests, without model execution."""
import ast
from dataclasses import fields, replace
import inspect
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from knowledge_base.runtime.config import RuntimeConfig, ScoreThreshold
from knowledge_base.runtime.context import ContextAssemblyError, RankedContextAssembler
from knowledge_base.runtime.models import (
    ChunkIdentity, ClarificationAssessment, ConflictAssessment, ConflictStatus, ContextEntry,
    ContextManifest, DocumentIdentity, DocumentVersionIdentity, DocumentVersionStatus,
    EvidenceChunk, ProcessingState, RuntimeDocument, RuntimeDocumentVersion, RuntimeEvidence,
    SourceCoordinates, SufficiencyStatus,
)
from knowledge_base.runtime.protocols import ClarificationPolicy, ConflictDetector, ContextAssembler, SufficiencyGate
from knowledge_base.runtime.sufficiency import SufficiencyPolicy

ROOT = Path(__file__).resolve().parents[1]
CORPORA = {
    'company': ('The fictional Cedar company receives supplies at its desk.',
                'Cedar records every incoming delivery.', 'Visitors sign the Cedar register.'),
    'laboratory': ('Glass containers in the fictional laboratory carry blue labels.',
                   'Polymer containers carry silver labels.', 'Sample labels are recorded before storage.'),
}


def make_evidence(texts=None, label='company'):
    texts = texts if texts is not None else CORPORA[label]
    document = DocumentIdentity('документ/自由/' + label, 'organization/' + label)
    version = DocumentVersionIdentity(document.document_id, 'version Ω/' + label)
    return tuple(RuntimeEvidence(
        EvidenceChunk(ChunkIdentity(document.document_id, version.version_id, 'snapshot#' + label,
                                    'fragment/opaque/' + label + '/' + str(i)),
                      document, version, text,
                      SourceCoordinates(source_file='source Ω.docx', source_format='docx',
                                        page_numbers=(i + 1,), paragraph_indices=(i,), section_ref='section/' + str(i)),
                      local_chunk_index=0, metadata={'nested': {'ordinal': i, 'tags': ['synthetic', label]}}),
        retrieval_score=0.9 - i * 0.2, retrieval_score_type='cosine', retrieval_scorer_identity='synthetic/space-v1',
        reranker_score=20.0 - i * 7, reranker_score_type='raw-logit', reranker_scorer_identity='fake-reranker/rev1',
    ) for i, text in enumerate(texts))


class FakeRegistry:
    """Only the authoritative read operations used by the policy are needed."""
    def __init__(self, evidence):
        self.chunks = {x.chunk.identity: x.chunk for x in evidence}
        self.documents = {x.chunk.document: RuntimeDocument(x.chunk.document, 'Synthetic source') for x in evidence}
        self.versions = {x.chunk.version: RuntimeDocumentVersion(x.chunk.version, 'current',
            DocumentVersionStatus.ACTIVE, ProcessingState.INDEXED, approved=True) for x in evidence}
        self.snapshots = {x.chunk.version: x.chunk.identity.snapshot for x in evidence}

    def get_document(self, identity): return self.documents.get(identity)
    def get_version(self, identity): return self.versions.get(identity)
    def get_current_snapshot(self, version): return self.snapshots.get(version)
    def get_chunk(self, identity): return self.chunks.get(identity)


class FixedConflict:
    def __init__(self, status=ConflictStatus.UNKNOWN):
        self.status, self.calls = status, []

    def detect(self, question, context):
        self.calls.append((question, context))
        return ConflictAssessment(self.status)


class FixedClarification:
    def __init__(self, missing=()):
        self.missing, self.calls = missing, []

    def assess(self, question, context):
        self.calls.append((question, context))
        return ClarificationAssessment(self.missing)


class RuntimeContextTests(unittest.TestCase):
    def setUp(self):
        self.evidence = make_evidence()
        self.repository = FakeRegistry(self.evidence)
        self.assembler = RankedContextAssembler()
        self.policy = SufficiencyPolicy(self.repository)
        self.config = RuntimeConfig()

    def manifest(self, evidence=None, budget=None):
        return self.assembler.assemble(self.evidence if evidence is None else evidence,
                                      max_chars=self.config.context_max_chars if budget is None else budget)

    def evaluate(self, *, evidence=None, context=None, config=None, policy=None):
        return (policy or self.policy).evaluate('A new arbitrary question Ω?',
            context if context is not None else self.manifest(evidence), config=config or self.config)

    def dense_threshold(self, value=0.4, **kwargs):
        return ScoreThreshold(value, kwargs.get('score_type', 'cosine'),
                              kwargs.get('scorer_identity', 'synthetic/space-v1'))

    def reranker_threshold(self, value=5.0, **kwargs):
        return ScoreThreshold(value, kwargs.get('score_type', 'raw-logit'),
                              kwargs.get('scorer_identity', 'fake-reranker/rev1'))

    def test_order_is_input_ranking_without_score_resorting(self):
        ranked = (self.evidence[2], self.evidence[0], self.evidence[1])
        context = self.manifest(ranked)
        self.assertEqual(tuple(x.evidence for x in context.entries), ranked)

    def test_manifest_contains_only_selected_whole_chunks(self):
        budget = len(self.evidence[0].chunk.text)
        context = self.manifest(budget=budget)
        self.assertEqual(tuple(x.evidence for x in context.entries), (self.evidence[0],))
        self.assertEqual(context.entries[0].evidence.chunk.text, self.evidence[0].chunk.text)
        self.assertEqual(context.diagnostics['used_chars'], budget)
        self.assertEqual(context.diagnostics['selected_evidence_count'], 1)
        self.assertEqual(context.diagnostics['input_evidence_count'], 3)

    def test_budget_boundary_is_inclusive(self):
        budget = sum(len(x.chunk.text) for x in self.evidence[:2])
        context = self.manifest(budget=budget)
        self.assertEqual(tuple(x.evidence for x in context.entries), self.evidence[:2])
        self.assertEqual(context.diagnostics['used_chars'], budget)

    def test_oversized_first_chunk_is_skipped_and_reported_without_truncation(self):
        evidence = make_evidence(('X' * 100, 'whole text', 'more'))
        context = self.manifest(evidence, budget=20)
        self.assertEqual(tuple(x.evidence for x in context.entries), evidence[1:])
        self.assertEqual(context.diagnostics['skipped'], ({'input_position': 0,
            'reason_code': 'OVERSIZED_CHUNK', 'text_chars': 100},))
        self.assertEqual([x.source_handle for x in context.entries], ['S1', 'S2'])
        self.assertEqual(context.diagnostics['oversized_behavior'], 'skip_whole_chunk')

    def test_nonfitting_middle_chunk_does_not_block_smaller_later_chunk(self):
        evidence = make_evidence(('aaa', 'bbbbbbbb', 'cc'))
        context = self.manifest(evidence, budget=9)
        self.assertEqual(tuple(x.evidence for x in context.entries), (evidence[0], evidence[2]))
        self.assertEqual(context.diagnostics['skipped'][0]['reason_code'], 'CONTEXT_BUDGET_EXCEEDED')
        self.assertLessEqual(context.diagnostics['used_chars'], 9)

    def test_all_oversized_returns_empty_manifest_with_diagnostics(self):
        context = self.manifest(budget=1)
        self.assertEqual(context.entries, ())
        self.assertEqual(context.diagnostics['used_chars'], 0)
        self.assertEqual(len(context.diagnostics['skipped']), len(self.evidence))
        decision = self.evaluate(context=context)
        self.assertEqual(decision.reason_code, 'NO_EVIDENCE')

    def test_duplicate_identity_keeps_first_representation(self):
        first = self.evidence[0]
        duplicate = replace(first, retrieval_score=0.99, reranker_score=100)
        context = self.manifest((first, duplicate, self.evidence[1], first))
        self.assertEqual(tuple(x.evidence for x in context.entries), self.evidence[:2])
        self.assertIs(context.entries[0].evidence, first)
        self.assertEqual([x['reason_code'] for x in context.diagnostics['skipped']], ['DUPLICATE_CHUNK'] * 2)

    def test_duplicate_oversized_identity_does_not_choose_changed_short_text(self):
        original = make_evidence(('a' * 100,))[0]
        changed = replace(original, chunk=replace(original.chunk, text='short'))
        context = self.manifest((original, changed), budget=20)
        self.assertEqual(context.entries, ())
        self.assertEqual([x['reason_code'] for x in context.diagnostics['skipped']],
                         ['OVERSIZED_CHUNK', 'DUPLICATE_CHUNK'])

    def test_distinct_chunks_with_same_text_or_local_index_are_preserved(self):
        a = make_evidence(('identical text',), 'company')[0]
        b = make_evidence(('identical text',), 'laboratory')[0]
        context = self.manifest((a, b))
        self.assertEqual(len(context.entries), 2)
        self.assertEqual({x.evidence.chunk.local_chunk_index for x in context.entries}, {0})
        self.assertNotEqual(context.entries[0].evidence.chunk.identity, context.entries[1].evidence.chunk.identity)

    def test_handles_are_stable_contiguous_and_resolve_exact_identity(self):
        evidence = (self.evidence[0], self.evidence[0], self.evidence[2])
        a, b = self.manifest(evidence), self.manifest(evidence)
        self.assertEqual(a, b)
        self.assertEqual([x.source_handle for x in a.entries], ['S1', 'S2'])
        self.assertEqual({x.source_handle: x.evidence.chunk.identity for x in a.entries},
                         {'S1': self.evidence[0].chunk.identity, 'S2': self.evidence[2].chunk.identity})

    def test_every_score_identity_coordinate_and_metadata_is_preserved(self):
        context = self.manifest()
        for original, entry in zip(self.evidence, context.entries):
            self.assertIs(entry.evidence, original)
            self.assertEqual(entry.evidence.chunk, original.chunk)
            self.assertEqual(entry.evidence.chunk.identity.snapshot, original.chunk.identity.snapshot)
            self.assertEqual(entry.evidence.chunk.coordinates, original.chunk.coordinates)
            self.assertEqual(entry.evidence.chunk.metadata, original.chunk.metadata)
            self.assertEqual(entry.evidence.retrieval_score, original.retrieval_score)
            self.assertEqual(entry.evidence.retrieval_score_type, original.retrieval_score_type)
            self.assertEqual(entry.evidence.retrieval_scorer_identity, original.retrieval_scorer_identity)
            self.assertEqual(entry.evidence.reranker_score, original.reranker_score)
            self.assertEqual(entry.evidence.reranker_score_type, original.reranker_score_type)
            self.assertEqual(entry.evidence.reranker_scorer_identity, original.reranker_scorer_identity)
        with self.assertRaises(TypeError):
            context.diagnostics['used_chars'] = 1
        with self.assertRaises(TypeError):
            context.entries[0].evidence.chunk.metadata['nested']['ordinal'] = 100

    def test_unicode_char_budget_is_not_bytes_or_tokens(self):
        evidence = make_evidence(('自由Ω🌱', 'абв'))
        context = self.manifest(evidence, budget=4)
        self.assertEqual(tuple(x.evidence for x in context.entries), evidence[:1])
        self.assertEqual(context.diagnostics['used_chars'], 4)
        self.assertEqual(context.diagnostics['budget_unit'], 'chunk_text_characters')

    def test_empty_evidence_creates_empty_manifest(self):
        context = self.manifest(())
        self.assertEqual(context.entries, ())
        self.assertEqual(context.chunk_ids, ())
        self.assertEqual(context.diagnostics['skipped'], ())
        self.assertEqual(context.diagnostics['used_chars'], 0)

    def test_arbitrary_new_text_and_domains_use_same_assembler(self):
        for label in CORPORA:
            with self.subTest(corpus=label):
                evidence = make_evidence(label=label)
                context = self.manifest(evidence)
                self.assertEqual(tuple(x.evidence for x in context.entries), evidence)
        evidence = make_evidence(('¿Новая инструкция для устройства δ-47?', '自由な文章'))
        self.assertEqual(tuple(x.evidence for x in self.manifest(evidence).entries), evidence)

    def test_invalid_budget_and_evidence_raise_safe_codes(self):
        for budget in (0, -1, True, 1.5, None):
            with self.subTest(budget=budget), self.assertRaises(ContextAssemblyError) as error:
                self.assembler.assemble(self.evidence, max_chars=budget)
            self.assertEqual(str(error.exception), 'INVALID_CONTEXT_BUDGET')
        with self.assertRaises(ContextAssemblyError) as error:
            self.manifest((object(),))
        self.assertEqual(str(error.exception), 'INVALID_CONTEXT_EVIDENCE')

    def test_empty_manifest_is_insufficient(self):
        result = self.evaluate(evidence=())
        self.assertEqual(result.status, SufficiencyStatus.INSUFFICIENT)
        self.assertEqual(result.reason_code, 'NO_EVIDENCE')
        self.assertEqual(result.evidence_count, 0)
        self.assertEqual(result.diagnostics['evidence_count'], 0)

    def test_below_configured_minimum_is_insufficient(self):
        result = self.evaluate(evidence=self.evidence[:1], config=replace(self.config, minimum_evidence_count=2))
        self.assertEqual(result.status, SufficiencyStatus.INSUFFICIENT)
        self.assertEqual(result.reason_code, 'BELOW_MINIMUM_EVIDENCE')
        self.assertEqual(result.diagnostics['minimum_evidence_count'], 2)
        self.assertEqual(result.evidence_count, 1)

    def test_minimum_one_allows_single_source(self):
        result = self.evaluate(evidence=self.evidence[:1])
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        self.assertEqual(result.reason_code, 'GATES_PASSED')
        self.assertEqual(result.evidence_count, 1)

    def test_retrieval_threshold_passes_inclusively(self):
        result = self.evaluate(evidence=self.evidence[:1], config=replace(self.config, dense_threshold=self.dense_threshold(0.9)))
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        gate = result.diagnostics['retrieval_threshold']
        self.assertEqual(gate['status'], 'passed')
        self.assertEqual(gate['threshold_value'], 0.9)
        self.assertEqual(gate['score_type'], 'cosine')
        self.assertEqual(gate['scorer_identity'], 'synthetic/space-v1')
        self.assertEqual(gate['observed_values'], (0.9,))

    def test_retrieval_threshold_requires_all_selected_entries_to_pass(self):
        result = self.evaluate(config=replace(self.config, dense_threshold=self.dense_threshold(0.8)))
        self.assertEqual(result.status, SufficiencyStatus.INSUFFICIENT)
        self.assertEqual(result.reason_code, 'RETRIEVAL_SCORE_TOO_LOW')
        self.assertEqual(result.diagnostics['retrieval_threshold']['status'], 'failed')
        self.assertEqual(result.diagnostics['retrieval_threshold']['observed_values'],
                         tuple(x.retrieval_score for x in self.evidence))

    def test_reranker_threshold_passes_inclusively_without_probability_conversion(self):
        result = self.evaluate(evidence=self.evidence[:1], config=replace(self.config, reranker_threshold=self.reranker_threshold(20)))
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        self.assertEqual(result.diagnostics['reranker_threshold']['observed_values'], (20.0,))
        self.assertEqual(result.diagnostics['reranker_threshold']['score_type'], 'raw-logit')

    def test_reranker_threshold_failure_is_insufficient(self):
        result = self.evaluate(config=replace(self.config, reranker_threshold=self.reranker_threshold(10)))
        self.assertEqual(result.status, SufficiencyStatus.INSUFFICIENT)
        self.assertEqual(result.reason_code, 'RERANKER_SCORE_TOO_LOW')
        self.assertEqual(result.diagnostics['reranker_threshold']['status'], 'failed')

    def test_negative_logit_threshold_is_valid(self):
        item = replace(self.evidence[0], reranker_score=-5)
        result = self.evaluate(evidence=(item,), config=replace(self.config, reranker_threshold=self.reranker_threshold(-6)))
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        self.assertEqual(result.diagnostics['reranker_threshold']['observed_values'], (-5,))

    def test_wrong_type_or_scorer_identity_never_compares_score(self):
        for name, threshold in (
            ('dense_threshold', self.dense_threshold(-100, score_type='BM25')),
            ('dense_threshold', self.dense_threshold(-100, score_type='RRF')),
            ('dense_threshold', self.dense_threshold(-100, scorer_identity='different-space')),
            ('reranker_threshold', self.reranker_threshold(-100, score_type='cosine')),
            ('reranker_threshold', self.reranker_threshold(-100, scorer_identity='other-revision')),
        ):
            with self.subTest(name=name, threshold=threshold):
                result = self.evaluate(config=replace(self.config, **{name: threshold}))
                self.assertEqual(result.status, SufficiencyStatus.ERROR)
                self.assertEqual(result.reason_code, 'SCORE_SPACE_MISMATCH')

    def test_mixed_score_spaces_are_rejected_for_all_entry_gate(self):
        other = replace(self.evidence[1], retrieval_score_type='BM25')
        result = self.evaluate(evidence=(self.evidence[0], other),
                               config=replace(self.config, dense_threshold=self.dense_threshold(-100)))
        self.assertEqual(result.reason_code, 'SCORE_SPACE_MISMATCH')
        self.assertEqual(result.diagnostics['retrieval_threshold']['observed_score_types'], ('cosine', 'BM25'))

    def test_unconfigured_thresholds_are_explicitly_skipped(self):
        result = self.evaluate()
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        self.assertEqual(result.diagnostics['unconfigured_threshold_policy'], 'skip')
        for name in ('retrieval_threshold', 'reranker_threshold'):
            self.assertIn(name, result.diagnostics['skipped_gates'])
            self.assertNotIn(name, result.diagnostics['applied_gates'])
            self.assertEqual(result.diagnostics[name]['status'], 'skipped_unconfigured')
            self.assertIsNone(result.diagnostics[name]['threshold_value'])
        self.assertTrue(result.diagnostics['retrieval_threshold']['observed_values'])

    def test_require_thresholds_blocks_missing_configuration(self):
        for kwargs in ({}, {'dense_threshold': self.dense_threshold()},
                       {'reranker_threshold': self.reranker_threshold()}):
            with self.subTest(kwargs=kwargs):
                config = replace(self.config, unconfigured_threshold_policy='require', **kwargs)
                result = self.evaluate(config=config)
                self.assertEqual(result.status, SufficiencyStatus.INSUFFICIENT)
                self.assertEqual(result.reason_code, 'THRESHOLD_UNCONFIGURED')
        config = replace(self.config, unconfigured_threshold_policy='require',
                         dense_threshold=self.dense_threshold(), reranker_threshold=self.reranker_threshold())
        self.assertEqual(self.evaluate(config=config).status, SufficiencyStatus.SUFFICIENT)

    def test_invalid_threshold_policy_configuration_is_rejected(self):
        for value in ('automatic', '', True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.config, unconfigured_threshold_policy=value)

    def test_reranker_score_absent_with_unset_gate_is_allowed_and_reported(self):
        item = replace(self.evidence[0], reranker_score=None, reranker_score_type=None, reranker_scorer_identity=None)
        result = self.evaluate(evidence=(item,))
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        self.assertEqual(result.diagnostics['reranker_threshold']['observed_values'], (None,))
        self.assertEqual(result.diagnostics['reranker_threshold']['status'], 'skipped_unconfigured')

    def test_configured_reranker_gate_without_score_is_insufficient(self):
        item = replace(self.evidence[0], reranker_score=None, reranker_score_type=None, reranker_scorer_identity=None)
        result = self.evaluate(evidence=(item,), config=replace(self.config, reranker_threshold=self.reranker_threshold()))
        self.assertEqual(result.status, SufficiencyStatus.INSUFFICIENT)
        self.assertEqual(result.reason_code, 'RERANKER_SCORE_MISSING')

    def test_nonfinite_scores_are_rejected_by_constructor_and_integrity_boundaries(self):
        for field in ('retrieval_score', 'reranker_score'):
            for value in (float('nan'), float('inf'), float('-inf')):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        replace(self.evidence[0], **{field: value})
                    item = replace(self.evidence[0])
                    object.__setattr__(item, field, value)
                    with self.assertRaises(ContextAssemblyError):
                        self.manifest((item,))
                    context = ContextManifest((ContextEntry('S1', item),))
                    result = self.evaluate(context=context)
                    self.assertEqual(result.status, SufficiencyStatus.ERROR)
                    self.assertEqual(result.reason_code, 'INVALID_PROVENANCE')
                    self.assertEqual(result.evidence_count, 1)

    def test_unknown_document_version_snapshot_or_chunk_is_integrity_error(self):
        for store, key in [('documents', self.evidence[0].chunk.document),
                           ('versions', self.evidence[0].chunk.version),
                           ('snapshots', self.evidence[0].chunk.version),
                           ('chunks', self.evidence[0].chunk.identity)]:
            with self.subTest(store=store):
                registry = FakeRegistry(self.evidence)
                getattr(registry, store).pop(key)
                result = self.evaluate(evidence=self.evidence[:1], policy=SufficiencyPolicy(registry))
                self.assertEqual(result.status, SufficiencyStatus.ERROR)
                self.assertEqual(result.reason_code, 'INVALID_PROVENANCE')

    def test_changed_text_source_coordinates_or_metadata_is_integrity_error(self):
        mutations = {'text': 'fabricated text', 'coordinates': SourceCoordinates(section_ref='different'),
                     'metadata': {'fabricated': True}, 'local_chunk_index': 100}
        for field, value in mutations.items():
            with self.subTest(field=field):
                item = replace(self.evidence[0], chunk=replace(self.evidence[0].chunk, **{field: value}))
                result = self.evaluate(evidence=(item,))
                self.assertEqual(result.status, SufficiencyStatus.ERROR)
                self.assertEqual(result.reason_code, 'INVALID_PROVENANCE')

    def test_damaged_identity_and_incompatible_metadata_are_integrity_errors(self):
        for field, value in [('identity', DocumentIdentity('wrong kind', 'org')),
                             ('metadata', {'value': object()})]:
            with self.subTest(field=field):
                chunk = replace(self.evidence[0].chunk)
                item = replace(self.evidence[0], chunk=chunk)
                context = ContextManifest((ContextEntry('S1', item),))
                object.__setattr__(chunk, field, value)
                result = self.evaluate(context=context)
                self.assertEqual(result.reason_code, 'INVALID_PROVENANCE')

    def test_corrupted_duplicate_handles_are_integrity_error(self):
        context = self.manifest()
        object.__setattr__(context.entries[1], 'source_handle', 'S1')
        self.assertEqual(self.evaluate(context=context).reason_code, 'INVALID_PROVENANCE')

    def test_source_lifecycle_changes_are_integrity_errors(self):
        version_key = self.evidence[0].chunk.version
        for mutation in ({'status': DocumentVersionStatus.DRAFT}, {'status': DocumentVersionStatus.SUPERSEDED},
                         {'status': DocumentVersionStatus.ARCHIVED}, {'processing_state': ProcessingState.CHUNKED},
                         {'approved': False}):
            with self.subTest(mutation=mutation):
                registry = FakeRegistry(self.evidence)
                registry.versions[version_key] = replace(registry.versions[version_key], **mutation)
                result = self.evaluate(policy=SufficiencyPolicy(registry))
                self.assertEqual(result.reason_code, 'INVALID_PROVENANCE')

    def test_unavailable_registry_is_error_without_leaking_exception(self):
        with patch.object(self.repository, 'get_chunk', side_effect=RuntimeError('SECRET_DATABASE_DETAIL')):
            result = self.evaluate()
        self.assertEqual(result.status, SufficiencyStatus.ERROR)
        self.assertEqual(result.reason_code, 'PROVENANCE_CHECK_FAILED')
        self.assertNotIn('SECRET', str(result))

    def test_invalid_provenance_precedes_low_scores_and_minimum_count(self):
        item = replace(self.evidence[0], chunk=replace(self.evidence[0].chunk, text='changed source'))
        result = self.evaluate(evidence=(item,), config=replace(self.config, minimum_evidence_count=2,
            dense_threshold=self.dense_threshold(100)))
        self.assertEqual(result.status, SufficiencyStatus.ERROR)
        self.assertEqual(result.reason_code, 'INVALID_PROVENANCE')

    def test_policy_checks_actual_budget_instead_of_trusting_manifest_diagnostics(self):
        context = replace(self.manifest(), diagnostics={'used_chars': 0, 'SECRET': 'PRIVATE_VALUE'})
        result = self.evaluate(context=context, config=replace(self.config, context_max_chars=1))
        self.assertEqual(result.status, SufficiencyStatus.ERROR)
        self.assertEqual(result.reason_code, 'CONTEXT_BUDGET_EXCEEDED')
        self.assertGreater(result.diagnostics['context_text_chars'], result.diagnostics['context_max_chars'])
        self.assertNotIn('PRIVATE', str(result))

    def test_conflict_hook_reports_conflict_only_on_final_manifest(self):
        hook = FixedConflict(ConflictStatus.CONFLICT)
        context = self.manifest(self.evidence[:1])
        result = self.evaluate(context=context, policy=replace(self.policy, conflict_detector=hook))
        self.assertEqual(result.status, SufficiencyStatus.CONFLICT)
        self.assertEqual(result.reason_code, 'CONFLICT_DETECTED')
        self.assertEqual(result.diagnostics['conflict_hook_result'], 'conflict')
        self.assertIs(hook.calls[0][1], context)
        self.assertEqual(len(hook.calls[0][1].entries), 1)
        self.assertEqual(result.missing_information, ())

    def test_default_no_conflict_hook_asserts_only_unknown(self):
        result = self.evaluate()
        self.assertEqual(result.diagnostics['conflict_hook_result'], 'unknown')
        self.assertIn('conflict_hook', result.diagnostics['skipped_gates'])
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)

    def test_unknown_and_clear_conflict_hooks_do_not_assert_conflict(self):
        for status in (ConflictStatus.UNKNOWN, ConflictStatus.CLEAR):
            with self.subTest(status=status):
                result = self.evaluate(policy=replace(self.policy, conflict_detector=FixedConflict(status)))
                self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
                self.assertEqual(result.diagnostics['conflict_hook_result'], status.value)
                self.assertIn('conflict_hook', result.diagnostics['applied_gates'])

    def test_clarification_hook_returns_only_explicit_missing_parameters(self):
        hook = FixedClarification(('sample category', 'effective date'))
        result = self.evaluate(policy=replace(self.policy, clarification_policy=hook))
        self.assertEqual(result.status, SufficiencyStatus.CLARIFICATION_REQUIRED)
        self.assertEqual(result.reason_code, 'CLARIFICATION_REQUIRED')
        self.assertEqual(result.missing_information, ('sample category', 'effective date'))
        self.assertEqual(result.diagnostics['clarification_hook_result'], 'required')
        self.assertEqual(result.diagnostics['missing_parameters'], result.missing_information)

    def test_default_policy_never_guesses_missing_parameters_from_question(self):
        result = self.policy.evaluate('An ambiguous request without a date or identifier?', self.manifest(), config=self.config)
        self.assertEqual(result.missing_information, ())
        self.assertEqual(result.diagnostics['missing_parameters'], ())
        self.assertEqual(result.diagnostics['clarification_hook_result'], 'not_configured')
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)

    def test_clear_clarification_hook_allows_gates_to_pass(self):
        result = self.evaluate(policy=replace(self.policy, clarification_policy=FixedClarification()))
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        self.assertEqual(result.diagnostics['clarification_hook_result'], 'clear')

    def test_conflict_has_precedence_over_clarification(self):
        clarification = FixedClarification(('date',))
        result = self.evaluate(policy=replace(self.policy, conflict_detector=FixedConflict(ConflictStatus.CONFLICT),
                                             clarification_policy=clarification))
        self.assertEqual(result.status, SufficiencyStatus.CONFLICT)
        self.assertEqual(clarification.calls, [])
        self.assertEqual(result.diagnostics['clarification_hook_result'], 'not_run')

    def test_empty_or_low_evidence_does_not_invoke_hooks(self):
        conflict, clarification = FixedConflict(), FixedClarification()
        policy = replace(self.policy, conflict_detector=conflict, clarification_policy=clarification)
        self.evaluate(evidence=(), policy=policy)
        result = self.evaluate(policy=policy, config=replace(self.config, dense_threshold=self.dense_threshold(100)))
        self.assertEqual(result.status, SufficiencyStatus.INSUFFICIENT)
        self.assertEqual(conflict.calls, [])
        self.assertEqual(clarification.calls, [])

    def test_hook_errors_or_invalid_return_types_are_safe_errors(self):
        for kind, hook, method in [('conflict_detector', FixedConflict(), 'detect'),
                                   ('clarification_policy', FixedClarification(), 'assess')]:
            for outcome in ('raise', 'wrong_type'):
                with self.subTest(kind=kind, outcome=outcome):
                    kwargs = {'side_effect': RuntimeError('SECRET_HOOK_INPUT')} if outcome == 'raise' else {'return_value': True}
                    with patch.object(hook, method, **kwargs):
                        result = self.evaluate(policy=replace(self.policy, **{kind: hook}))
                    self.assertEqual(result.status, SufficiencyStatus.ERROR)
                    self.assertEqual(result.reason_code, 'HOOK_FAILED')
                    self.assertNotIn('SECRET', str(result))

    def test_hook_result_models_validate_status_and_missing_names(self):
        with self.assertRaises(ValueError):
            ConflictAssessment('conflict')
        for values in (('date', 'date'), ('',), 'date'):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ClarificationAssessment(values)
        self.assertEqual(ClarificationAssessment(['a', 'b']).missing_parameters, ('a', 'b'))

    def test_gates_passed_is_technical_permission_not_semantic_proof(self):
        result = self.policy.evaluate('What is a completely unrelated astronomical constant?',
                                      self.manifest(), config=self.config)
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        self.assertEqual(result.reason_code, 'GATES_PASSED')
        self.assertEqual(result.diagnostics['semantic_answerability'], 'not_checked')
        self.assertIn('semantic answerability is not established', result.message)
        self.assertIn('does not prove an answer', SufficiencyPolicy.__doc__)

    def test_both_corpora_use_the_same_policy_without_domain_rules(self):
        for label in CORPORA:
            with self.subTest(corpus=label):
                evidence = make_evidence(label=label)
                policy = SufficiencyPolicy(FakeRegistry(evidence))
                result = policy.evaluate('Any new question', self.manifest(evidence), config=self.config)
                self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
                self.assertEqual(result.evidence_count, len(evidence))

    def test_selection_affects_sufficiency_only_through_exact_final_manifest(self):
        # The low scoring third candidate is not part of context and cannot fail its gate.
        context = self.manifest(self.evidence, budget=len(self.evidence[0].chunk.text))
        result = self.evaluate(context=context, config=replace(self.config, dense_threshold=self.dense_threshold(0.8)))
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        self.assertEqual(result.diagnostics['retrieval_threshold']['observed_values'], (0.9,))
        result = self.evaluate(context=context, config=replace(self.config, minimum_evidence_count=2))
        self.assertEqual(result.reason_code, 'BELOW_MINIMUM_EVIDENCE')

    def test_context_and_decision_diagnostics_are_deterministic_and_immutable(self):
        config = replace(self.config, dense_threshold=self.dense_threshold(), reranker_threshold=self.reranker_threshold())
        a, b = self.manifest(), self.manifest()
        self.assertEqual(a, b)
        first, second = self.evaluate(context=a, config=config), self.evaluate(context=b, config=config)
        self.assertEqual(first, second)
        self.assertEqual(first.diagnostics['applied_gates'],
                         ('provenance', 'context_budget', 'minimum_evidence', 'retrieval_threshold', 'reranker_threshold'))
        self.assertEqual(first.diagnostics['skipped_gates'], ('conflict_hook', 'clarification_hook'))
        with self.assertRaises(TypeError):
            first.diagnostics['retrieval_threshold']['status'] = 'changed'

    def test_invalid_question_configuration_and_corrupt_threshold_return_error(self):
        context = self.manifest()
        self.assertEqual(self.policy.evaluate(' ', context, config=self.config).reason_code, 'INVALID_QUESTION')
        self.assertEqual(self.policy.evaluate('question', context, config=None).reason_code, 'INVALID_CONFIGURATION')
        threshold = self.dense_threshold()
        object.__setattr__(threshold, 'value', float('nan'))
        result = self.evaluate(config=replace(self.config, dense_threshold=threshold))
        self.assertEqual(result.reason_code, 'INVALID_CONFIGURATION')
        self.assertEqual(result.evidence_count, len(context.entries))

    def test_protocol_conformance(self):
        self.assertIsInstance(self.assembler, ContextAssembler)
        self.assertIsInstance(self.policy, SufficiencyGate)
        self.assertIsInstance(FixedConflict(), ConflictDetector)
        self.assertIsInstance(FixedClarification(), ClarificationPolicy)

    def test_existing_retrieval_can_feed_new_components_without_changes(self):
        import test_runtime_retrieval as retrieval_tests
        fixture = retrieval_tests.RuntimeRetrievalTests('test_both_different_corpora_use_same_runtime')
        fixture.setUp()
        try:
            fixture.ready()
            retrieval = fixture.search()
            context = self.assembler.assemble(retrieval.evidence, max_chars=fixture.pipeline.config.context_max_chars)
            decision = SufficiencyPolicy(fixture.repository).evaluate('New question', context, config=fixture.pipeline.config)
            self.assertEqual(decision.status, SufficiencyStatus.SUFFICIENT)
            self.assertEqual(tuple(x.evidence for x in context.entries), retrieval.evidence)
        finally:
            fixture.doCleanups()

    def test_modules_have_no_benchmark_labels_fixed_domains_or_model_dependencies(self):
        forbidden = {'answer_hint', 'gold', 'gold_labels', 'forbidden_claims', 'expected_refusal',
                     'expected_answer', 'expected_behavior', 'query_id', 'benchmark_probes'}
        for path in (ROOT / 'src/knowledge_base/runtime/context.py', ROOT / 'src/knowledge_base/runtime/sufficiency.py'):
            source = path.read_text(encoding='utf8')
            tree = ast.parse(source)
            names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
            self.assertFalse(forbidden & (names | literals))
            numeric = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and type(node.value) is int}
            self.assertFalse(numeric & {119, 149, 10, 8})
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level == 0:
                        self.assertIn(node.module.split('.')[0], sys.stdlib_module_names)
                    else:
                        self.assertEqual(node.level, 1)
        for callable_ in (self.assembler.assemble, self.policy.evaluate):
            self.assertFalse(forbidden & set(inspect.signature(callable_).parameters))
        self.assertFalse(forbidden & {f.name for f in fields(ContextManifest)})

    def test_stage4_execution_does_not_import_benchmark_or_model_modules(self):
        code = r'''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('knowledge_base.benchmark', 'knowledge_base.team', 'benchmarks', 'scripts',
                                'knowledge_base.bge_reranker', 'knowledge_base.candidate_retrieval',
                                'torch', 'transformers', 'openai', 'ollama')):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Reject())
from test_runtime_context import RuntimeContextTests
case = RuntimeContextTests('test_both_corpora_use_the_same_policy_without_domain_rules')
case.setUp()
case.test_both_corpora_use_the_same_policy_without_domain_rules()
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, 'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(ROOT / 'tests'))),
                 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
