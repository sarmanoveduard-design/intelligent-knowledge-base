"""Offline answering integration with two deterministic providers and real runtime gates."""
import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_runtime_retrieval as retrieval_tests
from knowledge_base.runtime.ask import AskService
from knowledge_base.runtime.citations import StructuralCitationValidator
from knowledge_base.runtime.config import RuntimeConfig
from knowledge_base.runtime.context import RankedContextAssembler
from knowledge_base.runtime.generation import (
    GenerationOutputError, GenerationRequestBuilder, OUTPUT_SCHEMA_VERSION, PRODUCTION_INSTRUCTIONS,
    validate_generation_draft,
)
from knowledge_base.runtime.models import (
    AnswerState, AskRequest, AskResult, CitationReference, ClarificationAssessment,
    ConflictAssessment, ConflictStatus, ContextManifest, CitationValidationResult, GenerationDraft, GenerationLimits,
    GenerationRequest, SemanticSupport, StructuralValidity, SufficiencyDecision, SufficiencyStatus,
)
from knowledge_base.runtime.policy import MemoryEscalationRepository, RuntimeFinalAnswerPolicy
from knowledge_base.runtime.protocols import CitationValidator, EscalationRepository, FinalAnswerPolicy, GenerationProvider
from knowledge_base.runtime.sufficiency import SufficiencyPolicy

ROOT = Path(__file__).resolve().parents[1]


class FakeProviderA:
    def __init__(self, outcome=None):
        self.calls, self.outcome = [], outcome

    def generate(self, request):
        self.calls.append(request)
        if self.outcome is not None:
            return self.outcome(request)
        return GenerationDraft(request.context_manifest.entries[0].evidence.chunk.text,
            tuple(CitationReference(x.source_handle, x.evidence.chunk.identity) for x in request.context_manifest.entries),
            provider_metadata={'provider': 'fake-alpha'})


class FakeProviderB:
    def __init__(self): self.calls = []

    def generate(self, request):
        self.calls.append(request)
        entry = request.context_manifest.entries[-1]
        return GenerationDraft('Selected source: ' + entry.evidence.chunk.text,
            (CitationReference(entry.source_handle, entry.evidence.chunk.identity),),
            provider_metadata={'provider': 'fake-beta'})


class ConflictHook:
    def detect(self, question, context): return ConflictAssessment(ConflictStatus.CONFLICT)


class ClarificationHook:
    def assess(self, question, context): return ClarificationAssessment(('effective date',))


class ErrorGate:
    def evaluate(self, question, context, *, config):
        return SufficiencyDecision(SufficiencyStatus.ERROR, 'INVALID_PROVENANCE', 'Integrity failure',
                                   len(context.entries), {'gate': 'synthetic-integrity'})


class RuntimeAnsweringTests(unittest.TestCase):
    def setUp(self):
        self.fixture = retrieval_tests.RuntimeRetrievalTests('test_both_different_corpora_use_same_runtime')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.snapshot = self.fixture.ready()
        self.provider = FakeProviderA()
        self.service = AskService(self.fixture.pipeline, RankedContextAssembler(), SufficiencyPolicy(self.fixture.repository),
            self.provider, StructuralCitationValidator(), RuntimeFinalAnswerPolicy(), config=self.fixture.pipeline.config)
        self.request = AskRequest('request/自由 Ω', 'An arbitrary previously unseen question δ-47?', self.fixture.access)

    def ask(self, **kwargs):
        return self.service.ask(replace(self.request, **kwargs))

    def draft(self, request, *, handle=None, identity=None, limitations=()):
        entry = request.context_manifest.entries[0]
        return GenerationDraft('A deterministic draft.',
            (CitationReference(handle or entry.source_handle, identity or entry.evidence.chunk.identity),), limitations)

    def invalid_provider(self):
        return FakeProviderA(lambda request: self.draft(request, handle='invented/source'))

    def test_insufficient_context_never_calls_provider(self):
        result = self.ask(retrieval_filters=replace(self.request.retrieval_filters, allowed_document_ids=frozenset()))
        self.assertEqual(result.state, AnswerState.REFUSE_INSUFFICIENT_CONTEXT)
        self.assertEqual(result.reason_code, 'NO_EVIDENCE')
        self.assertEqual(self.provider.calls, [])
        self.assertFalse(result.diagnostics['generation']['called'])

    def test_clarification_never_calls_provider_and_preserves_missing_parameter(self):
        self.service = replace(self.service, sufficiency=replace(self.service.sufficiency, clarification_policy=ClarificationHook()))
        result = self.ask()
        self.assertEqual(result.state, AnswerState.CLARIFY)
        self.assertEqual(result.missing_information, ('effective date',))
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(result.diagnostics['sufficiency']['status'], 'clarification_required')

    def test_conflict_never_calls_provider(self):
        self.service = replace(self.service, sufficiency=replace(self.service.sufficiency, conflict_detector=ConflictHook()))
        result = self.ask()
        self.assertEqual(result.state, AnswerState.CONFLICT)
        self.assertEqual(result.reason_code, 'CONFLICT_DETECTED')
        self.assertEqual(self.provider.calls, [])

    def test_sufficiency_integrity_error_never_calls_provider(self):
        self.service = replace(self.service, sufficiency=ErrorGate())
        result = self.ask()
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertEqual(result.reason_code, 'INVALID_PROVENANCE')
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(result.diagnostics['sufficiency']['diagnostics']['gate'], 'synthetic-integrity')

    def test_sufficient_calls_provider_exactly_once_and_returns_answer(self):
        result = self.ask()
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(result.state, AnswerState.ANSWER)
        self.assertEqual(result.reason_code, 'ANSWER_READY')
        self.assertTrue(result.answer)
        self.assertEqual(result.citation_validation.structural_validity, StructuralValidity.PASS)
        self.assertEqual(result.citation_validation.semantic_support, SemanticSupport.NOT_CHECKED)

    def test_request_whitelist_contains_no_access_filters_candidate_pool_or_labels(self):
        result = self.ask()
        request = self.provider.calls[0]
        self.assertEqual({x.name for x in fields(GenerationRequest)}, {
            'question', 'context_manifest', 'prompt_version', 'output_schema_version',
            'language', 'limits', 'instructions', 'request_id',
        })
        for name in ('access', 'retrieval_filters', 'metadata', 'query_id', 'expected_answer',
                     'answer_hint', 'expected_behavior', 'expected_refusal', 'forbidden_claims', 'gold', 'probes'):
            self.assertFalse(hasattr(request, name), name)
        self.assertIs(request.context_manifest, result.context_manifest)
        self.assertEqual(len(request.context_manifest.entries), 2)
        self.assertEqual(len(self.fixture.repository.list_snapshot(self.snapshot)), 3)

    def test_exact_source_handles_and_identities_reach_provider(self):
        result = self.ask()
        request = self.provider.calls[0]
        self.assertEqual([x.source_handle for x in request.context_manifest.entries], ['S1', 'S2'])
        for supplied, final in zip(request.context_manifest.entries, result.context_manifest.entries):
            self.assertIs(supplied, final)
            self.assertEqual(supplied.evidence.chunk.identity, final.evidence.chunk.identity)
            self.assertEqual(supplied.evidence.chunk.version, final.evidence.chunk.version)

    def test_unknown_source_handle_is_rejected(self):
        self.service = replace(self.service, provider=self.invalid_provider())
        result = self.ask()
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertEqual(result.reason_code, 'INVALID_CITATIONS')
        self.assertIn('UNKNOWN_SOURCE_HANDLE', result.citation_validation.errors)
        self.assertIsNone(result.answer)

    def test_source_outside_final_context_is_rejected_even_when_retrieved_as_candidate(self):
        outside = self.fixture.repository.list_snapshot(self.snapshot)[0]
        provider = FakeProviderA(lambda request: self.draft(request, identity=outside.identity))
        result = replace(self.service, provider=provider).ask(self.request)
        self.assertNotIn(outside.identity, {x.evidence.chunk.identity for x in result.context_manifest.entries})
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertIn('SOURCE_OUTSIDE_CONTEXT', result.citation_validation.errors)
        self.assertIn('CHUNK_IDENTITY_MISMATCH', result.citation_validation.errors)

    def test_wrong_document_version_snapshot_and_chunk_identity_are_rejected(self):
        for field in ('document_id', 'version_id', 'processing_snapshot_id', 'chunk_id'):
            with self.subTest(field=field):
                def wrong(request):
                    original = request.context_manifest.entries[0].evidence.chunk.identity
                    return self.draft(request, identity=replace(original, **{field: 'wrong/opaque'}))
                result = replace(self.service, provider=FakeProviderA(wrong)).ask(self.request)
                self.assertEqual(result.state, AnswerState.ERROR)
                self.assertIn('CHUNK_IDENTITY_MISMATCH', result.citation_validation.errors)

    def test_cross_handle_identity_swap_is_rejected(self):
        provider = FakeProviderA(lambda request: self.draft(request,
            identity=request.context_manifest.entries[1].evidence.chunk.identity))
        result = replace(self.service, provider=provider).ask(self.request)
        self.assertIn('CHUNK_IDENTITY_MISMATCH', result.citation_validation.errors)
        self.assertEqual(result.state, AnswerState.ERROR)

    def test_duplicate_citations_are_deduplicated_in_first_occurrence_order(self):
        def duplicate(request):
            draft = self.draft(request)
            return replace(draft, citations=draft.citations * 3)
        result = replace(self.service, provider=FakeProviderA(duplicate)).ask(self.request)
        self.assertEqual(result.state, AnswerState.ANSWER)
        self.assertEqual(len(result.citation_validation.resolved_citations), 1)
        self.assertEqual(result.citation_validation.diagnostics['duplicate_citation_count'], 2)
        self.assertEqual(result.citation_validation.diagnostics['citation_count'], 3)

    def test_no_citations_is_structural_failure(self):
        provider = FakeProviderA(lambda request: GenerationDraft('Unreferenced answer.'))
        result = replace(self.service, provider=provider).ask(self.request)
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertEqual(result.citation_validation.errors, ('NO_CITATIONS',))

    def test_structural_pass_never_claims_semantic_support(self):
        self.provider.outcome = lambda request: replace(self.draft(request), answer='A claim unrelated to the source.')
        result = self.ask()
        self.assertEqual(result.state, AnswerState.ANSWER)
        self.assertEqual(result.citation_validation.semantic_support, SemanticSupport.NOT_CHECKED)
        self.assertEqual(result.diagnostics['citation']['semantic_support'], 'not_checked')
        self.assertEqual(result.diagnostics['sufficiency']['diagnostics']['semantic_answerability'], 'not_checked')

    def test_provider_exception_returns_error_without_secret_details(self):
        def fail(request): raise RuntimeError('PRIVATE_PROVIDER_TOKEN')
        self.provider.outcome = fail
        result = self.ask()
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertEqual(result.reason_code, 'GENERATION_FAILED')
        self.assertEqual(len(self.provider.calls), 1)
        self.assertIsNone(result.answer)
        self.assertNotIn('PRIVATE', str(result))
        self.assertEqual(result.diagnostics['sufficiency']['status'], 'sufficient')
        self.assertEqual(result.diagnostics['generation']['status'], 'failed')

    def test_empty_provider_outputs_return_error(self):
        for output in (None, '', '   '):
            with self.subTest(output=output):
                provider = FakeProviderA(lambda request: output)
                result = replace(self.service, provider=provider).ask(self.request)
                self.assertEqual(result.state, AnswerState.ERROR)
                self.assertEqual(result.reason_code, 'EMPTY_GENERATION_OUTPUT')
                self.assertEqual(len(provider.calls), 1)

    def test_empty_typed_draft_returns_error(self):
        def empty(request):
            draft = self.draft(request)
            object.__setattr__(draft, 'answer', '')
            return draft
        result = replace(self.service, provider=FakeProviderA(empty)).ask(self.request)
        self.assertEqual(result.reason_code, 'EMPTY_GENERATION_OUTPUT')
        self.assertEqual(result.state, AnswerState.ERROR)

    def test_malformed_structured_outputs_return_error(self):
        for output in ('free text', {'answer': 'unparsed structured dict'}, [], object(), object.__new__(GenerationDraft)):
            with self.subTest(output=type(output).__name__):
                provider = FakeProviderA(lambda request: output)
                result = replace(self.service, provider=provider).ask(self.request)
                self.assertEqual(result.state, AnswerState.ERROR)
                self.assertEqual(result.reason_code, 'MALFORMED_GENERATION_OUTPUT')

    def test_malformed_nested_citations_or_metadata_return_error(self):
        for mutation in ('citations', 'metadata'):
            with self.subTest(mutation=mutation):
                def malformed(request):
                    draft = self.draft(request)
                    if mutation == 'citations':
                        return replace(draft, citations=({'source_handle': 'S1'},))
                    object.__setattr__(draft, 'provider_metadata', {'invalid': float('nan')})
                    return draft
                result = replace(self.service, provider=FakeProviderA(malformed)).ask(self.request)
                self.assertEqual(result.state, AnswerState.ERROR)
                self.assertEqual(result.reason_code, 'MALFORMED_GENERATION_OUTPUT')

    def test_partial_answers_disabled_by_default(self):
        self.assertFalse(RuntimeConfig().partial_answers_enabled)
        self.provider.outcome = lambda request: self.draft(request, limitations=('Requested detail is absent.',))
        result = self.ask()
        self.assertEqual(result.state, AnswerState.REFUSE_INSUFFICIENT_CONTEXT)
        self.assertEqual(result.reason_code, 'PARTIAL_ANSWER_DISABLED')
        self.assertIsNone(result.answer)
        self.assertEqual(result.declared_limitations, ('Requested detail is absent.',))

    def test_explicit_partial_answers_enabled_requires_valid_citations(self):
        provider = FakeProviderA(lambda request: self.draft(request, limitations=('Requested detail is absent.',)))
        service = replace(self.service, provider=provider, config=replace(self.service.config, partial_answers_enabled=True))
        result = service.ask(self.request)
        self.assertEqual(result.state, AnswerState.PARTIAL_ANSWER)
        self.assertTrue(result.answer)
        result = replace(service, provider=self.invalid_provider()).ask(self.request)
        self.assertEqual(result.state, AnswerState.ERROR)

    def test_two_fake_providers_use_same_contract_and_pipeline(self):
        for provider in (FakeProviderA(), FakeProviderB()):
            with self.subTest(provider=type(provider).__name__):
                result = replace(self.service, provider=provider).ask(self.request)
                self.assertEqual(result.state, AnswerState.ANSWER)
                self.assertEqual(len(provider.calls), 1)
                self.assertIsInstance(provider, GenerationProvider)

    def test_company_and_laboratory_corpora_use_same_runtime(self):
        for label, texts in retrieval_tests.CORPORA.items():
            with self.subTest(corpus=label):
                snapshot = self.fixture.ready(texts)
                request = replace(self.request, question='A new question about ' + label,
                    retrieval_filters=replace(self.request.retrieval_filters, allowed_document_ids={snapshot.version.document_id}))
                result = self.service.ask(request)
                self.assertEqual(result.state, AnswerState.ANSWER)
                self.assertTrue(all(x.evidence.chunk.version == snapshot.version for x in result.context_manifest.entries))
                self.assertIn(result.answer, texts)

    def test_arbitrary_question_language_and_request_id_pass_through(self):
        request = replace(self.request, request_id='id/自由-77', question='¿Как работает совершенно новый прибор Ω?', language='xx-Ω')
        result = self.service.ask(request)
        supplied = self.provider.calls[0]
        self.assertEqual(supplied.question, request.question)
        self.assertEqual(supplied.request_id, request.request_id)
        self.assertEqual(supplied.language, request.language)
        self.assertEqual(result.request_id, request.request_id)

    def test_manifest_after_sufficiency_is_same_object_with_exact_records(self):
        manifests = []
        evaluate = self.service.sufficiency.evaluate
        def capture(question, context, *, config):
            manifests.append(context)
            return evaluate(question, context, config=config)
        with patch.object(SufficiencyPolicy, 'evaluate', side_effect=capture):
            result = self.ask()
        self.assertIs(self.provider.calls[0].context_manifest, manifests[0])
        self.assertIs(result.context_manifest, manifests[0])
        self.assertEqual(result.context_manifest.entries, manifests[0].entries)
        with self.assertRaises(FrozenInstanceError):
            self.provider.calls[0].context_manifest.entries = ()

    def test_source_handles_deterministic_across_repeated_questions(self):
        first, second = self.ask(), self.ask()
        self.assertEqual(first.context_manifest, second.context_manifest)
        self.assertEqual(first.citation_validation, second.citation_validation)
        self.assertEqual(first.diagnostics, second.diagnostics)

    def test_generic_prompt_and_schema_are_provider_neutral(self):
        self.ask()
        request = self.provider.calls[0]
        self.assertEqual(request.instructions, PRODUCTION_INSTRUCTIONS)
        self.assertEqual(request.output_schema_version, OUTPUT_SCHEMA_VERSION)
        self.assertIn('only from the supplied context', request.instructions)
        self.assertIn('never invent handles', request.instructions)
        self.assertIn('declare that limitation', request.instructions)
        self.assertIn('Do not conceal conflicts', request.instructions)

    def test_request_builder_requires_sufficient_decision(self):
        context = ContextManifest()
        for status in (SufficiencyStatus.INSUFFICIENT, SufficiencyStatus.CONFLICT,
                       SufficiencyStatus.CLARIFICATION_REQUIRED, SufficiencyStatus.ERROR):
            decision = SufficiencyDecision(status, 'test', 'Synthetic decision', 0)
            with self.subTest(status=status), self.assertRaises(ValueError):
                GenerationRequestBuilder().build(self.request, context, decision, config=self.service.config)

    def test_prompt_overhead_is_explicit_and_manifest_is_not_truncated(self):
        config = replace(self.service.config, context_max_chars=40)
        result = replace(self.service, config=config).ask(self.request)
        self.assertEqual(result.state, AnswerState.ANSWER)
        supplied = self.provider.calls[0]
        trace = result.diagnostics['generation']
        self.assertLessEqual(trace['chunk_text_chars'], 40)
        self.assertGreater(trace['instructions_chars'], 40)
        self.assertEqual(trace['provider_formatting_overhead'], 'not_measured')
        self.assertFalse(trace['full_model_token_budget_known'])
        self.assertFalse(trace['context_changed_after_sufficiency'])
        self.assertIs(supplied.context_manifest, result.context_manifest)

    def test_generation_limits_are_forwarded_without_model_defaults(self):
        limits = GenerationLimits(max_output_tokens=17, timeout_seconds=3.5)
        result = replace(self.service, config=replace(self.service.config, generation_limits=limits)).ask(self.request)
        self.assertEqual(result.state, AnswerState.ANSWER)
        self.assertIs(self.provider.calls[0].limits, limits)

    def test_escalation_record_created_for_configured_citation_failure(self):
        repository = MemoryEscalationRepository()
        service = replace(self.service, provider=self.invalid_provider(), escalations=repository,
                          config=replace(self.service.config, citation_failure_action='escalate'))
        result = service.ask(self.request)
        self.assertEqual(result.state, AnswerState.ESCALATE_EXPERT)
        self.assertIsNone(result.answer)
        record = repository.get(result.escalation_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.request_id, self.request.request_id)
        self.assertEqual(record.question, self.request.question)
        self.assertEqual(record.access, self.request.access)
        self.assertEqual(record.reason, 'INVALID_CITATIONS')
        self.assertIs(record.context_manifest, result.context_manifest)
        self.assertEqual(record.retrieval_diagnostics, result.diagnostics['retrieval'])
        self.assertEqual(record.sufficiency_diagnostics, result.diagnostics['sufficiency'])
        self.assertIsNotNone(record.draft)
        self.assertIsNotNone(record.created_at.utcoffset())
        self.assertEqual(result.diagnostics['escalation']['status'], 'recorded')

    def test_custom_policy_can_escalate_before_generation(self):
        class EscalateConflict(RuntimeFinalAnswerPolicy):
            def decide(self, request, decision, *, context, draft, citation_validation, escalation_id, config):
                return AskResult(request.request_id, AnswerState.ESCALATE_EXPERT, 'CONFLICT_REVIEW',
                                 context_manifest=context, escalation_id=escalation_id)
        repository = MemoryEscalationRepository()
        service = replace(self.service, final_policy=EscalateConflict(), escalations=repository,
            sufficiency=replace(self.service.sufficiency, conflict_detector=ConflictHook()))
        result = service.ask(self.request)
        self.assertEqual(result.state, AnswerState.ESCALATE_EXPERT)
        self.assertEqual(self.provider.calls, [])
        record = repository.get(result.escalation_id)
        self.assertIsNone(record.draft)
        self.assertEqual(record.sufficiency_diagnostics['status'], 'conflict')

    def test_escalation_without_repository_returns_error(self):
        service = replace(self.service, provider=self.invalid_provider(),
                          config=replace(self.service.config, citation_failure_action='escalate'))
        result = service.ask(self.request)
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertEqual(result.reason_code, 'ESCALATION_UNAVAILABLE')
        self.assertIsNone(result.escalation_id)

    def test_escalation_storage_failure_does_not_claim_record_created(self):
        repository = MemoryEscalationRepository()
        service = replace(self.service, provider=self.invalid_provider(), escalations=repository,
                          config=replace(self.service.config, citation_failure_action='escalate'))
        with patch.object(repository, 'save', side_effect=RuntimeError('SECRET_STORAGE')):
            result = service.ask(self.request)
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertEqual(result.reason_code, 'ESCALATION_STORAGE_FAILED')
        self.assertIsNone(result.escalation_id)
        self.assertNotIn('SECRET', str(result))

    def test_escalation_repository_is_idempotent_and_rejects_changed_record(self):
        repository = MemoryEscalationRepository()
        service = replace(self.service, provider=self.invalid_provider(), escalations=repository,
                          config=replace(self.service.config, citation_failure_action='escalate'))
        result = service.ask(self.request)
        record = repository.get(result.escalation_id)
        repository.save(record)
        self.assertEqual(repository.get(result.escalation_id), record)
        with self.assertRaises(ValueError): repository.save(replace(record, reason='changed'))
        self.assertIsNone(repository.get('missing'))
        self.assertIsInstance(repository, EscalationRepository)

    def test_access_cannot_be_expanded_by_provider_metadata(self):
        foreign = self.fixture.ready(organization='org/B', scopes={'secret'})
        def metadata(request):
            self.assertTrue(all(x.evidence.chunk.document.organization_id == 'org/A' for x in request.context_manifest.entries))
            return replace(self.draft(request), provider_metadata={
                'access': {'organization_id': 'org/B', 'scopes': ['secret']}, 'final_state': 'escalate_expert'})
        result = replace(self.service, provider=FakeProviderA(metadata)).ask(self.request)
        self.assertEqual(result.state, AnswerState.ANSWER)
        self.assertNotIn(foreign.version, {x.evidence.chunk.version for x in result.context_manifest.entries})
        self.assertEqual(self.request.access.organization_id, 'org/A')
        self.assertNotIn('access', result.diagnostics)

    def test_foreign_organization_citation_is_rejected(self):
        foreign = self.fixture.ready(organization='org/B')
        chunk = self.fixture.repository.list_snapshot(foreign)[0]
        result = replace(self.service, provider=FakeProviderA(lambda request: self.draft(request, identity=chunk.identity))).ask(self.request)
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertIn('SOURCE_OUTSIDE_CONTEXT', result.citation_validation.errors)

    def test_retrieval_sufficiency_and_citation_diagnostics_are_preserved(self):
        result = self.ask()
        self.assertEqual(result.diagnostics['retrieval']['candidate_count'], 3)
        self.assertEqual(result.diagnostics['retrieval']['returned_evidence_count'], 2)
        self.assertEqual(result.diagnostics['retrieval']['visible_version_count'], 1)
        self.assertTrue(result.diagnostics['retrieval']['query_embedded'])
        self.assertEqual(result.diagnostics['sufficiency']['reason_code'], 'GATES_PASSED')
        self.assertEqual(result.diagnostics['sufficiency']['diagnostics']['evidence_count'], 2)
        self.assertEqual(result.diagnostics['citation']['diagnostics'], result.citation_validation.diagnostics)
        self.assertEqual(result.request_id, self.provider.calls[0].request_id)

    def test_retrieval_context_and_sufficiency_exceptions_do_not_call_provider(self):
        for component, method, reason in (
            (self.service.retrieval, 'retrieve', 'RETRIEVAL_FAILED'),
            (self.service.assembler, 'assemble', 'CONTEXT_ASSEMBLY_FAILED'),
        ):
            with self.subTest(reason=reason), patch.object(type(component), method, side_effect=RuntimeError('SECRET_STAGE')):
                result = self.ask()
                self.assertEqual(result.reason_code, reason)
                self.assertEqual(result.state, AnswerState.ERROR)
                self.assertEqual(self.provider.calls, [])
                self.assertNotIn('SECRET', str(result))
        with patch.object(SufficiencyPolicy, 'evaluate', side_effect=RuntimeError('SECRET_STAGE')):
            result = self.ask()
        self.assertEqual(result.reason_code, 'SUFFICIENCY_FAILED')
        self.assertEqual(self.provider.calls, [])

    def test_citation_validator_exception_returns_error(self):
        with patch.object(self.service.citation_validator, 'validate', side_effect=RuntimeError('SECRET_CITATION')):
            result = self.ask()
        self.assertEqual(result.reason_code, 'CITATION_VALIDATION_FAILED')
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertNotIn('SECRET', str(result))

    def test_final_policy_exception_or_wrong_request_id_returns_error(self):
        with patch.object(self.service.final_policy, 'decide', side_effect=RuntimeError('SECRET_POLICY')):
            result = self.ask()
        self.assertEqual(result.reason_code, 'FINAL_POLICY_FAILED')
        wrong = AskResult('another-request', AnswerState.ERROR, 'wrong')
        with patch.object(self.service.final_policy, 'decide', return_value=wrong):
            result = self.ask()
        self.assertEqual(result.reason_code, 'FINAL_POLICY_FAILED')
        self.assertEqual(result.request_id, self.request.request_id)

    def test_request_builder_failure_or_context_replacement_prevents_provider_call(self):
        with patch.object(self.service.request_builder, 'build', side_effect=RuntimeError('SECRET_BUILDER')):
            result = self.ask()
        self.assertEqual(result.reason_code, 'GENERATION_REQUEST_FAILED')
        self.assertEqual(self.provider.calls, [])
        build = self.service.request_builder.build
        def different(*args, **kwargs):
            generated = build(*args, **kwargs)
            return replace(generated, context_manifest=replace(generated.context_manifest))
        with patch.object(self.service.request_builder, 'build', side_effect=different):
            result = self.ask()
        self.assertEqual(result.reason_code, 'GENERATION_REQUEST_FAILED')
        self.assertEqual(self.provider.calls, [])

    def test_final_policy_cannot_bypass_early_gate_generation_failure_or_invalid_citations(self):
        class UnsafePolicy:
            def decide(self, request, decision, **kwargs):
                return AskResult(request.request_id, AnswerState.ANSWER, 'unsafe', answer='Unvalidated answer.')
        unsafe = replace(self.service, final_policy=UnsafePolicy())
        restricted = replace(self.request, retrieval_filters=replace(self.request.retrieval_filters,
            allowed_document_ids=frozenset()))
        result = unsafe.ask(restricted)
        self.assertEqual(result.reason_code, 'INVALID_FINAL_POLICY_RESULT')
        self.assertEqual(result.state, AnswerState.ERROR)
        self.assertEqual(self.provider.calls, [])
        def fail(request): raise RuntimeError('provider failure')
        for provider in (FakeProviderA(fail), self.invalid_provider()):
            with self.subTest(provider=provider):
                result = replace(unsafe, provider=provider).ask(self.request)
                self.assertEqual(result.state, AnswerState.ERROR)
                self.assertEqual(result.reason_code, 'INVALID_FINAL_POLICY_RESULT')
                self.assertIsNone(result.answer)

    def test_final_policy_rejects_pass_without_resolved_references(self):
        result = self.ask()
        decision = self.service.sufficiency.evaluate(self.request.question, result.context_manifest, config=self.service.config)
        final = self.service.final_policy.decide(self.request, decision, context=result.context_manifest,
            draft=self.draft(self.provider.calls[0]), citation_validation=CitationValidationResult(StructuralValidity.PASS),
            escalation_id=None, config=self.service.config)
        self.assertEqual(final.state, AnswerState.ERROR)
        self.assertEqual(final.reason_code, 'INVALID_CITATIONS')

    def test_citation_validator_handles_malformed_inputs_with_structural_fail(self):
        validator = StructuralCitationValidator()
        result = validator.validate('unstructured', ContextManifest())
        self.assertEqual(result.errors, ('INVALID_DRAFT',))
        self.assertEqual(result.semantic_support, SemanticSupport.NOT_CHECKED)
        result = validator.validate(GenerationDraft('draft'), object())
        self.assertEqual(result.errors, ('INVALID_CONTEXT',))

    def test_output_validator_rejects_raw_text_and_preserves_valid_structured_draft(self):
        with self.assertRaises(GenerationOutputError): validate_generation_draft('raw text')
        self.ask()
        request = self.provider.calls[0]
        draft = self.draft(request)
        self.assertIsNone(validate_generation_draft(draft))

    def test_configuration_rejects_invalid_escalation_action_and_generation_limits(self):
        for action in ('automatic', '', None, True):
            with self.subTest(action=action), self.assertRaises(ValueError):
                replace(self.service.config, citation_failure_action=action)
        with self.assertRaises(ValueError): replace(self.service.config, generation_limits={})

    def test_final_policy_and_validator_conform_to_protocols(self):
        self.assertIsInstance(self.service.final_policy, FinalAnswerPolicy)
        self.assertIsInstance(self.service.citation_validator, CitationValidator)

    def test_runtime_answering_has_no_benchmark_labels_or_fixed_model_domains(self):
        forbidden = {'expected_answer', 'answer_hint', 'probes', 'expected_behavior', 'expected_refusal',
                     'forbidden_claims', 'gold', 'query_id'}
        paths = ('generation', 'citations', 'policy', 'ask')
        for name in paths:
            tree = ast.parse((ROOT / ('src/knowledge_base/runtime/' + name + '.py')).read_text(encoding='utf8'))
            literals = {x.value for x in ast.walk(tree) if isinstance(x, ast.Constant) and isinstance(x.value, str)}
            identifiers = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name)}
            self.assertFalse(forbidden & (literals | identifiers))
            self.assertFalse(literals & {'MAIN119', 'TEAM HOLDOUT', 'MOST', 'company', 'laboratory', 'openai', 'ollama', 'Qwen'})
        self.assertFalse(forbidden & set(inspect.signature(AskService.ask).parameters))

    def test_answering_executes_without_importing_benchmarks_models_or_external_clients(self):
        code = r'''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('knowledge_base.benchmark', 'knowledge_base.team', 'benchmarks', 'scripts',
                                'knowledge_base.bge_reranker', 'knowledge_base.candidate_retrieval',
                                'torch', 'transformers', 'openai', 'ollama', 'requests', 'httpx')):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Reject())
from test_runtime_answering import RuntimeAnsweringTests
case = RuntimeAnsweringTests('test_two_fake_providers_use_same_contract_and_pipeline')
case.setUp()
try:
    case.test_two_fake_providers_use_same_contract_and_pipeline()
finally:
    case.doCleanups()
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, 'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(ROOT / 'tests'))),
                 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
