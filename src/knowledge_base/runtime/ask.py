"""Ask orchestration only; providers cannot choose access or final answer state."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from .config import RuntimeConfig
from .generation import (GenerationOutputError, GenerationRequestBuilder,
                         is_limitation_only_draft, validate_generation_draft)
from .models import (
    AccessContext, AnswerState, AskRequest, AskResult, ContextManifest, ExpertEscalation,
    StructuralValidity,
    RetrievalFilters, SufficiencyDecision, SufficiencyStatus,
)
from .protocols import (
    CitationValidator, ContextAssembler, EscalationRepository, FinalAnswerPolicy,
    GenerationProvider, SufficiencyGate,
)
from .retrieval import RetrievalResult


class _Retriever(Protocol):
    def retrieve(
        self, question: str, access: AccessContext, *, filters: RetrievalFilters,
        config: RuntimeConfig,
    ) -> RetrievalResult: ...


@dataclass(frozen=True)
class AskService:
    """One provider call only after SUFFICIENT, using the unchanged final manifest.

    The request builder adds generic instructions and reports formatting overhead
    without truncation. Failures return safe codes and preserve stage diagnostics.
    Escalations, when chosen by final policy, are records only, never messages.
    """
    retrieval: _Retriever
    assembler: ContextAssembler
    sufficiency: SufficiencyGate
    provider: GenerationProvider
    citation_validator: CitationValidator
    final_policy: FinalAnswerPolicy
    config: RuntimeConfig = RuntimeConfig()
    request_builder: GenerationRequestBuilder = field(default_factory=GenerationRequestBuilder)
    escalations: EscalationRepository | None = None

    def ask(self, request: AskRequest) -> AskResult:
        diagnostics = {
            'retrieval': {'status': 'not_run'}, 'context': {'status': 'not_run'},
            'sufficiency': {'status': 'not_run'}, 'generation': {'called': False, 'status': 'not_called'},
            'citation': {'status': 'not_run'},
        }
        context = ContextManifest()
        phase = 'INVALID_ASK_REQUEST'
        try:
            replace(request, access=replace(request.access), retrieval_filters=replace(request.retrieval_filters))
            replace(self.config)
            phase = 'RETRIEVAL_FAILED'
            retrieved = self.retrieval.retrieve(request.question, request.access,
                filters=request.retrieval_filters, config=self.config)
            if not isinstance(retrieved, RetrievalResult):
                raise ValueError
            diagnostics['retrieval'] = {
                'status': 'completed', 'visible_version_count': retrieved.visible_version_count,
                'candidate_count': retrieved.candidate_count, 'query_embedded': retrieved.query_embedded,
                'empty_reason': retrieved.empty_reason, 'returned_evidence_count': len(retrieved.evidence),
            }
            phase = 'CONTEXT_ASSEMBLY_FAILED'
            assembled = self.assembler.assemble(retrieved.evidence, max_chars=self.config.context_max_chars)
            if not isinstance(assembled, ContextManifest):
                raise ValueError
            context = assembled
            diagnostics['context'] = {'status': 'completed', 'diagnostics': context.diagnostics}
            phase = 'SUFFICIENCY_FAILED'
            decision = self.sufficiency.evaluate(request.question, context, config=self.config)
            if not isinstance(decision, SufficiencyDecision):
                raise ValueError
            replace(decision)
            diagnostics['sufficiency'] = {'status': decision.status.value, 'reason_code': decision.reason_code,
                'evidence_count': decision.evidence_count, 'diagnostics': decision.diagnostics}
        except Exception:
            return self._error(request, phase, context, diagnostics)
        if decision.status != SufficiencyStatus.SUFFICIENT:
            return self._finish(request, decision, context, None, None, diagnostics)
        try:
            generation_request = self.request_builder.build(request, context, decision, config=self.config)
            if generation_request.context_manifest is not context:
                raise ValueError
            diagnostics['generation'].update(self.request_builder.diagnostics(generation_request))
        except Exception:
            diagnostics['generation']['status'] = 'request_build_failed'
            return self._finish(request, decision, context, None, None, diagnostics, 'GENERATION_REQUEST_FAILED')
        diagnostics['generation'].update({'called': True, 'status': 'started'})
        try:
            draft = self.provider.generate(generation_request)
        except Exception:
            diagnostics['generation']['status'] = 'failed'
            return self._finish(request, decision, context, None, None, diagnostics, 'GENERATION_FAILED')
        try:
            validate_generation_draft(draft)
        except GenerationOutputError as error:
            diagnostics['generation']['status'] = 'invalid_output'
            return self._finish(request, decision, context, None, None, diagnostics, str(error))
        diagnostics['generation']['status'] = 'completed'
        if is_limitation_only_draft(draft):
            diagnostics['citation'] = {'status': 'not_applicable', 'reason_code': 'NO_FACTUAL_ANSWER',
                                       'semantic_support': 'not_checked'}
            return self._finish(request, decision, context, draft, None, diagnostics)
        try:
            validation = self.citation_validator.validate(draft, context)
            replace(validation)
            diagnostics['citation'] = {
                'status': 'completed', 'structural_validity': validation.structural_validity.value,
                'semantic_support': validation.semantic_support.value, 'errors': validation.errors,
                'diagnostics': validation.diagnostics,
            }
        except Exception:
            diagnostics['citation']['status'] = 'failed'
            return self._finish(request, decision, context, draft, None, diagnostics, 'CITATION_VALIDATION_FAILED')
        return self._finish(request, decision, context, draft, validation, diagnostics)

    @staticmethod
    def _error(request, code, context, diagnostics):
        return AskResult(request.request_id, AnswerState.ERROR, code,
                         context_manifest=context, diagnostics=diagnostics)

    def _finish(self, request, decision, context, draft, validation, diagnostics, failure_code=None):
        escalation_id = str(uuid4()) if self.escalations is not None else None
        try:
            result = self.final_policy.decide(request, decision, context=context, draft=draft,
                citation_validation=validation, escalation_id=escalation_id, config=self.config)
            if not isinstance(result, AskResult) or result.request_id != request.request_id:
                raise ValueError
            replace(result)
        except Exception:
            return self._error(request, 'FINAL_POLICY_FAILED', context, diagnostics)
        if (result.state in (AnswerState.ANSWER, AnswerState.PARTIAL_ANSWER)
                and (decision.status != SufficiencyStatus.SUFFICIENT or draft is None or validation is None
                     or validation.structural_validity != StructuralValidity.PASS
                     or not validation.resolved_citations
                     or (result.state == AnswerState.PARTIAL_ANSWER and not self.config.partial_answers_enabled))):
            return self._error(request, 'INVALID_FINAL_POLICY_RESULT', context, diagnostics)
        if (failure_code is not None and result.state == AnswerState.ERROR
                and result.reason_code in ('GENERATION_FAILED', 'CITATION_VALIDATION_MISSING')):
            result = replace(result, reason_code=failure_code)
        if result.state == AnswerState.ESCALATE_EXPERT:
            if self.escalations is None:
                return self._error(request, 'ESCALATION_UNAVAILABLE', context, diagnostics)
            try:
                self.escalations.save(ExpertEscalation(
                    result.escalation_id, request.request_id, datetime.now(timezone.utc), request.question,
                    request.access, context, result.reason_code, missing_information=decision.missing_information,
                    retrieval_diagnostics=diagnostics['retrieval'], draft=draft,
                    sufficiency_diagnostics=diagnostics['sufficiency'],
                ))
                diagnostics['escalation'] = {'status': 'recorded'}
            except Exception:
                return self._error(request, 'ESCALATION_STORAGE_FAILED', context, diagnostics)
        return replace(result, context_manifest=context, citation_validation=validation,
                       diagnostics={**result.diagnostics, **diagnostics})
