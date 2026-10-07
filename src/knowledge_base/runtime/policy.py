"""Final states are application decisions, independent of provider/model names."""
from dataclasses import replace
from threading import RLock

from .config import RuntimeConfig
from .generation import GenerationOutputError, is_limitation_only_draft, validate_generation_draft
from .models import (
    AnswerState, AskRequest, AskResult, CitationValidationResult, ContextManifest,
    ExpertEscalation, GenerationDraft, StructuralValidity, SufficiencyDecision, SufficiencyStatus,
)


class RuntimeFinalAnswerPolicy:
    """Structural PASS allows a technical answer attempt, not a factual guarantee.

    A limitation-only draft is a late refusal, without any factual answer.
    A nonempty draft declaring limitations is a partial answer: disabled by default.
    Citation failures become ERROR by default or an explicit configured escalation.
    This policy makes no provider calls and sends no external notifications.
    """
    def decide(
        self, request: AskRequest, decision: SufficiencyDecision, *,
        context: ContextManifest, draft: GenerationDraft | None,
        citation_validation: CitationValidationResult | None,
        escalation_id: str | None, config: RuntimeConfig,
    ) -> AskResult:
        def result(state, reason, **kwargs):
            return AskResult(request.request_id, state, reason, context_manifest=context,
                             citation_validation=citation_validation, **kwargs)

        early = {
            SufficiencyStatus.INSUFFICIENT: AnswerState.REFUSE_INSUFFICIENT_CONTEXT,
            SufficiencyStatus.CLARIFICATION_REQUIRED: AnswerState.CLARIFY,
            SufficiencyStatus.CONFLICT: AnswerState.CONFLICT,
            SufficiencyStatus.ERROR: AnswerState.ERROR,
        }
        if decision.status in early:
            return result(early[decision.status], decision.reason_code,
                          missing_information=decision.missing_information)
        if decision.status != SufficiencyStatus.SUFFICIENT:
            return result(AnswerState.ERROR, 'INVALID_SUFFICIENCY_DECISION')
        if draft is None:
            return result(AnswerState.ERROR, 'GENERATION_FAILED')
        try:
            validate_generation_draft(draft)
        except GenerationOutputError as error:
            return result(AnswerState.ERROR, str(error))
        if is_limitation_only_draft(draft):
            return result(AnswerState.REFUSE_INSUFFICIENT_CONTEXT,
                          'GENERATION_REPORTED_INSUFFICIENT_CONTEXT',
                          declared_limitations=draft.declared_limitations)
        if citation_validation is None:
            return result(AnswerState.ERROR, 'CITATION_VALIDATION_MISSING')
        try:
            if not isinstance(citation_validation, CitationValidationResult):
                raise ValueError
            replace(citation_validation)
        except Exception:
            return result(AnswerState.ERROR, "INVALID_CITATION_RESULT")
        if (citation_validation.structural_validity != StructuralValidity.PASS
                or not citation_validation.resolved_citations):
            if config.citation_failure_action == 'escalate':
                if escalation_id is None:
                    return result(AnswerState.ERROR, 'ESCALATION_UNAVAILABLE')
                return result(AnswerState.ESCALATE_EXPERT, 'INVALID_CITATIONS', escalation_id=escalation_id)
            return result(AnswerState.ERROR, 'INVALID_CITATIONS')
        if draft.declared_limitations:
            if not config.partial_answers_enabled:
                return result(AnswerState.REFUSE_INSUFFICIENT_CONTEXT, 'PARTIAL_ANSWER_DISABLED',
                              declared_limitations=draft.declared_limitations)
            return result(AnswerState.PARTIAL_ANSWER, 'PARTIAL_ANSWER_ALLOWED', answer=draft.answer,
                          declared_limitations=draft.declared_limitations)
        return result(AnswerState.ANSWER, 'ANSWER_READY', answer=draft.answer)


class MemoryEscalationRepository:
    """Explicitly created local records only; restart loses them. No delivery."""
    def __init__(self):
        self._records: dict[str, ExpertEscalation] = {}
        self._lock = RLock()

    def save(self, escalation: ExpertEscalation) -> None:
        if not isinstance(escalation, ExpertEscalation):
            raise ValueError("invalid_escalation_record")
        replace(escalation)
        with self._lock:
            previous = self._records.get(escalation.escalation_id)
            if previous is not None and previous != escalation:
                raise ValueError('escalation_identity_conflict')
            self._records[escalation.escalation_id] = escalation

    def get(self, escalation_id: str) -> ExpertEscalation | None:
        with self._lock:
            return self._records.get(escalation_id)
