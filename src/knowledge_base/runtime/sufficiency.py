"""Technical evidence gates, not semantic answerability or calibrated confidence."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .config import RuntimeConfig
from .context import validate_evidence
from .models import (
    ClarificationAssessment, ConflictAssessment, ConflictStatus, ContextEntry, ContextManifest,
    DocumentVersionStatus, ProcessingState, SufficiencyDecision, SufficiencyStatus, _nonempty,
)
from .protocols import ClarificationPolicy, ConflictDetector, IndexingRepository


_MESSAGES = {
    'CONTEXT_BUDGET_EXCEEDED': 'Selected chunk text exceeds the configured context budget.',
    'NO_EVIDENCE': 'No evidence was selected for context.',
    'BELOW_MINIMUM_EVIDENCE': 'The selected evidence count is below the configured minimum.',
    'RETRIEVAL_SCORE_TOO_LOW': 'At least one selected retrieval score is below its configured threshold.',
    'RERANKER_SCORE_TOO_LOW': 'At least one selected reranker score is below its configured threshold.',
    'RERANKER_SCORE_MISSING': 'A configured reranker gate has no score for a selected entry.',
    'THRESHOLD_UNCONFIGURED': 'The configured policy requires both score thresholds.',
    'SCORE_SPACE_MISMATCH': 'A threshold cannot be compared with a different score type or scorer identity.',
    'INVALID_PROVENANCE': 'Context provenance does not match the authoritative source registry.',
    'PROVENANCE_CHECK_FAILED': 'The authoritative provenance check could not be completed.',
    'INVALID_CONFIGURATION': 'Sufficiency configuration is invalid.',
    'INVALID_QUESTION': 'A nonempty question is required.',
    'HOOK_FAILED': 'A configured assessment hook failed or returned an invalid assessment.',
    'CLARIFICATION_REQUIRED': 'A configured hook reports concrete missing parameters.',
    'CONFLICT_DETECTED': 'A configured hook asserts a conflict in the selected context.',
    'GATES_PASSED': 'Technical and configured evidence gates passed; semantic answerability is not established.',
}


@dataclass(frozen=True)
class SufficiencyPolicy:
    """SUFFICIENT permits a future generation attempt; it does not prove an answer.

    Validate exact provenance against the registry first. Every selected entry
    must meet every configured threshold, with inclusive cutoffs and matching
    score type AND scorer identity. Unset thresholds are reported as skipped;
    'require' configuration blocks until both thresholds are configured.

    Precedence: integrity/budget, empty/minimum count, retrieval/reranker gates,
    conflict hook, clarification hook. Hooks receive only the final manifest.
    No regex, benchmark labels, inferred missing parameters, model calls or
    generation. Access authorization remains the retrieval boundary; the
    evaluate protocol has no AccessContext and grants no access itself.
    """
    repository: IndexingRepository
    conflict_detector: ConflictDetector | None = None
    clarification_policy: ClarificationPolicy | None = None

    def evaluate(
        self, question: str, context: ContextManifest, *, config: RuntimeConfig,
    ) -> SufficiencyDecision:
        diagnostics = {
            'evidence_count': len(context.entries) if isinstance(context, ContextManifest)
                and isinstance(context.entries, tuple) else 0,
            'semantic_answerability': 'not_checked',
            'applied_gates': [], 'skipped_gates': [],
            'conflict_hook_result': 'unknown' if self.conflict_detector is None else 'not_run',
            'clarification_hook_result': 'not_configured' if self.clarification_policy is None else 'not_run',
            'missing_parameters': (),
        }
        gates = ('provenance', 'context_budget', 'minimum_evidence', 'retrieval_threshold', 'reranker_threshold',
                 'conflict_hook', 'clarification_hook')

        def decide(status, code, missing=()):
            diagnostics['skipped_gates'] = tuple(g for g in gates if g not in diagnostics['applied_gates'])
            return SufficiencyDecision(status, code, _MESSAGES[code], diagnostics['evidence_count'],
                                       diagnostics, missing_information=missing)

        try:
            if not isinstance(config, RuntimeConfig):
                raise ValueError
            replace(config)
            for threshold in (config.dense_threshold, config.reranker_threshold):
                if threshold is not None:
                    replace(threshold)
        except Exception:
            return decide(SufficiencyStatus.ERROR, 'INVALID_CONFIGURATION')
        diagnostics['minimum_evidence_count'] = config.minimum_evidence_count
        diagnostics['unconfigured_threshold_policy'] = config.unconfigured_threshold_policy
        for name, threshold in (('retrieval_threshold', config.dense_threshold),
                                ('reranker_threshold', config.reranker_threshold)):
            diagnostics[name] = {
                'status': 'skipped_unconfigured' if threshold is None else 'not_reached',
                'threshold_value': threshold.value if threshold else None,
                'score_type': threshold.score_type if threshold else None,
                'scorer_identity': threshold.scorer_identity if threshold else None,
                'observed_values': (), 'observed_score_types': (), 'observed_scorer_identities': (),
            }
        try:
            _nonempty(question, 'question')
        except ValueError:
            return decide(SufficiencyStatus.ERROR, 'INVALID_QUESTION')
        diagnostics['applied_gates'].append('provenance')
        try:
            if not isinstance(context, ContextManifest):
                raise ValueError
            replace(context)
            diagnostics['evidence_count'] = len(context.entries)
            for entry in context.entries:
                if not isinstance(entry, ContextEntry):
                    raise ValueError
                replace(entry)
                validate_evidence(entry.evidence)
        except Exception:
            return decide(SufficiencyStatus.ERROR, 'INVALID_PROVENANCE')
        try:
            for entry in context.entries:
                chunk = entry.evidence.chunk
                version = self.repository.get_version(chunk.version)
                document = self.repository.get_document(chunk.document)
                if (version is None or document is None or version.identity != chunk.version
                        or document.identity != chunk.document or version.status != DocumentVersionStatus.ACTIVE
                        or version.processing_state != ProcessingState.INDEXED or not version.approved
                        or self.repository.get_current_snapshot(chunk.version) != chunk.identity.snapshot
                        or self.repository.get_chunk(chunk.identity) != chunk):
                    return decide(SufficiencyStatus.ERROR, 'INVALID_PROVENANCE')
        except Exception:
            return decide(SufficiencyStatus.ERROR, 'PROVENANCE_CHECK_FAILED')
        evidence = tuple(entry.evidence for entry in context.entries)
        diagnostics['context_text_chars'] = sum(len(item.chunk.text) for item in evidence)
        diagnostics['context_max_chars'] = config.context_max_chars
        diagnostics['applied_gates'].append('context_budget')
        if diagnostics['context_text_chars'] > config.context_max_chars:
            return decide(SufficiencyStatus.ERROR, 'CONTEXT_BUDGET_EXCEEDED')
        for name, prefix in (('retrieval_threshold', 'retrieval'), ('reranker_threshold', 'reranker')):
            diagnostics[name].update({
                'observed_values': tuple(getattr(item, prefix + '_score') for item in evidence),
                'observed_score_types': tuple(getattr(item, prefix + '_score_type') for item in evidence),
                'observed_scorer_identities': tuple(getattr(item, prefix + '_scorer_identity') for item in evidence),
            })
        if not evidence:
            return decide(SufficiencyStatus.INSUFFICIENT, 'NO_EVIDENCE')
        diagnostics['applied_gates'].append('minimum_evidence')
        if len(evidence) < config.minimum_evidence_count:
            return decide(SufficiencyStatus.INSUFFICIENT, 'BELOW_MINIMUM_EVIDENCE')
        for name, threshold, low_code in (
            ('retrieval_threshold', config.dense_threshold, 'RETRIEVAL_SCORE_TOO_LOW'),
            ('reranker_threshold', config.reranker_threshold, 'RERANKER_SCORE_TOO_LOW'),
        ):
            gate = diagnostics[name]
            if threshold is None:
                if config.unconfigured_threshold_policy == 'require':
                    gate['status'] = 'blocked_unconfigured'
                    return decide(SufficiencyStatus.INSUFFICIENT, 'THRESHOLD_UNCONFIGURED')
                continue
            diagnostics['applied_gates'].append(name)
            if any(value is None for value in gate['observed_values']):
                gate['status'] = 'missing_score'
                return decide(SufficiencyStatus.INSUFFICIENT, 'RERANKER_SCORE_MISSING')
            if (any(value != threshold.score_type for value in gate['observed_score_types'])
                    or any(value != threshold.scorer_identity for value in gate['observed_scorer_identities'])):
                gate['status'] = 'score_space_mismatch'
                return decide(SufficiencyStatus.ERROR, 'SCORE_SPACE_MISMATCH')
            if any(value < threshold.value for value in gate['observed_values']):
                gate['status'] = 'failed'
                return decide(SufficiencyStatus.INSUFFICIENT, low_code)
            gate['status'] = 'passed'
        if self.conflict_detector is not None:
            diagnostics['applied_gates'].append('conflict_hook')
            try:
                conflict = self.conflict_detector.detect(question, context)
                if not isinstance(conflict, ConflictAssessment):
                    raise ValueError
                replace(conflict)
                diagnostics['conflict_hook_result'] = conflict.status.value
            except Exception:
                diagnostics['conflict_hook_result'] = 'error'
                return decide(SufficiencyStatus.ERROR, 'HOOK_FAILED')
            if conflict.status == ConflictStatus.CONFLICT:
                return decide(SufficiencyStatus.CONFLICT, 'CONFLICT_DETECTED')
        if self.clarification_policy is not None:
            diagnostics['applied_gates'].append('clarification_hook')
            try:
                clarification = self.clarification_policy.assess(question, context)
                if not isinstance(clarification, ClarificationAssessment):
                    raise ValueError
                replace(clarification)
                diagnostics['missing_parameters'] = clarification.missing_parameters
                diagnostics['clarification_hook_result'] = 'required' if clarification.missing_parameters else 'clear'
            except Exception:
                diagnostics['clarification_hook_result'] = 'error'
                return decide(SufficiencyStatus.ERROR, 'HOOK_FAILED')
            if clarification.missing_parameters:
                return decide(SufficiencyStatus.CLARIFICATION_REQUIRED, 'CLARIFICATION_REQUIRED',
                              clarification.missing_parameters)
        return decide(SufficiencyStatus.SUFFICIENT, 'GATES_PASSED')
