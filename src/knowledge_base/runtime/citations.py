"""Structural citation checks against exact model context; no entailment judge."""
from dataclasses import replace

from .context import validate_evidence
from .generation import validate_generation_draft
from .models import (
    CitationValidationResult, ContextEntry, ContextManifest, GenerationDraft,
    SemanticSupport, StructuralValidity,
)


class StructuralCitationValidator:
    """Require at least one reference; deduplicate identical references in order.

    A handle must resolve to the exact document/version/snapshot/chunk tuple
    supplied in this manifest. PASS proves reference structure only. It neither
    checks factual correctness nor verifies that every claim has a citation.
    Semantic support always remains NOT_CHECKED.
    """
    def validate(self, draft: GenerationDraft, context: ContextManifest) -> CitationValidationResult:
        errors, resolved = [], []
        duplicates = 0
        count = 0

        def finish():
            return CitationValidationResult(
                StructuralValidity.FAIL if errors else StructuralValidity.PASS,
                semantic_support=SemanticSupport.NOT_CHECKED,
                resolved_citations=tuple(resolved), errors=tuple(dict.fromkeys(errors)),
                diagnostics={'citation_count': count, 'resolved_citation_count': len(resolved),
                             'duplicate_citation_count': duplicates, 'semantic_support': 'not_checked'},
            )

        try:
            if not isinstance(context, ContextManifest):
                raise ValueError
            replace(context)
            for entry in context.entries:
                if not isinstance(entry, ContextEntry):
                    raise ValueError
                replace(entry)
                validate_evidence(entry.evidence)
        except Exception:
            errors.append('INVALID_CONTEXT')
            return finish()
        try:
            validate_generation_draft(draft)
        except Exception:
            errors.append('INVALID_DRAFT')
            return finish()
        count = len(draft.citations)
        if not draft.citations:
            errors.append('NO_CITATIONS')
            return finish()
        handles = {entry.source_handle: entry.evidence.chunk.identity for entry in context.entries}
        identities = set(handles.values())
        seen = set()
        for reference in draft.citations:
            key = (reference.source_handle, reference.chunk_identity)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            expected = handles.get(reference.source_handle)
            if expected is None:
                errors.append('UNKNOWN_SOURCE_HANDLE')
            elif expected != reference.chunk_identity:
                errors.append('CHUNK_IDENTITY_MISMATCH')
            if reference.chunk_identity not in identities:
                errors.append('SOURCE_OUTSIDE_CONTEXT')
            if expected == reference.chunk_identity:
                resolved.append(reference)
        return finish()
