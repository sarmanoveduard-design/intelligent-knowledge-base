"""Deterministic whole-chunk assembly with an explicit text character budget."""
from __future__ import annotations

from dataclasses import replace

from .models import (
    ChunkIdentity, ContextEntry, ContextManifest, DocumentIdentity,
    DocumentVersionIdentity, EvidenceChunk, RuntimeEvidence, SourceCoordinates,
    _positive_int,
)


class ContextAssemblyError(RuntimeError):
    """Stable diagnostic code, without source text or underlying exception details."""


def validate_evidence(evidence: RuntimeEvidence) -> None:
    """Revalidate the complete immutable record at a runtime integrity boundary."""
    if not isinstance(evidence, RuntimeEvidence) or not isinstance(evidence.chunk, EvidenceChunk):
        raise ValueError('invalid_evidence_record')
    chunk = evidence.chunk
    for value, expected in ((chunk.identity, ChunkIdentity), (chunk.document, DocumentIdentity),
                            (chunk.version, DocumentVersionIdentity), (chunk.coordinates, SourceCoordinates)):
        if not isinstance(value, expected):
            raise ValueError('invalid_source_record')
    # Nested replacements run existing constructors, including finite scores,
    # matching identities, source coordinates and finite JSON-like metadata.
    validated = replace(chunk, identity=replace(chunk.identity), document=replace(chunk.document),
                        version=replace(chunk.version), coordinates=replace(chunk.coordinates))
    replace(evidence, chunk=validated)


class RankedContextAssembler:
    """First representation wins per full ChunkIdentity; ranking is preserved.

    Walk top to bottom, skip whole chunks that do not fit, and continue with
    lower ranked chunks. Handles are allocated only after selection. The budget
    counts Unicode characters in exact chunk texts, without truncation. It does
    not include future provider formatting/prompt overhead or estimate tokens.
    Pass RuntimeConfig.context_max_chars as max_chars.
    """
    def assemble(
        self, candidates: tuple[RuntimeEvidence, ...], *, max_chars: int,
    ) -> ContextManifest:
        try:
            _positive_int(max_chars, 'max_chars')
        except ValueError:
            raise ContextAssemblyError('INVALID_CONTEXT_BUDGET') from None
        try:
            candidates = tuple(candidates)
            selected = []
            seen = set()
            skipped = []
            used = 0
            for position, evidence in enumerate(candidates):
                validate_evidence(evidence)
                identity = evidence.chunk.identity
                chars = len(evidence.chunk.text)
                if identity in seen:
                    reason = 'DUPLICATE_CHUNK'
                else:
                    seen.add(identity)
                    if chars > max_chars:
                        reason = 'OVERSIZED_CHUNK'
                    elif used + chars > max_chars:
                        reason = 'CONTEXT_BUDGET_EXCEEDED'
                    else:
                        selected.append(evidence)
                        used += chars
                        continue
                skipped.append({'input_position': position, 'reason_code': reason, 'text_chars': chars})
            return ContextManifest(
                tuple(ContextEntry(f'S{position}', evidence) for position, evidence in enumerate(selected, start=1)),
                diagnostics={'input_evidence_count': len(candidates), 'selected_evidence_count': len(selected),
                             'max_chars': max_chars, 'used_chars': used, 'budget_unit': 'chunk_text_characters',
                             'oversized_behavior': 'skip_whole_chunk', 'skipped': tuple(skipped)},
            )
        except Exception:
            raise ContextAssemblyError('INVALID_CONTEXT_EVIDENCE') from None
