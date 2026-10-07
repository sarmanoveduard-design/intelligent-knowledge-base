"""Runtime contracts only. Adapters must enforce the documented semantics.

Protocol conformance describes shape, not authorization or behavioral proofs.
No protocol here imports an execution provider, benchmark or storage backend.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .config import RuntimeConfig
from .models import (
    AccessContext, AskRequest, AskResult, CitationValidationResult,
    ChunkIdentity, ContextManifest, DocumentIdentity, DocumentVersionIdentity, EvidenceChunk,
    ExpertEscalation, GenerationDraft, GenerationRequest, IndexIdentity,
    IndexedEvidence, IndexManifest, ProcessingSnapshotIdentity,
    RetrievalFilters, RuntimeDocument, RuntimeDocumentVersion, RuntimeEvidence,
    SufficiencyDecision,
)


@runtime_checkable
class DocumentRepository(Protocol):
    def save_document(self, document: RuntimeDocument) -> None:
        """Create/update a logical document and trusted access metadata."""
        ...

    def get_document(self, identity: DocumentIdentity) -> RuntimeDocument | None: ...

    def save_version(self, version: RuntimeDocumentVersion) -> None:
        """Create/update processing and approval metadata.

        Must preserve lifecycle invariants; publication goes through activate.
        """
        ...

    def get_version(self, identity: DocumentVersionIdentity) -> RuntimeDocumentVersion | None: ...

    def list_visible_versions(
        self, access: AccessContext, *, filters: RetrievalFilters,
    ) -> tuple[RuntimeDocumentVersion, ...]:
        """Only authorized, approved, active and index-ready versions.

        Missing access metadata fails closed. Filters can only narrow grants.
        """
        ...

    def activate_version(
        self, identity: DocumentVersionIdentity, *, expected_current_version_id: str | None,
    ) -> None:
        """Atomically approve/publish a prepared version and supersede the old one.

        Compare current identity with expected_current_version_id; None means
        expect no current version. Reject archived/superseded versions or concurrent edits.
        Prepared CHUNKED versions can be published before indexing; retrieval
        visibility still requires INDEXED. Activation is an explicit approval.
        """
        ...

    def archive_version(self, identity: DocumentVersionIdentity) -> None:
        """Immediately exclude the version from visible results."""
        ...


@runtime_checkable
class ChunkRepository(Protocol):
    def save_snapshot(
        self, snapshot: ProcessingSnapshotIdentity, chunks: Sequence[EvidenceChunk],
    ) -> None:
        """Save an immutable snapshot atomically; identical retries are idempotent.

        Reject mismatched identities, duplicate IDs and changed existing content.
        """
        ...

    def list_snapshot(self, snapshot: ProcessingSnapshotIdentity) -> tuple[EvidenceChunk, ...]: ...

    def get_chunk(self, identity: ChunkIdentity) -> EvidenceChunk | None: ...

    def delete_snapshot(self, snapshot: ProcessingSnapshotIdentity) -> None: ...


@runtime_checkable
class VectorIndex(Protocol):
    @property
    def identity(self) -> IndexIdentity: ...

    def upsert(self, manifest: IndexManifest, chunks: Sequence[IndexedEvidence]) -> None:
        """Idempotent complete-snapshot publication.

        Validate embedding/preprocessing compatibility, snapshot identities,
        dimensions, unique IDs and manifest count before publishing readiness.
        Failed writes must not expose a partly ready snapshot.
        """
        ...

    def search(
        self, query_vector: Sequence[float], *, access: AccessContext,
        filters: RetrievalFilters, top_k: int, expected_identity: IndexIdentity,
    ) -> tuple[RuntimeEvidence, ...]:
        """Apply trusted visibility/access restrictions BEFORE top-k.

        Search only ready, active, approved, authorized snapshots. An adapter
        unable to enforce these restrictions must reject the search. Validate
        expected identity, dimensions and finite values; never expand filters.
        Visibility must be supplied/rechecked against an authoritative registry.
        """
        ...

    def get_manifest(self, snapshot: ProcessingSnapshotIdentity) -> IndexManifest | None: ...

    def delete_snapshot(self, snapshot: ProcessingSnapshotIdentity) -> None: ...

    def deactivate_version(self, identity: DocumentVersionIdentity) -> None:
        """Make all snapshots of the version immediately unsearchable."""
        ...


@runtime_checkable
class RuntimeReranker(Protocol):
    """Neutral boundary until the existing benchmark-coupled contract is adapted.

    Preserve the complete candidate identity set, source texts and retrieval
    scores; add separate finite reranker scores and their type. Never inject or
    drop chunks. Scorer identity identifies the configured scoring space.
    """
    @property
    def name(self) -> str: ...

    @property
    def score_type(self) -> str: ...

    @property
    def scorer_identity(self) -> str: ...

    def rerank(
        self, question: str, candidates: tuple[RuntimeEvidence, ...],
    ) -> tuple[RuntimeEvidence, ...]: ...


@runtime_checkable
class ContextAssembler(Protocol):
    def assemble(
        self, candidates: tuple[RuntimeEvidence, ...], *, max_chars: int,
    ) -> ContextManifest:
        """Return only ordered evidence actually passed to generation."""
        ...


@runtime_checkable
class SufficiencyGate(Protocol):
    def evaluate(
        self, question: str, context: ContextManifest, *, config: RuntimeConfig,
    ) -> SufficiencyDecision:
        """Handle unconfigured thresholds explicitly; sufficient is not entailment."""
        ...


@runtime_checkable
class GenerationProvider(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationDraft: ...


@runtime_checkable
class CitationValidator(Protocol):
    def validate(
        self, draft: GenerationDraft, context: ContextManifest,
    ) -> CitationValidationResult:
        """Resolve exact source handles/identities against actual model context.

        Structural validity and semantic support must remain separate outcomes.
        """
        ...


@runtime_checkable
class FinalAnswerPolicy(Protocol):
    def decide(
        self, request: AskRequest, decision: SufficiencyDecision, *,
        context: ContextManifest, draft: GenerationDraft | None,
        citation_validation: CitationValidationResult | None,
        escalation_id: str | None, config: RuntimeConfig,
    ) -> AskResult: ...


@runtime_checkable
class EscalationRepository(Protocol):
    def save(self, escalation: ExpertEscalation) -> None:
        """Persist a record idempotently; this does not deliver it externally."""
        ...

    def get(self, escalation_id: str) -> ExpertEscalation | None: ...
