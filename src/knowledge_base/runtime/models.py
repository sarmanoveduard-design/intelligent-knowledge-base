"""Immutable, provider-neutral runtime records; no execution or storage adapters.

Identifiers are opaque strings supplied by future ingestion implementations.
Scores describe their named scoring space, never an implicit probability.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias


MetadataValue: TypeAlias = (
    str | int | float | bool | None
    | tuple["MetadataValue", ...] | Mapping[str, "MetadataValue"]
)
Metadata: TypeAlias = Mapping[str, MetadataValue]


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _positive_int(value: int, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _score(value: float, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _strings(values, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a collection of strings")
    result = tuple(values)
    for value in result:
        _nonempty(value, name)
    return result


def _freeze_value(value):
    if isinstance(value, Mapping):
        return _metadata(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and isfinite(value):
        return value
    raise ValueError("metadata must contain finite JSON-like values")


def _metadata(values: Metadata) -> Metadata:
    if not isinstance(values, Mapping):
        raise ValueError("metadata must be a mapping")
    for key in values:
        _nonempty(key, "metadata key")
    return MappingProxyType({key: _freeze_value(value) for key, value in values.items()})


def _enum(value, enum_type) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"expected {enum_type.__name__}")


class DocumentVersionStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class ProcessingState(str, Enum):
    REGISTERED = "registered"
    PARSED = "parsed"
    CHUNKED = "chunked"
    INDEXED = "indexed"
    FAILED = "failed"


class AnswerState(str, Enum):
    ANSWER = "answer"
    PARTIAL_ANSWER = "partial_answer"
    CLARIFY = "clarify"
    REFUSE_INSUFFICIENT_CONTEXT = "refuse_insufficient_context"
    CONFLICT = "conflict"
    ESCALATE_EXPERT = "escalate_expert"
    ERROR = "error"


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CLARIFICATION_REQUIRED = "clarification_required"
    CONFLICT = "conflict"
    ERROR = "error"


class ConflictStatus(str, Enum):
    UNKNOWN = "unknown"
    CLEAR = "clear"
    CONFLICT = "conflict"


class StructuralValidity(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class SemanticSupport(str, Enum):
    NOT_CHECKED = "not_checked"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class EscalationStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class DocumentIdentity:
    document_id: str
    organization_id: str

    def __post_init__(self):
        _nonempty(self.document_id, "document_id")
        _nonempty(self.organization_id, "organization_id")


@dataclass(frozen=True)
class DocumentVersionIdentity:
    document_id: str
    version_id: str

    def __post_init__(self):
        _nonempty(self.document_id, "document_id")
        _nonempty(self.version_id, "version_id")


@dataclass(frozen=True)
class ProcessingSnapshotIdentity:
    version: DocumentVersionIdentity
    processing_snapshot_id: str

    def __post_init__(self):
        _nonempty(self.processing_snapshot_id, "processing_snapshot_id")


@dataclass(frozen=True)
class ChunkIdentity:
    """The complete identity is this tuple; local index is never its key.

    Future writers must also assign distinct chunk_id handles to distinct chunks
    in an index. No hashing or allocation algorithm is prescribed here.
    """
    document_id: str
    version_id: str
    processing_snapshot_id: str
    chunk_id: str

    def __post_init__(self):
        for name in ("document_id", "version_id", "processing_snapshot_id", "chunk_id"):
            _nonempty(getattr(self, name), name)

    @property
    def snapshot(self) -> ProcessingSnapshotIdentity:
        return ProcessingSnapshotIdentity(
            DocumentVersionIdentity(self.document_id, self.version_id),
            self.processing_snapshot_id,
        )


@dataclass(frozen=True)
class AccessContext:
    """Trusted caller identity; documents are authorized by a future repository."""
    organization_id: str
    principal_id: str
    scopes: frozenset[str] = frozenset()

    def __post_init__(self):
        _nonempty(self.organization_id, "organization_id")
        _nonempty(self.principal_id, "principal_id")
        object.__setattr__(self, "scopes", frozenset(_strings(self.scopes, "scopes")))


@dataclass(frozen=True)
class RetrievalFilters:
    """Restrictions, never access grants.

    None ID sets mean no additional restriction; empty sets mean match nothing.
    Required scopes are conjunctive. User filters must be composed with trusted
    access and repository restrictions before search.
    """
    allowed_document_ids: frozenset[str] | None = None
    allowed_version_ids: frozenset[str] | None = None
    organization_id: str | None = None
    required_scopes: frozenset[str] = frozenset()

    def __post_init__(self):
        for name in ("allowed_document_ids", "allowed_version_ids"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, frozenset(_strings(value, name)))
        if self.organization_id is not None:
            _nonempty(self.organization_id, "organization_id")
        object.__setattr__(self, "required_scopes", frozenset(_strings(self.required_scopes, "required_scopes")))

    def narrow(self, other: RetrievalFilters) -> RetrievalFilters:
        """Intersect restrictions; no document or version grant can be added."""
        if (self.organization_id is not None and other.organization_id is not None
                and self.organization_id != other.organization_id):
            raise ValueError("organization restrictions conflict")

        def intersect(left, right):
            if left is None:
                return right
            if right is None:
                return left
            return left & right

        return RetrievalFilters(
            intersect(self.allowed_document_ids, other.allowed_document_ids),
            intersect(self.allowed_version_ids, other.allowed_version_ids),
            self.organization_id if self.organization_id is not None else other.organization_id,
            self.required_scopes | other.required_scopes,
        )


def compose_retrieval_filters(
    access: AccessContext,
    authorized: RetrievalFilters,
    requested: RetrievalFilters | None = None,
) -> RetrievalFilters:
    """Bind trusted organization and intersect user restrictions.

    'authorized' comes from trusted repository/policy code, not user input.
    This helper does not discover document grants or implement authorization.
    """
    result = RetrievalFilters(organization_id=access.organization_id).narrow(authorized)
    if requested is not None:
        result = result.narrow(requested)
    if not result.required_scopes <= access.scopes:
        raise ValueError("required scopes are unavailable in access context")
    return result


@dataclass(frozen=True)
class RuntimeDocument:
    identity: DocumentIdentity
    title: str
    required_scopes: frozenset[str] = frozenset()
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        _nonempty(self.title, "title")
        object.__setattr__(self, "required_scopes", frozenset(_strings(self.required_scopes, "required_scopes")))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class RuntimeDocumentVersion:
    identity: DocumentVersionIdentity
    version_label: str
    status: DocumentVersionStatus = DocumentVersionStatus.DRAFT
    processing_state: ProcessingState = ProcessingState.REGISTERED
    approved: bool = False
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        _nonempty(self.version_label, "version_label")
        _enum(self.status, DocumentVersionStatus)
        _enum(self.processing_state, ProcessingState)
        if type(self.approved) is not bool:
            raise ValueError("approved must be boolean")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class SourceCoordinates:
    source_file: str | None = None
    source_format: str | None = None
    page_numbers: tuple[int, ...] = ()
    paragraph_indices: tuple[int, ...] = ()
    section_title: str | None = None
    section_ref: str | None = None

    def __post_init__(self):
        for name in ("source_file", "source_format", "section_title", "section_ref"):
            if getattr(self, name) is not None:
                _nonempty(getattr(self, name), name)
        for name, minimum in (("page_numbers", 1), ("paragraph_indices", 0)):
            values = tuple(getattr(self, name))
            for value in values:
                _positive_int(value, name, minimum=minimum)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class EvidenceChunk:
    """Source content, usable before retrieval scores exist."""
    identity: ChunkIdentity
    document: DocumentIdentity
    version: DocumentVersionIdentity
    text: str
    coordinates: SourceCoordinates = field(default_factory=SourceCoordinates)
    local_chunk_index: int | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        _nonempty(self.text, "text")
        if (self.identity.document_id != self.document.document_id
                or self.version.document_id != self.document.document_id
                or self.identity.version_id != self.version.version_id):
            raise ValueError("chunk, document and version identities must agree")
        if self.local_chunk_index is not None:
            _positive_int(self.local_chunk_index, "local_chunk_index", minimum=0)
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class RuntimeEvidence:
    chunk: EvidenceChunk
    retrieval_score: float
    retrieval_score_type: str
    retrieval_scorer_identity: str
    reranker_score: float | None = None
    reranker_score_type: str | None = None
    reranker_scorer_identity: str | None = None

    def __post_init__(self):
        _score(self.retrieval_score, "retrieval_score")
        _nonempty(self.retrieval_score_type, "retrieval_score_type")
        _nonempty(self.retrieval_scorer_identity, "retrieval_scorer_identity")
        if not (self.reranker_score is None
                and self.reranker_score_type is None and self.reranker_scorer_identity is None):
            if self.reranker_score is None or self.reranker_score_type is None or self.reranker_scorer_identity is None:
                raise ValueError("reranker score, type and scorer identity must be supplied together")
        if self.reranker_score is not None:
            _score(self.reranker_score, "reranker_score")
            _nonempty(self.reranker_score_type, "reranker_score_type")
            _nonempty(self.reranker_scorer_identity, "reranker_scorer_identity")


@dataclass(frozen=True)
class ContextEntry:
    source_handle: str
    evidence: RuntimeEvidence

    def __post_init__(self):
        _nonempty(self.source_handle, "source_handle")


@dataclass(frozen=True)
class ContextManifest:
    """Ordered, exact model input after selection/truncation, not a candidate pool.

    Each entry's text is the exact text sent to the provider; callers creating
    excerpts must preserve their source identity and corresponding coordinates.
    """
    entries: tuple[ContextEntry, ...] = ()
    diagnostics: Metadata = field(default_factory=dict)

    def __post_init__(self):
        entries = tuple(self.entries)
        handles = [entry.source_handle for entry in entries]
        identities = [entry.evidence.chunk.identity for entry in entries]
        chunk_ids = [identity.chunk_id for identity in identities]
        if len(set(handles)) != len(handles) or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("context source handles and chunk IDs must be unique")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "diagnostics", _metadata(self.diagnostics))

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(entry.evidence.chunk.identity.chunk_id for entry in self.entries)


@dataclass(frozen=True)
class ConflictAssessment:
    """An explicit hook assertion; UNKNOWN makes no semantic claim."""
    status: ConflictStatus = ConflictStatus.UNKNOWN

    def __post_init__(self):
        _enum(self.status, ConflictStatus)


@dataclass(frozen=True)
class ClarificationAssessment:
    """Missing parameter names supplied by a trusted hook, never guessed here.

    Hooks should return parameter names only, without secrets or input values.
    """
    missing_parameters: tuple[str, ...] = ()

    def __post_init__(self):
        values = _strings(self.missing_parameters, "missing_parameters")
        if len(set(values)) != len(values):
            raise ValueError("missing parameter names must be unique")
        object.__setattr__(self, "missing_parameters", values)


@dataclass(frozen=True)
class SufficiencyDecision:
    status: SufficiencyStatus
    reason_code: str
    message: str
    evidence_count: int
    diagnostics: Metadata = field(default_factory=dict)
    missing_information: tuple[str, ...] = ()

    def __post_init__(self):
        _enum(self.status, SufficiencyStatus)
        _nonempty(self.reason_code, "reason_code")
        _nonempty(self.message, "message")
        _positive_int(self.evidence_count, "evidence_count", minimum=0)
        object.__setattr__(self, "diagnostics", _metadata(self.diagnostics))
        object.__setattr__(self, "missing_information", _strings(self.missing_information, "missing_information"))


@dataclass(frozen=True)
class GenerationLimits:
    max_output_tokens: int | None = None
    timeout_seconds: float | None = None

    def __post_init__(self):
        if self.max_output_tokens is not None:
            _positive_int(self.max_output_tokens, "max_output_tokens")
        if self.timeout_seconds is not None:
            _score(self.timeout_seconds, "timeout_seconds")
            if self.timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    context_manifest: ContextManifest
    prompt_version: str
    output_schema_version: str
    language: str | None = None
    limits: GenerationLimits = field(default_factory=GenerationLimits)
    instructions: str = ""
    request_id: str | None = None

    def __post_init__(self):
        for name in ("question", "prompt_version", "output_schema_version"):
            _nonempty(getattr(self, name), name)
        if not isinstance(self.instructions, str):
            raise ValueError("instructions must be a string")
        if self.request_id is not None:
            _nonempty(self.request_id, "request_id")
        if self.language is not None:
            _nonempty(self.language, "language")


@dataclass(frozen=True)
class CitationReference:
    source_handle: str
    chunk_identity: ChunkIdentity

    def __post_init__(self):
        _nonempty(self.source_handle, "source_handle")


@dataclass(frozen=True)
class GenerationDraft:
    answer: str
    citations: tuple[CitationReference, ...] = ()
    declared_limitations: tuple[str, ...] = ()
    provider_metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        _nonempty(self.answer, "answer")
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "declared_limitations", _strings(self.declared_limitations, "declared_limitations"))
        object.__setattr__(self, "provider_metadata", _metadata(self.provider_metadata))


@dataclass(frozen=True)
class CitationValidationResult:
    structural_validity: StructuralValidity
    semantic_support: SemanticSupport = SemanticSupport.NOT_CHECKED
    resolved_citations: tuple[CitationReference, ...] = ()
    errors: tuple[str, ...] = ()
    diagnostics: Metadata = field(default_factory=dict)

    def __post_init__(self):
        _enum(self.structural_validity, StructuralValidity)
        _enum(self.semantic_support, SemanticSupport)
        object.__setattr__(self, "resolved_citations", tuple(self.resolved_citations))
        object.__setattr__(self, "errors", _strings(self.errors, "errors"))
        object.__setattr__(self, "diagnostics", _metadata(self.diagnostics))
        if self.structural_validity == StructuralValidity.PASS and self.errors:
            raise ValueError("structural PASS cannot contain errors")


@dataclass(frozen=True)
class AskRequest:
    request_id: str
    question: str
    access: AccessContext
    retrieval_filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    language: str | None = None

    def __post_init__(self):
        _nonempty(self.request_id, "request_id")
        _nonempty(self.question, "question")
        if self.language is not None:
            _nonempty(self.language, "language")


@dataclass(frozen=True)
class AskResult:
    request_id: str
    state: AnswerState
    reason_code: str
    answer: str | None = None
    context_manifest: ContextManifest | None = None
    citation_validation: CitationValidationResult | None = None
    escalation_id: str | None = None
    diagnostics: Metadata = field(default_factory=dict)
    declared_limitations: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()

    def __post_init__(self):
        _nonempty(self.request_id, "request_id")
        _enum(self.state, AnswerState)
        _nonempty(self.reason_code, "reason_code")
        for name in ("answer", "escalation_id"):
            if getattr(self, name) is not None:
                _nonempty(getattr(self, name), name)
        if self.state in (AnswerState.ANSWER, AnswerState.PARTIAL_ANSWER) and self.answer is None:
            raise ValueError("answer states require answer text")
        if self.state == AnswerState.ESCALATE_EXPERT and self.escalation_id is None:
            raise ValueError("expert escalation state requires escalation_id")
        object.__setattr__(self, "diagnostics", _metadata(self.diagnostics))
        object.__setattr__(self, "declared_limitations", _strings(self.declared_limitations, "declared_limitations"))
        object.__setattr__(self, "missing_information", _strings(self.missing_information, "missing_information"))


@dataclass(frozen=True)
class ExpertEscalation:
    escalation_id: str
    request_id: str
    created_at: datetime
    question: str
    access: AccessContext
    context_manifest: ContextManifest
    reason: str
    missing_information: tuple[str, ...] = ()
    retrieval_diagnostics: Metadata = field(default_factory=dict)
    draft: GenerationDraft | None = None
    status: EscalationStatus = EscalationStatus.PENDING
    sufficiency_diagnostics: Metadata = field(default_factory=dict)

    def __post_init__(self):
        for name in ("escalation_id", "request_id", "question", "reason"):
            _nonempty(getattr(self, name), name)
        if not isinstance(self.created_at, datetime) or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be a timezone-aware datetime")
        _enum(self.status, EscalationStatus)
        object.__setattr__(self, "missing_information", _strings(self.missing_information, "missing_information"))
        object.__setattr__(self, "retrieval_diagnostics", _metadata(self.retrieval_diagnostics))
        object.__setattr__(self, "sufficiency_diagnostics", _metadata(self.sufficiency_diagnostics))


@dataclass(frozen=True)
class IndexIdentity:
    """Embedding space and preprocessing identities, supplied as configuration.

    Equal dimensions alone do not imply compatibility. Snapshot IDs live in the
    per-snapshot manifest; one index may contain many compatible snapshots.
    """
    embedding_provider: str
    embedding_model: str
    vector_dimension: int
    preprocessing_version: str
    chunking_version: str
    representation_version: str
    embedding_revision: str | None = None

    def __post_init__(self):
        for name in ("embedding_provider", "embedding_model", "preprocessing_version",
                     "chunking_version", "representation_version"):
            _nonempty(getattr(self, name), name)
        _positive_int(self.vector_dimension, "vector_dimension")
        if self.embedding_revision is not None:
            _nonempty(self.embedding_revision, "embedding_revision")

    def is_compatible_with(self, other: IndexIdentity) -> bool:
        return self == other


@dataclass(frozen=True)
class IndexManifest:
    identity: IndexIdentity
    snapshot: ProcessingSnapshotIdentity
    chunk_count: int
    ready: bool = False

    def __post_init__(self):
        _positive_int(self.chunk_count, "chunk_count", minimum=0)
        if type(self.ready) is not bool:
            raise ValueError("ready must be boolean")


@dataclass(frozen=True)
class IndexedEvidence:
    chunk: EvidenceChunk
    vector: tuple[float, ...]

    def __post_init__(self):
        vector = tuple(self.vector)
        if not vector:
            raise ValueError("vector cannot be empty")
        for value in vector:
            _score(value, "vector value")
        object.__setattr__(self, "vector", vector)
