"""Explicit local composition; provider names belong in caller configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knowledge_base.embeddings import EmbeddingProvider, embedding_identity

from .ask import AskService
from .citations import StructuralCitationValidator
from .config import RuntimeConfig
from .context import RankedContextAssembler
from .indexing import IndexingService, validate_embedding_identity
from .ingestion import IngestionService
from .models import AccessContext, AnswerState, AskRequest, AskResult, IndexIdentity, RetrievalFilters
from .ollama import OllamaGenerationConfig, OllamaGenerationProvider, OllamaTransport
from .policy import RuntimeFinalAnswerPolicy
from .protocols import RuntimeReranker
from .retrieval import RetrievalPipeline
from .storage import SQLiteRepository
from .sufficiency import SufficiencyPolicy
from .vector_index import MemoryVectorIndex


@dataclass(frozen=True)
class IndexReadiness:
    status: str
    visible_versions: int
    ready_versions: int


@dataclass(frozen=True)
class LocalRuntime:
    repository: SQLiteRepository
    ingestion: IngestionService
    indexing: IndexingService
    index: MemoryVectorIndex
    generation: OllamaGenerationProvider
    service: AskService

    def readiness(self, access: AccessContext, *, filters: RetrievalFilters | None = None) -> IndexReadiness:
        versions = self.repository.list_visible_versions(access, filters=filters or RetrievalFilters())
        ready = 0
        for version in versions:
            snapshot = self.repository.get_current_snapshot(version.identity)
            manifest = self.index.get_manifest(snapshot) if snapshot else None
            if (manifest is not None and manifest.ready
                    and self.repository.get_index_manifest(snapshot, self.index.identity) == manifest):
                ready += 1
        status = 'EMPTY' if not versions else ('READY' if ready == len(versions) else 'INDEX_NOT_READY')
        return IndexReadiness(status, len(versions), ready)

    def rebuild(self, access: AccessContext, *, filters: RetrievalFilters | None = None):
        """Explicitly re-embed persisted ACTIVE/INDEXED versions allowed to this access.

        This calls the configured embedding provider, never generation. It does
        not publish drafts or change access. New CHUNKED drafts use indexing first.
        """
        versions = self.repository.list_visible_versions(access, filters=filters or RetrievalFilters())
        return tuple(self.indexing.index_version(version.identity) for version in
                     sorted(versions, key=lambda v: (v.identity.document_id, v.identity.version_id)))

    def ask(self, request: AskRequest) -> AskResult:
        try:
            readiness = self.readiness(request.access, filters=request.retrieval_filters)
        except Exception:
            return AskResult(request.request_id, AnswerState.ERROR, 'INDEX_READINESS_FAILED',
                             diagnostics={'generation': {'called': False, 'status': 'not_called'}})
        if readiness.status == 'INDEX_NOT_READY':
            return AskResult(request.request_id, AnswerState.ERROR, 'INDEX_NOT_READY', diagnostics={
                'index_readiness': {'status': readiness.status, 'visible_versions': readiness.visible_versions,
                                    'ready_versions': readiness.ready_versions},
                'generation': {'called': False, 'status': 'not_called'},
            })
        return self.service.ask(request)

    def close(self):
        self.repository.close()

    def __enter__(self): return self

    def __exit__(self, *args): self.close()


def compose_local_runtime(
    database_path: str | Path, *, embeddings: EmbeddingProvider, reranker: RuntimeReranker,
    generation_config: OllamaGenerationConfig, index_identity: IndexIdentity | None = None,
    runtime_config: RuntimeConfig = RuntimeConfig(), transport: OllamaTransport | None = None,
    max_chunk_chars: int = 1800, overlap_pieces: int = 1,
) -> LocalRuntime:
    """Open SQLite and wire stages 1–5; no model health/loading/generation here.

    Supply the existing embedding adapter and an explicitly prepared runtime
    reranker. Composition never imports the legacy benchmark-coupled loader.
    Memory starts empty on every construction: check readiness and call rebuild.
    Local model inventory must be checked separately before any live model use.
    """
    actual = embedding_identity(embeddings)
    identity = index_identity or IndexIdentity(actual.provider, actual.model_name, actual.dimensions,
        'source_blocks/v1', 'chunk_source_blocks/v1', 'exact_chunk_text/v1', getattr(embeddings, 'revision', None))
    validate_embedding_identity(embeddings, identity)
    repository = SQLiteRepository(database_path)
    try:
        index = MemoryVectorIndex(identity, repository)
        indexing = IndexingService(repository, index, embeddings, identity)
        ingestion = IngestionService(repository, max_chunk_chars, overlap_pieces)
        generation = OllamaGenerationProvider(generation_config, transport=transport)
        retrieval = RetrievalPipeline(repository, index, embeddings, reranker, identity, runtime_config)
        service = AskService(retrieval, RankedContextAssembler(), SufficiencyPolicy(repository), generation,
                             StructuralCitationValidator(), RuntimeFinalAnswerPolicy(), config=runtime_config)
        return LocalRuntime(repository, ingestion, indexing, index, generation, service)
    except Exception:
        repository.close()
        raise
