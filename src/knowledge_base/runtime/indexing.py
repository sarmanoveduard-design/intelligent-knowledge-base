"""Embedding and index publication orchestration, independent of vector backend."""
from __future__ import annotations

from dataclasses import dataclass

from knowledge_base.embeddings import EmbeddingProvider, embedding_identity

from .models import (
    DocumentVersionIdentity, DocumentVersionStatus, IndexIdentity, IndexedEvidence,
    IndexManifest, ProcessingState,
)
from .protocols import IndexingRepository, VectorIndex
from .vector_index import validate_vector


def validate_embedding_identity(provider: EmbeddingProvider, identity: IndexIdentity) -> None:
    """Revision, when configured, must be declared by the provider as revision.

    The existing optional EmbeddingIdentity extension identifies provider/model.
    Processing/representation revisions are supplied in IndexIdentity by the caller.
    """
    actual = embedding_identity(provider)
    if (actual.provider != identity.embedding_provider or actual.model_name != identity.embedding_model
            or actual.dimensions != identity.vector_dimension
            or getattr(provider, 'revision', None) != identity.embedding_revision):
        raise ValueError('incompatible_embedding_identity')


class IndexingError(RuntimeError):
    def __init__(self, reason_code: str, version: DocumentVersionIdentity):
        self.reason_code = reason_code
        self.version = version
        super().__init__(reason_code)


@dataclass(frozen=True)
class IndexingService:
    repository: IndexingRepository
    index: VectorIndex
    embeddings: EmbeddingProvider
    identity: IndexIdentity

    def index_version(self, identity: DocumentVersionIdentity) -> IndexManifest:
        reason = 'indexing_validation_failed'
        new_publication = False
        snapshot = None
        try:
            reason = 'incompatible_index_identity'
            if self.identity != self.index.identity:
                raise ValueError('incompatible_index_identity')
            reason = 'incompatible_embedding_identity'
            validate_embedding_identity(self.embeddings, self.identity)
            reason = 'invalid_indexing_state'
            version = self.repository.get_version(identity)
            if (version is None or version.status not in (DocumentVersionStatus.DRAFT, DocumentVersionStatus.ACTIVE)
                    or version.processing_state not in (ProcessingState.CHUNKED, ProcessingState.INDEXED)):
                raise ValueError('invalid_indexing_state')
            reason = 'index_snapshot_mismatch'
            snapshot = self.repository.get_current_snapshot(identity)
            if snapshot is None or snapshot.version != identity:
                raise ValueError('index_snapshot_mismatch')
            reason = 'invalid_index_chunks'
            chunks = self.repository.list_snapshot(snapshot)
            if not chunks or any(x.identity.snapshot != snapshot for x in chunks):
                raise ValueError('invalid_index_chunks')
            manifest = IndexManifest(self.identity, snapshot, len(chunks), ready=True)
            previous = self.index.get_manifest(snapshot)
            if previous == manifest and self.repository.get_index_manifest(snapshot, self.identity) == manifest:
                return manifest
            reason = 'embedding_failed'
            vectors = tuple(self.embeddings.embed_texts(tuple(x.text for x in chunks)))
            reason = 'invalid_embedding_vectors'
            if len(vectors) != len(chunks):
                raise ValueError('embedding_count_mismatch')
            indexed = tuple(IndexedEvidence(chunk, validate_vector(vector, self.identity.vector_dimension))
                            for chunk, vector in zip(chunks, vectors))
            reason = 'index_write_failed'
            new_publication = previous is None or not previous.ready
            self.index.upsert(manifest, indexed)
            if self.index.get_manifest(snapshot) != manifest:
                raise ValueError('index_not_ready')
            reason = 'index_completion_failed'
            self.repository.complete_indexing(manifest)
            return manifest
        except Exception:
            # No processing-state downgrade; CHUNKED remains retryable, old active
            # versions and ready snapshots are untouched. Missing memory is closed.
            if new_publication and snapshot is not None:
                try:
                    self.index.delete_snapshot(snapshot)
                except Exception:
                    pass
            raise IndexingError(reason, identity) from None
