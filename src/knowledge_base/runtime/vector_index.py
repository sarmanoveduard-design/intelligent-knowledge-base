"""In-memory, registry-backed vector index; explicit construction, no model I/O."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from math import fsum, hypot, isfinite
from threading import RLock
from typing import Sequence

from .models import (
    AccessContext, DocumentVersionIdentity, DocumentVersionStatus, EvidenceChunk,
    IndexIdentity, IndexedEvidence, IndexManifest, ProcessingSnapshotIdentity,
    ProcessingState, RetrievalFilters, RuntimeEvidence, compose_retrieval_filters,
)
from .protocols import IndexingRepository


def validate_vector(values: Sequence[float], dimension: int) -> tuple[float, ...]:
    vector = tuple(values)
    if len(vector) != dimension or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) for x in vector
    ):
        raise ValueError('invalid_vector')
    return vector


def scorer_identity(identity: IndexIdentity) -> str:
    return json.dumps(asdict(identity), sort_keys=True, separators=(',', ':'))


def _unit(vector):
    # Scaling first also handles large finite coordinates without norm overflow.
    scale = max(abs(x) for x in vector)
    if scale == 0:
        return tuple(0.0 for _ in vector)
    scaled = tuple(x / scale for x in vector)
    norm = hypot(*scaled)
    return tuple(x / norm for x in scaled)


def chunk_is_visible(
    repository: IndexingRepository, chunk: EvidenceChunk, access: AccessContext,
    filters: RetrievalFilters, allowed_versions: frozenset[DocumentVersionIdentity],
    identity: IndexIdentity, manifest: IndexManifest | None,
) -> bool:
    """Recheck registry, selected snapshot, durable readiness and exact source."""
    if chunk.version not in allowed_versions or chunk.document.organization_id != access.organization_id:
        return False
    version = repository.get_version(chunk.version)
    document = repository.get_document(chunk.document)
    if (version is None or document is None or version.status != DocumentVersionStatus.ACTIVE
            or version.processing_state != ProcessingState.INDEXED or not version.approved
            or not document.required_scopes <= access.scopes
            or not filters.required_scopes <= document.required_scopes
            or (filters.allowed_document_ids is not None and chunk.identity.document_id not in filters.allowed_document_ids)
            or (filters.allowed_version_ids is not None and chunk.identity.version_id not in filters.allowed_version_ids)):
        return False
    snapshot = chunk.identity.snapshot
    return (manifest is not None and manifest.ready and manifest.identity == identity
            and manifest.snapshot == snapshot
            and repository.get_current_snapshot(chunk.version) == snapshot
            and repository.get_index_manifest(snapshot, identity) == manifest
            and repository.get_chunk(chunk.identity) == chunk)


class MemoryVectorIndex:
    """Atomic complete snapshots, keyed by full ChunkIdentity.

    Vectors disappear on restart; durable manifests do not imply this adapter is
    ready. Rebuild it through IndexingService. Registry filtering precedes top-k.
    A lock protects memory publication; repositories use their own transactions.
    """
    def __init__(self, identity: IndexIdentity, repository: IndexingRepository):
        self._identity = identity
        self._repository = repository
        self._snapshots: dict[ProcessingSnapshotIdentity, tuple[IndexManifest, tuple[IndexedEvidence, ...]]] = {}
        self._lock = RLock()

    @property
    def identity(self) -> IndexIdentity:
        return self._identity

    def upsert(self, manifest: IndexManifest, chunks: Sequence[IndexedEvidence]) -> None:
        chunks = tuple(chunks)
        if manifest.identity != self.identity:
            raise ValueError('incompatible_index_identity')
        if manifest.chunk_count != len(chunks) or not chunks:
            raise ValueError('index_count_mismatch')
        if len({x.chunk.identity for x in chunks}) != len(chunks):
            raise ValueError('duplicate_chunk_identity')
        for item in chunks:
            if item.chunk.identity.snapshot != manifest.snapshot:
                raise ValueError('index_snapshot_mismatch')
            validate_vector(item.vector, self.identity.vector_dimension)
        # Reject changed content/vectors under an immutable snapshot identity.
        with self._lock:
            previous = self._snapshots.get(manifest.snapshot)
            if previous is not None and previous[1] != chunks:
                raise ValueError('index_snapshot_conflict')
            self._snapshots[manifest.snapshot] = (manifest, chunks)

    def get_manifest(self, snapshot: ProcessingSnapshotIdentity) -> IndexManifest | None:
        with self._lock:
            entry = self._snapshots.get(snapshot)
            return entry[0] if entry else None

    def delete_snapshot(self, snapshot: ProcessingSnapshotIdentity) -> None:
        with self._lock:
            self._snapshots.pop(snapshot, None)

    def deactivate_version(self, identity: DocumentVersionIdentity) -> None:
        with self._lock:
            for snapshot, (manifest, chunks) in tuple(self._snapshots.items()):
                if snapshot.version == identity:
                    self._snapshots[snapshot] = (replace(manifest, ready=False), chunks)

    def search(
        self, query_vector: Sequence[float], *, access: AccessContext,
        filters: RetrievalFilters, top_k: int, expected_identity: IndexIdentity,
    ) -> tuple[RuntimeEvidence, ...]:
        if expected_identity != self.identity:
            raise ValueError('incompatible_index_identity')
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError('invalid_top_k')
        query = _unit(validate_vector(query_vector, self.identity.vector_dimension))
        filters = compose_retrieval_filters(access, filters)
        visible = frozenset(v.identity for v in self._repository.list_visible_versions(access, filters=filters))
        with self._lock:
            snapshots = tuple(self._snapshots.values())
        candidates = []
        for manifest, chunks in snapshots:
            if not manifest.ready or manifest.snapshot.version not in visible:
                continue
            for item in chunks:
                if not chunk_is_visible(self._repository, item.chunk, access, filters, visible, self.identity, manifest):
                    continue
                cosine = max(-1.0, min(1.0, fsum(a * b for a, b in zip(query, _unit(item.vector)))))
                candidates.append(RuntimeEvidence(item.chunk, cosine, 'cosine', scorer_identity(self.identity)))
        candidates.sort(key=lambda x: (-x.retrieval_score, x.chunk.identity.document_id,
                                      x.chunk.identity.version_id, x.chunk.identity.processing_snapshot_id,
                                      x.chunk.identity.chunk_id))
        return tuple(candidates[:top_k])
