"""SQLite document and chunk repositories, with explicit initialization only.

One adapter implements both repository contracts so final snapshot persistence
and processing-state changes share a transaction. Administrative lookups are
trusted operations; list_visible_versions additionally applies access filters.
Original source bytes are stored in the same database as the registry.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Protocol, Sequence

from .models import (
    AccessContext, ChunkIdentity, DocumentIdentity, DocumentVersionIdentity,
    DocumentVersionStatus, EvidenceChunk, IndexIdentity, IndexManifest, Metadata, ProcessingSnapshotIdentity,
    ProcessingState, RetrievalFilters, RuntimeDocument, RuntimeDocumentVersion,
    SourceCoordinates, _metadata, _nonempty, compose_retrieval_filters,
)
from .protocols import ChunkRepository, DocumentRepository


class StorageError(RuntimeError):
    """Safe error code only, without database paths, source text or SQL details."""


@dataclass(frozen=True)
class SourceInfo:
    sha256: str
    file_name: str
    source_format: str
    size_bytes: int
    created_at: datetime


class IngestionRepository(DocumentRepository, ChunkRepository, Protocol):
    """Small transactional extension for ingestion; no SQLite dependency required."""
    def register_source(
        self, document: RuntimeDocument, version: RuntimeDocumentVersion, *,
        source: bytes, file_name: str, source_format: str,
        external_key: str | None, explicit_identity: bool, resolve_external_key: bool = False,
    ) -> tuple[RuntimeDocumentVersion, bool]: ...

    def find_by_hash(self, organization_id: str, sha256: str) -> tuple[RuntimeDocumentVersion, ...]: ...

    def get_document_by_external_key(self, organization_id: str, external_key: str) -> RuntimeDocument | None: ...

    def get_current_version(self, document: DocumentIdentity) -> RuntimeDocumentVersion | None: ...

    def get_current_snapshot(self, version: DocumentVersionIdentity) -> ProcessingSnapshotIdentity | None: ...

    def get_source_info(self, version: DocumentVersionIdentity) -> SourceInfo: ...

    def finish_processing(
        self, snapshot: ProcessingSnapshotIdentity, chunks: Sequence[EvidenceChunk], *, metadata: Metadata,
    ) -> None: ...

    def fail_processing(self, version: DocumentVersionIdentity, reason_code: str) -> None: ...


_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    title TEXT NOT NULL,
    required_scopes TEXT NOT NULL,
    metadata TEXT NOT NULL,
    external_key TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (organization_id, external_key)
);
CREATE TABLE IF NOT EXISTS versions (
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    version_id TEXT NOT NULL UNIQUE,
    version_label TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft','active','superseded','archived')),
    processing_state TEXT NOT NULL CHECK (processing_state IN ('registered','parsed','chunked','indexed','failed')),
    approved INTEGER NOT NULL CHECK (approved IN (0,1)),
    metadata TEXT NOT NULL,
    sha256 TEXT,
    source BLOB,
    file_name TEXT,
    source_format TEXT,
    current_snapshot_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (document_id, version_id),
    UNIQUE (document_id, sha256),
    CHECK (status != 'active' OR (approved = 1 AND processing_state IN ('chunked','indexed')))
);
CREATE UNIQUE INDEX IF NOT EXISTS single_active_version
    ON versions(document_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS versions_hash ON versions(sha256);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id, version_id) REFERENCES versions(document_id, version_id)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (snapshot_id, ordinal)
);
CREATE TABLE IF NOT EXISTS index_manifests (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    index_identity TEXT NOT NULL,
    chunk_count INTEGER NOT NULL CHECK (chunk_count > 0),
    PRIMARY KEY (snapshot_id, index_identity)
);
PRAGMA user_version = 2;
COMMIT;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [_json_value(item) for item in value]
    return value


def _dump(value) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _version(row) -> RuntimeDocumentVersion:
    return RuntimeDocumentVersion(
        DocumentVersionIdentity(row['document_id'], row['version_id']), row['version_label'],
        DocumentVersionStatus(row['status']), ProcessingState(row['processing_state']),
        bool(row['approved']), json.loads(row['metadata']),
    )


def _document(row) -> RuntimeDocument:
    return RuntimeDocument(DocumentIdentity(row['document_id'], row['organization_id']),
                           row['title'], frozenset(json.loads(row['required_scopes'])),
                           json.loads(row['metadata']))


def _chunk_payload(chunk: EvidenceChunk) -> str:
    coordinates = chunk.coordinates
    return _dump({
        'document_id': chunk.identity.document_id, 'version_id': chunk.identity.version_id,
        'snapshot_id': chunk.identity.processing_snapshot_id, 'chunk_id': chunk.identity.chunk_id,
        'organization_id': chunk.document.organization_id, 'text': chunk.text,
        'local_chunk_index': chunk.local_chunk_index, 'metadata': chunk.metadata,
        'coordinates': {name: getattr(coordinates, name) for name in (
            'source_file', 'source_format', 'page_numbers', 'paragraph_indices', 'section_title', 'section_ref')},
    })


def _chunk(payload: str) -> EvidenceChunk:
    data = json.loads(payload)
    return EvidenceChunk(
        ChunkIdentity(data['document_id'], data['version_id'], data['snapshot_id'], data['chunk_id']),
        DocumentIdentity(data['document_id'], data['organization_id']),
        DocumentVersionIdentity(data['document_id'], data['version_id']), data['text'],
        SourceCoordinates(**data['coordinates']), data['local_chunk_index'], data['metadata'],
    )


class SQLiteRepository:
    """Explicitly opened SQLite store; no default path or import-time I/O.

    IDs are globally unique opaque strings. Source bytes and creation timestamps
    are immutable. Use one connection per worker; SQLite serializes writers.
    """
    def __init__(self, database_path: str | Path):
        self.connection = None
        try:
            self.connection = sqlite3.connect(str(database_path), isolation_level=None, timeout=5)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute('PRAGMA foreign_keys = ON')
            schema_version = self.connection.execute('PRAGMA user_version').fetchone()[0]
            if schema_version not in (0, 1, 2):
                raise StorageError('unsupported_schema_version')
            self.connection.executescript(_SCHEMA)
        except (sqlite3.Error, StorageError):
            if self.connection is not None:
                if self.connection.in_transaction:
                    self.connection.rollback()
                self.connection.close()
            raise StorageError('storage_initialization_failed') from None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _execute(self, sql, parameters=()):
        try:
            return self.connection.execute(sql, parameters)
        except sqlite3.Error:
            raise StorageError('database_failure') from None

    @contextmanager
    def _transaction(self):
        self._execute('BEGIN IMMEDIATE')
        try:
            yield
            self._execute('COMMIT')
        except Exception:
            try:
                if self.connection.in_transaction:
                    self.connection.rollback()
            except sqlite3.Error:
                raise StorageError('database_failure') from None
            raise

    def _version_row(self, identity: DocumentVersionIdentity):
        return self._execute('SELECT * FROM versions WHERE document_id=? AND version_id=?',
                             (identity.document_id, identity.version_id)).fetchone()

    def _require_version(self, identity):
        row = self._version_row(identity)
        if row is None:
            raise StorageError('version_not_found')
        return row

    def _save_document(self, document):
        row = self._execute('SELECT * FROM documents WHERE document_id=?',
                            (document.identity.document_id,)).fetchone()
        if row is not None and row['organization_id'] != document.identity.organization_id:
            raise StorageError('document_identity_conflict')
        self._execute('''INSERT INTO documents
            (document_id,organization_id,title,required_scopes,metadata,created_at) VALUES (?,?,?,?,?,?)
            ON CONFLICT(document_id) DO UPDATE SET title=excluded.title,
            required_scopes=excluded.required_scopes,metadata=excluded.metadata''',
            (document.identity.document_id, document.identity.organization_id, document.title,
             _dump(sorted(document.required_scopes)), _dump(document.metadata), _now()))

    def save_document(self, document: RuntimeDocument) -> None:
        with self._transaction():
            self._save_document(document)

    def get_document(self, identity: DocumentIdentity) -> RuntimeDocument | None:
        row = self._execute('SELECT * FROM documents WHERE document_id=? AND organization_id=?',
                             (identity.document_id, identity.organization_id)).fetchone()
        return _document(row) if row is not None else None

    def get_document_by_external_key(self, organization_id: str, external_key: str) -> RuntimeDocument | None:
        row = self._execute('SELECT * FROM documents WHERE organization_id=? AND external_key=?',
                            (organization_id, external_key)).fetchone()
        return _document(row) if row is not None else None

    def _save_version(self, version):
        row = self._version_row(version.identity)
        if row is None:
            if version.status != DocumentVersionStatus.DRAFT or version.processing_state != ProcessingState.REGISTERED:
                raise StorageError('new_version_must_be_registered_draft')
            self._execute('''INSERT INTO versions
                (document_id,version_id,version_label,status,processing_state,approved,metadata,created_at)
                VALUES (?,?,?,?,?,?,?,?)''',
                (version.identity.document_id, version.identity.version_id, version.version_label,
                 version.status.value, version.processing_state.value, int(version.approved),
                 _dump(version.metadata), _now()))
            return
        if row['status'] != version.status.value:
            raise StorageError('lifecycle_requires_explicit_operation')
        if row['status'] != 'draft' and row['processing_state'] != version.processing_state.value:
            if not (row['status'] == 'active' and row['processing_state'] == 'chunked'
                    and version.processing_state == ProcessingState.INDEXED):
                raise StorageError('published_processing_is_immutable')
        transitions = {
            'registered': {'registered', 'parsed', 'failed'},
            'parsed': {'parsed', 'failed'},
            'chunked': {'chunked', 'indexed', 'failed'},
            'indexed': {'indexed'},
            'failed': {'failed'},
        }
        if version.processing_state.value not in transitions[row['processing_state']]:
            raise StorageError('invalid_processing_transition')
        if version.processing_state in (ProcessingState.CHUNKED, ProcessingState.INDEXED):
            self._require_prepared(row)
        self._execute('''UPDATE versions SET version_label=?,processing_state=?,approved=?,metadata=?
            WHERE document_id=? AND version_id=?''',
            (version.version_label, version.processing_state.value, int(version.approved), _dump(version.metadata),
             version.identity.document_id, version.identity.version_id))

    def save_version(self, version: RuntimeDocumentVersion) -> None:
        with self._transaction():
            self._save_version(version)

    def get_version(self, identity: DocumentVersionIdentity) -> RuntimeDocumentVersion | None:
        row = self._version_row(identity)
        return _version(row) if row is not None else None

    def get_current_version(self, document: DocumentIdentity) -> RuntimeDocumentVersion | None:
        row = self._execute('''SELECT v.* FROM versions v JOIN documents d USING(document_id)
            WHERE d.document_id=? AND d.organization_id=? AND v.status='active' ''',
            (document.document_id, document.organization_id)).fetchone()
        return _version(row) if row is not None else None

    def list_visible_versions(
        self, access: AccessContext, *, filters: RetrievalFilters,
    ) -> tuple[RuntimeDocumentVersion, ...]:
        effective = compose_retrieval_filters(access, RetrievalFilters(), filters)
        rows = self._execute('''SELECT v.*,d.required_scopes FROM versions v
            JOIN documents d USING(document_id) WHERE d.organization_id=?
            AND v.status='active' AND v.approved=1 AND v.processing_state='indexed'
            ORDER BY v.document_id,v.version_id''', (access.organization_id,)).fetchall()
        return tuple(_version(row) for row in rows
                     if frozenset(json.loads(row['required_scopes'])) <= access.scopes
                     and effective.required_scopes <= frozenset(json.loads(row['required_scopes']))
                     and (effective.allowed_document_ids is None or row['document_id'] in effective.allowed_document_ids)
                     and (effective.allowed_version_ids is None or row['version_id'] in effective.allowed_version_ids))

    def find_by_hash(self, organization_id: str, sha256: str) -> tuple[RuntimeDocumentVersion, ...]:
        rows = self._execute('''SELECT v.* FROM versions v JOIN documents d USING(document_id)
            WHERE d.organization_id=? AND v.sha256=? ORDER BY v.document_id,v.version_id''',
            (organization_id, sha256)).fetchall()
        return tuple(_version(row) for row in rows)

    def register_source(
        self, document: RuntimeDocument, version: RuntimeDocumentVersion, *,
        source: bytes, file_name: str, source_format: str,
        external_key: str | None, explicit_identity: bool, resolve_external_key: bool = False,
    ) -> tuple[RuntimeDocumentVersion, bool]:
        if not source:
            raise StorageError('empty_source')
        _nonempty(file_name, 'file_name')
        _nonempty(source_format, 'source_format')
        if external_key is not None:
            _nonempty(external_key, 'external_key')
        if version.identity.document_id != document.identity.document_id:
            raise StorageError('document_version_mismatch')
        if version.status != DocumentVersionStatus.DRAFT or version.processing_state != ProcessingState.REGISTERED:
            raise StorageError('new_version_must_be_registered_draft')
        sha256 = hashlib.sha256(source).hexdigest()
        with self._transaction():
            if external_key is not None:
                existing = self.get_document_by_external_key(document.identity.organization_id, external_key)
                if existing is not None:
                    if existing.identity != document.identity and not resolve_external_key:
                        raise StorageError('document_identity_conflict')
                    document = existing
                    version = replace(version, identity=DocumentVersionIdentity(
                        existing.identity.document_id, version.identity.version_id))
            existing_document = self._execute('SELECT * FROM documents WHERE document_id=?',
                                               (document.identity.document_id,)).fetchone()
            if existing_document is not None and existing_document['organization_id'] != document.identity.organization_id:
                raise StorageError('document_identity_conflict')
            if not explicit_identity:
                matches = self.find_by_hash(document.identity.organization_id, sha256)
                if len(matches) > 1:
                    raise StorageError('ambiguous_duplicate_requires_identity')
                if matches:
                    return matches[0], True
            row = self._execute('SELECT * FROM versions WHERE document_id=? AND sha256=?',
                                 (document.identity.document_id, sha256)).fetchone()
            if row is not None:
                return _version(row), True
            existing = self._execute('SELECT * FROM documents WHERE document_id=?',
                                      (document.identity.document_id,)).fetchone()
            if existing is None:
                self._save_document(document)
            elif existing['organization_id'] != document.identity.organization_id:
                raise StorageError('document_identity_conflict')
            if external_key is not None:
                if existing is not None and existing['external_key'] not in (None, external_key):
                    raise StorageError('external_key_conflict')
                self._execute('UPDATE documents SET external_key=? WHERE document_id=?',
                              (external_key, document.identity.document_id))
            same_identity = self._version_row(version.identity)
            if same_identity is not None and same_identity['source'] is not None:
                raise StorageError('immutable_source_conflict')
            self._save_version(version)
            self._execute('''UPDATE versions SET sha256=?,source=?,file_name=?,source_format=?
                WHERE document_id=? AND version_id=?''',
                (sha256, source, file_name, source_format, version.identity.document_id, version.identity.version_id))
            return version, False

    def get_source(self, version: DocumentVersionIdentity) -> bytes:
        row = self._require_version(version)
        if row['source'] is None:
            raise StorageError('source_not_found')
        return bytes(row['source'])

    def get_source_info(self, version: DocumentVersionIdentity) -> SourceInfo:
        row = self._require_version(version)
        if row['source'] is None:
            raise StorageError('source_not_found')
        return SourceInfo(row['sha256'], row['file_name'], row['source_format'], len(row['source']),
                          datetime.fromisoformat(row['created_at']))

    def get_current_snapshot(self, version: DocumentVersionIdentity) -> ProcessingSnapshotIdentity | None:
        row = self._require_version(version)
        return ProcessingSnapshotIdentity(version, row['current_snapshot_id']) if row['current_snapshot_id'] else None

    def get_index_manifest(
        self, snapshot: ProcessingSnapshotIdentity, identity: IndexIdentity,
    ) -> IndexManifest | None:
        row = self._execute('''SELECT m.chunk_count FROM index_manifests m
            JOIN snapshots s ON s.snapshot_id=m.snapshot_id
            WHERE s.snapshot_id=? AND s.document_id=? AND s.version_id=? AND m.index_identity=?''',
            (snapshot.processing_snapshot_id, snapshot.version.document_id,
             snapshot.version.version_id, _dump(asdict(identity)))).fetchone()
        return IndexManifest(identity, snapshot, row['chunk_count'], ready=True) if row else None

    def complete_indexing(self, manifest: IndexManifest) -> None:
        if not manifest.ready or manifest.chunk_count < 1:
            raise StorageError('index_not_ready')
        with self._transaction():
            row = self._require_version(manifest.snapshot.version)
            if row['status'] not in ('draft', 'active') or row['processing_state'] not in ('chunked', 'indexed'):
                raise StorageError('invalid_indexing_state')
            self._require_prepared(row)
            if row['current_snapshot_id'] != manifest.snapshot.processing_snapshot_id:
                raise StorageError('index_snapshot_mismatch')
            if len(self.list_snapshot(manifest.snapshot)) != manifest.chunk_count:
                raise StorageError('index_count_mismatch')
            self._execute('''INSERT INTO index_manifests(snapshot_id,index_identity,chunk_count)
                VALUES (?,?,?) ON CONFLICT(snapshot_id,index_identity)
                DO UPDATE SET chunk_count=excluded.chunk_count''',
                (manifest.snapshot.processing_snapshot_id, _dump(asdict(manifest.identity)), manifest.chunk_count))
            self._execute('''UPDATE versions SET processing_state='indexed'
                WHERE document_id=? AND version_id=?''',
                (manifest.snapshot.version.document_id, manifest.snapshot.version.version_id))

    def _save_snapshot(self, snapshot, chunks, metadata):
        version = self._require_version(snapshot.version)
        document_row = self._execute('SELECT * FROM documents WHERE document_id=?',
                                     (snapshot.version.document_id,)).fetchone()
        document = _document(document_row)
        chunks = tuple(chunks)
        if any(chunk.identity.snapshot != snapshot or chunk.document != document.identity for chunk in chunks):
            raise StorageError('snapshot_identity_mismatch')
        if len({chunk.identity.chunk_id for chunk in chunks}) != len(chunks):
            raise StorageError('duplicate_chunk_id')
        payloads = tuple(_chunk_payload(chunk) for chunk in chunks)
        existing = self._execute('SELECT * FROM snapshots WHERE snapshot_id=?',
                                  (snapshot.processing_snapshot_id,)).fetchone()
        if existing is not None:
            previous = self._execute('SELECT payload FROM chunks WHERE snapshot_id=? ORDER BY ordinal',
                                     (snapshot.processing_snapshot_id,)).fetchall()
            if (existing['document_id'] != snapshot.version.document_id
                    or existing['version_id'] != snapshot.version.version_id
                    or tuple(row['payload'] for row in previous) != payloads
                    or (metadata is not None and existing['metadata'] != _dump(metadata))):
                raise StorageError('immutable_snapshot_conflict')
            return
        if version['status'] != 'draft' or version['processing_state'] == 'failed':
            raise StorageError('snapshot_requires_processing_draft')
        self._execute('INSERT INTO snapshots VALUES (?,?,?,?,?)',
                      (snapshot.processing_snapshot_id, snapshot.version.document_id, snapshot.version.version_id,
                       _dump(metadata or {}), _now()))
        for ordinal, (chunk, payload) in enumerate(zip(chunks, payloads)):
            self._execute('INSERT INTO chunks VALUES (?,?,?,?,?)',
                          (chunk.identity.chunk_id, snapshot.processing_snapshot_id, ordinal, payload, _now()))

    def save_snapshot(self, snapshot: ProcessingSnapshotIdentity, chunks: Sequence[EvidenceChunk]) -> None:
        with self._transaction():
            self._save_snapshot(snapshot, chunks, None)

    def list_snapshot(self, snapshot: ProcessingSnapshotIdentity) -> tuple[EvidenceChunk, ...]:
        rows = self._execute('''SELECT c.payload FROM chunks c JOIN snapshots s ON s.snapshot_id=c.snapshot_id
            WHERE s.document_id=? AND s.version_id=? AND s.snapshot_id=? ORDER BY c.ordinal''',
            (snapshot.version.document_id, snapshot.version.version_id, snapshot.processing_snapshot_id)).fetchall()
        return tuple(_chunk(row['payload']) for row in rows)

    def get_chunk(self, identity: ChunkIdentity) -> EvidenceChunk | None:
        row = self._execute('''SELECT c.payload FROM chunks c JOIN snapshots s ON s.snapshot_id=c.snapshot_id
            WHERE c.chunk_id=? AND s.document_id=? AND s.version_id=? AND s.snapshot_id=?''',
            (identity.chunk_id, identity.document_id, identity.version_id, identity.processing_snapshot_id)).fetchone()
        return _chunk(row['payload']) if row is not None else None

    def list_version_chunks(self, version: DocumentVersionIdentity) -> tuple[EvidenceChunk, ...]:
        snapshot = self.get_current_snapshot(version)
        return self.list_snapshot(snapshot) if snapshot else ()

    def get_snapshot_metadata(self, snapshot: ProcessingSnapshotIdentity) -> Metadata:
        row = self._execute('SELECT metadata FROM snapshots WHERE snapshot_id=? AND document_id=? AND version_id=?',
                            (snapshot.processing_snapshot_id, snapshot.version.document_id, snapshot.version.version_id)).fetchone()
        if row is None:
            raise StorageError('snapshot_not_found')
        return _metadata(json.loads(row['metadata']))

    def delete_snapshot(self, snapshot: ProcessingSnapshotIdentity) -> None:
        with self._transaction():
            if self._execute('SELECT 1 FROM versions WHERE current_snapshot_id=?',
                             (snapshot.processing_snapshot_id,)).fetchone() is not None:
                raise StorageError('cannot_delete_selected_snapshot')
            self._execute('DELETE FROM snapshots WHERE snapshot_id=? AND document_id=? AND version_id=?',
                          (snapshot.processing_snapshot_id, snapshot.version.document_id, snapshot.version.version_id))

    def finish_processing(
        self, snapshot: ProcessingSnapshotIdentity, chunks: Sequence[EvidenceChunk], *, metadata: Metadata,
    ) -> None:
        chunks = tuple(chunks)
        if not chunks:
            raise StorageError('empty_chunks')
        with self._transaction():
            row = self._require_version(snapshot.version)
            if row['status'] != 'draft' or row['processing_state'] not in ('parsed', 'chunked'):
                raise StorageError('invalid_processing_transition')
            if row['current_snapshot_id'] not in (None, snapshot.processing_snapshot_id):
                raise StorageError('processing_snapshot_already_selected')
            self._save_snapshot(snapshot, chunks, metadata)
            self._execute('''UPDATE versions SET processing_state='chunked',current_snapshot_id=?
                WHERE document_id=? AND version_id=?''',
                (snapshot.processing_snapshot_id, snapshot.version.document_id, snapshot.version.version_id))

    def fail_processing(self, version: DocumentVersionIdentity, reason_code: str) -> None:
        safe_codes = {'parsing_failed', 'empty_extraction', 'chunking_failed', 'empty_chunks',
                      'storage_failed', 'processing_failed'}
        if reason_code not in safe_codes:
            raise StorageError('invalid_failure_reason')
        with self._transaction():
            row = self._require_version(version)
            if row['status'] != 'draft' or row['processing_state'] == 'indexed':
                raise StorageError('published_processing_is_immutable')
            metadata = json.loads(row['metadata'])
            metadata['failure_reason'] = reason_code
            self._execute('''UPDATE versions SET processing_state='failed',metadata=?
                WHERE document_id=? AND version_id=?''',
                (_dump(metadata), version.document_id, version.version_id))

    def _require_prepared(self, row):
        count = self._execute('''SELECT COUNT(*) FROM chunks c JOIN snapshots s ON s.snapshot_id=c.snapshot_id
            WHERE s.snapshot_id=? AND s.document_id=? AND s.version_id=?''',
            (row['current_snapshot_id'], row['document_id'], row['version_id'])).fetchone()[0]
        if not count:
            raise StorageError('version_not_prepared')

    def activate_version(
        self, identity: DocumentVersionIdentity, *, expected_current_version_id: str | None,
    ) -> None:
        """Explicit approval/publication; CHUNKED is not retrieval-ready yet."""
        with self._transaction():
            target = self._require_version(identity)
            current = self._execute("SELECT version_id FROM versions WHERE document_id=? AND status='active'",
                                    (identity.document_id,)).fetchone()
            if (current['version_id'] if current is not None else None) != expected_current_version_id:
                raise StorageError('current_version_conflict')
            if target['status'] not in ('draft', 'active') or target['processing_state'] not in ('chunked', 'indexed'):
                raise StorageError('version_not_publishable')
            self._require_prepared(target)
            self._execute("UPDATE versions SET status='superseded' WHERE document_id=? AND status='active' AND version_id!=?",
                          (identity.document_id, identity.version_id))
            self._execute("UPDATE versions SET status='active',approved=1 WHERE document_id=? AND version_id=?",
                          (identity.document_id, identity.version_id))

    def archive_version(self, identity: DocumentVersionIdentity) -> None:
        with self._transaction():
            self._require_version(identity)
            self._execute("UPDATE versions SET status='archived' WHERE document_id=? AND version_id=?",
                          (identity.document_id, identity.version_id))
