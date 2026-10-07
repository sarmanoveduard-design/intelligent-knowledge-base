"""Synchronous file ingestion using existing intake, parsers and chunker.

No embeddings, retrieval or model providers are constructed. Ingestion creates
prepared drafts; explicit publication is separate and uses compare-and-swap.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from knowledge_base.document_intake import intake_new_document, intake_new_version
from knowledge_base.document_registry import SUPPORTED_EXTENSIONS, register_document
from knowledge_base.document_versions import KnowledgeDocument, DocumentVersion, VersionStatus

from .models import (
    ChunkIdentity, DocumentIdentity, DocumentVersionIdentity, EvidenceChunk,
    Metadata, ProcessingSnapshotIdentity, ProcessingState, RuntimeDocument,
    RuntimeDocumentVersion, SourceCoordinates, _nonempty, _positive_int,
)
from .storage import IngestionRepository, StorageError


class IngestionError(RuntimeError):
    """Safe diagnostic only; underlying exception messages are never copied."""
    def __init__(self, reason_code: str, version: DocumentVersionIdentity | None = None,
                 *, failure_recorded: bool = False):
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.version = version
        self.failure_recorded = failure_recorded


@dataclass(frozen=True)
class IngestionResult:
    document: RuntimeDocument
    version: RuntimeDocumentVersion
    snapshot: ProcessingSnapshotIdentity | None
    chunk_count: int
    duplicate: bool = False


@dataclass(frozen=True)
class IngestionService:
    repository: IngestionRepository
    max_chunk_chars: int = 1800
    overlap_pieces: int = 1

    def __post_init__(self):
        _positive_int(self.max_chunk_chars, 'max_chunk_chars')
        _positive_int(self.overlap_pieces, 'overlap_pieces', minimum=0)

    def _read_source(self, file_path: Path) -> tuple[bytes, str, str]:
        file_path = Path(file_path)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise IngestionError('unsupported_format')
        try:
            record = register_document(file_path)
            source = file_path.read_bytes()
        except FileNotFoundError:
            raise IngestionError('file_not_found') from None
        except (OSError, ValueError):
            raise IngestionError('file_read_failed') from None
        if not source:
            raise IngestionError('empty_file')
        if hashlib.sha256(source).hexdigest() != record.sha256:
            raise IngestionError('source_changed_during_read')
        return source, file_path.name, file_path.suffix.lower()[1:]

    def ingest_new_document(
        self, file_path: Path, *, title: str, organization_id: str, version_label: str,
        document_id: str | None = None, external_key: str | None = None,
        required_scopes: frozenset[str] = frozenset(), metadata: Metadata | None = None,
    ) -> IngestionResult:
        """Upload a new logical document, or resolve an explicitly supplied key.

        With no identity/key a matching upload in this organization is reported
        as a duplicate; it never creates a new version from hash alone. Multiple
        matches require an explicit identity. Distinct explicit keys/IDs may
        intentionally represent separate documents with identical source bytes.
        """
        _nonempty(title, 'title')
        _nonempty(version_label, 'version_label')
        _nonempty(organization_id, 'organization_id')
        if document_id is not None:
            _nonempty(document_id, 'document_id')
        if external_key is not None:
            _nonempty(external_key, 'external_key')
        source, name, source_format = self._read_source(file_path)
        try:
            existing = self.repository.get_document_by_external_key(organization_id, external_key) if external_key else None
            if existing is not None and document_id is not None and existing.identity.document_id != document_id:
                raise IngestionError('document_identity_conflict')
            if existing is None and document_id is not None:
                existing = self.repository.get_document(DocumentIdentity(document_id, organization_id))
            document = existing or RuntimeDocument(
                DocumentIdentity(document_id or str(uuid4()), organization_id), title, required_scopes, metadata or {})
            return self._ingest(source, name, source_format, document, version_label,
                                new_document=existing is None, external_key=external_key,
                                explicit_identity=document_id is not None or external_key is not None,
                                resolve_external_key=document_id is None)
        except StorageError:
            raise IngestionError('storage_failed') from None

    def ingest_new_version(
        self, file_path: Path, *, document: DocumentIdentity, version_label: str,
    ) -> IngestionResult:
        """New source bytes require an explicit existing document identity."""
        _nonempty(version_label, 'version_label')
        source, name, source_format = self._read_source(file_path)
        try:
            record = self.repository.get_document(document)
            if record is None:
                raise IngestionError('document_not_found')
            return self._ingest(source, name, source_format, record, version_label,
                                new_document=False, external_key=None, explicit_identity=True, resolve_external_key=False)
        except StorageError:
            raise IngestionError('storage_failed') from None

    def _result(self, version: RuntimeDocumentVersion, organization_id: str, *, duplicate=False):
        document = self.repository.get_document(DocumentIdentity(version.identity.document_id, organization_id))
        if document is None:
            raise StorageError('document_not_found')
        snapshot = self.repository.get_current_snapshot(version.identity)
        chunks = self.repository.list_snapshot(snapshot) if snapshot else ()
        return IngestionResult(document, version, snapshot, len(chunks), duplicate)

    def _ingest(self, source, name, source_format, document, version_label, *,
                new_document, external_key, explicit_identity, resolve_external_key):
        version = None
        phase = 'processing_failed'
        try:
            with TemporaryDirectory(prefix='kb-ingest-') as folder:
                stable_file = Path(folder) / name
                stable_file.write_bytes(source)
                if new_document:
                    intake = intake_new_document(stable_file, document.title, version_label)
                else:
                    # Duplicate lookup precedes legacy intake's duplicate exception.
                    matches = self.repository.find_by_hash(document.identity.organization_id, hashlib.sha256(source).hexdigest())
                    same_document = next((item for item in matches if item.identity.document_id == document.identity.document_id), None)
                    if same_document is not None:
                        return self._result(same_document, document.identity.organization_id, duplicate=True)
                    current = self.repository.get_current_version(document.identity)
                    previous = []
                    if current is not None:
                        previous.append(DocumentVersion(current.identity.version_id, current.identity.document_id,
                            current.version_label, self.repository.get_source_info(current.identity).sha256,
                            VersionStatus(current.status.value)))
                    intake = intake_new_version(stable_file,
                        KnowledgeDocument(document.identity.document_id, document.title), version_label, previous)
                proposed = RuntimeDocumentVersion(
                    DocumentVersionIdentity(document.identity.document_id, intake.version.version_id), version_label)
                phase = 'storage_failed'
                version, duplicate = self.repository.register_source(document, proposed, source=source,
                    file_name=name, source_format=source_format, external_key=external_key,
                    explicit_identity=explicit_identity, resolve_external_key=resolve_external_key)
                if duplicate:
                    return self._result(version, document.identity.organization_id, duplicate=True)
                # A concurrent external-key registration may resolve a different
                # logical identity; always use the authoritative returned record.
                document = self.repository.get_document(DocumentIdentity(version.identity.document_id, document.identity.organization_id))
                phase = 'parsing_failed'
                from knowledge_base.source_blocks import build_source_blocks
                blocks = build_source_blocks(stable_file, document_id=version.identity.document_id,
                                             version_id=version.identity.version_id)
                if not blocks:
                    raise IngestionError('empty_extraction', version.identity)
                phase = 'storage_failed'
                version = replace(version, processing_state=ProcessingState.PARSED)
                self.repository.save_version(version)
                phase = 'chunking_failed'
                from knowledge_base.chunker import chunk_source_blocks
                chunks = chunk_source_blocks(blocks, max_chars=self.max_chunk_chars, overlap_pieces=self.overlap_pieces)
                if not chunks:
                    raise IngestionError('empty_chunks', version.identity)
                snapshot = ProcessingSnapshotIdentity(version.identity, str(uuid4()))
                prepared = tuple(EvidenceChunk(
                    ChunkIdentity(version.identity.document_id, version.identity.version_id,
                                  snapshot.processing_snapshot_id, str(uuid4())),
                    document.identity, version.identity, chunk.text,
                    SourceCoordinates(source_file=name, source_format=source_format,
                                      page_numbers=chunk.page_numbers, paragraph_indices=chunk.paragraph_indices,
                                      section_title=chunk.section_titles[0] if chunk.section_titles else None),
                    chunk.chunk_index, {'block_indices': chunk.block_indices, 'section_titles': chunk.section_titles},
                ) for chunk in chunks)
                phase = 'storage_failed'
                self.repository.finish_processing(snapshot, prepared, metadata={
                    'parser_version': 'source_blocks/v1', 'chunker_version': 'chunk_source_blocks/v1',
                    'max_chunk_chars': self.max_chunk_chars, 'overlap_pieces': self.overlap_pieces,
                })
                version = self.repository.get_version(version.identity)
            return self._result(version, document.identity.organization_id)
        except Exception as error:
            reason = error.reason_code if isinstance(error, IngestionError) else phase
            # Fixed repository codes can be reported without copying DB details.
            if isinstance(error, StorageError) and str(error) in {
                    'ambiguous_duplicate_requires_identity', 'external_key_conflict', 'document_identity_conflict'}:
                reason = str(error)
            failure_recorded = False
            if version is not None:
                try:
                    self.repository.fail_processing(version.identity, reason)
                    failure_recorded = True
                except StorageError:
                    pass
            raise IngestionError(reason, version.identity if version else None,
                                 failure_recorded=failure_recorded) from None

    def activate_version(
        self, identity: DocumentVersionIdentity, *, expected_current_version_id: str | None,
    ) -> None:
        """Explicitly approve/publish a prepared version with optimistic locking."""
        try:
            self.repository.activate_version(identity, expected_current_version_id=expected_current_version_id)
        except StorageError:
            raise IngestionError('activation_failed', identity) from None

    def archive_version(self, identity: DocumentVersionIdentity) -> None:
        try:
            self.repository.archive_version(identity)
        except StorageError:
            raise IngestionError('archival_failed', identity) from None
