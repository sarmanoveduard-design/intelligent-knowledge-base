from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knowledge_base.document_registry import (
    DocumentRecord,
    register_document,
)
from knowledge_base.document_versions import (
    DocumentVersion,
    KnowledgeDocument,
    VersionStatus,
    activate_version,
    create_document,
    create_document_version,
    validate_single_active_version,
)


@dataclass(frozen=True)
class IntakeResult:
    file_record: DocumentRecord
    document: KnowledgeDocument
    version: DocumentVersion


@dataclass(frozen=True)
class VersionIntakeResult:
    file_record: DocumentRecord
    document: KnowledgeDocument
    version: DocumentVersion
    versions: tuple[DocumentVersion, ...]


def intake_new_document(
    file_path: Path,
    title: str,
    version_label: str,
    owner: str | None = None,
    status: VersionStatus = VersionStatus.DRAFT,
) -> IntakeResult:
    file_record = register_document(file_path)

    document = create_document(
        title=title,
        owner=owner,
    )

    version = create_document_version(
        document=document,
        version_label=version_label,
        source_sha256=file_record.sha256,
        status=status,
    )

    return IntakeResult(
        file_record=file_record,
        document=document,
        version=version,
    )


def intake_new_version(
    file_path: Path,
    document: KnowledgeDocument,
    version_label: str,
    existing_versions: list[DocumentVersion],
    status: VersionStatus = VersionStatus.DRAFT,
) -> VersionIntakeResult:
    validate_single_active_version(
        existing_versions,
        document.document_id,
    )

    file_record = register_document(file_path)

    duplicate = next(
        (
            version
            for version in existing_versions
            if version.document_id == document.document_id
            and version.source_sha256 == file_record.sha256
        ),
        None,
    )

    if duplicate is not None:
        raise ValueError(
            "Этот файл уже зарегистрирован как версия "
            f"{duplicate.version_label} данного документа"
        )

    new_version = create_document_version(
        document=document,
        version_label=version_label,
        source_sha256=file_record.sha256,
        status=status,
    )

    versions = [
        *existing_versions,
        new_version,
    ]

    if status == VersionStatus.ACTIVE:
        versions = activate_version(
            versions,
            new_version.version_id,
        )

        new_version = next(
            version
            for version in versions
            if version.version_id == new_version.version_id
        )

    validate_single_active_version(
        versions,
        document.document_id,
    )

    return VersionIntakeResult(
        file_record=file_record,
        document=document,
        version=new_version,
        versions=tuple(versions),
    )
