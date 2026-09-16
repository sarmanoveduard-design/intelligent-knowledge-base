from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from uuid import uuid4


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class VersionStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    title: str
    owner: str | None = None


@dataclass(frozen=True)
class DocumentVersion:
    version_id: str
    document_id: str
    version_label: str
    source_sha256: str
    status: VersionStatus


def create_document(
    title: str,
    owner: str | None = None,
) -> KnowledgeDocument:
    clean_title = title.strip()

    if not clean_title:
        raise ValueError("Название документа не может быть пустым")

    return KnowledgeDocument(
        document_id=str(uuid4()),
        title=clean_title,
        owner=owner.strip() if owner else None,
    )


def create_document_version(
    document: KnowledgeDocument,
    version_label: str,
    source_sha256: str,
    status: VersionStatus = VersionStatus.DRAFT,
) -> DocumentVersion:
    clean_version = version_label.strip()
    clean_sha256 = source_sha256.strip().lower()

    if not clean_version:
        raise ValueError("Версия документа не может быть пустой")

    if not SHA256_PATTERN.fullmatch(clean_sha256):
        raise ValueError("Некорректный SHA-256")

    return DocumentVersion(
        version_id=str(uuid4()),
        document_id=document.document_id,
        version_label=clean_version,
        source_sha256=clean_sha256,
        status=status,
    )


def activate_version(
    versions: list[DocumentVersion],
    target_version_id: str,
) -> list[DocumentVersion]:
    target = next(
        (
            version
            for version in versions
            if version.version_id == target_version_id
        ),
        None,
    )

    if target is None:
        raise ValueError(
            f"Версия не найдена: {target_version_id}"
        )

    if target.status == VersionStatus.ARCHIVED:
        raise ValueError(
            "Архивную версию нельзя сделать актуальной"
        )

    updated_versions: list[DocumentVersion] = []

    for version in versions:
        if version.document_id != target.document_id:
            updated_versions.append(version)
            continue

        if version.version_id == target.version_id:
            updated_versions.append(
                replace(
                    version,
                    status=VersionStatus.ACTIVE,
                )
            )
            continue

        if version.status == VersionStatus.ACTIVE:
            updated_versions.append(
                replace(
                    version,
                    status=VersionStatus.SUPERSEDED,
                )
            )
            continue

        updated_versions.append(version)

    return updated_versions


def validate_single_active_version(
    versions: list[DocumentVersion],
    document_id: str,
) -> None:
    active_versions = [
        version
        for version in versions
        if version.document_id == document_id
        and version.status == VersionStatus.ACTIVE
    ]

    if len(active_versions) > 1:
        raise ValueError(
            "У документа обнаружено несколько актуальных версий"
        )


def get_active_version(
    versions: list[DocumentVersion],
    document_id: str,
) -> DocumentVersion | None:
    validate_single_active_version(
        versions=versions,
        document_id=document_id,
    )

    return next(
        (
            version
            for version in versions
            if version.document_id == document_id
            and version.status == VersionStatus.ACTIVE
        ),
        None,
    )
