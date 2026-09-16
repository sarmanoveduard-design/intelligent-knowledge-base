from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from knowledge_base.document_registry import (
    SUPPORTED_EXTENSIONS,
    DocumentRecord,
    register_document,
)


def scan_incoming_folder(folder_path: Path) -> list[DocumentRecord]:
    if not folder_path.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Это не папка: {folder_path}")

    records: list[DocumentRecord] = []

    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        records.append(register_document(file_path))

    return records


def find_exact_duplicates(
    records: list[DocumentRecord],
) -> dict[str, list[DocumentRecord]]:
    grouped: dict[str, list[DocumentRecord]] = defaultdict(list)

    for record in records:
        grouped[record.sha256].append(record)

    return {
        sha256: items
        for sha256, items in grouped.items()
        if len(items) > 1
    }
