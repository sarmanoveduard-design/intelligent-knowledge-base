from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from knowledge_base.document_registry import DocumentRecord, record_to_dict
from knowledge_base.incoming_scanner import find_exact_duplicates


def build_registry(records: list[DocumentRecord]) -> dict:
    duplicates = find_exact_duplicates(records)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(records),
        "documents": [record_to_dict(record) for record in records],
        "exact_duplicates": [
            {
                "sha256": sha256,
                "file_names": sorted(
                    record.file_name for record in duplicate_records
                ),
            }
            for sha256, duplicate_records in sorted(duplicates.items())
        ],
    }


def save_registry(registry: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            registry,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_registry(registry_path: Path) -> dict:
    if not registry_path.is_file():
        raise FileNotFoundError(f"Реестр не найден: {registry_path}")

    return json.loads(registry_path.read_text(encoding="utf-8"))
