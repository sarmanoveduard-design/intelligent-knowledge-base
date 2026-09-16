from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


@dataclass(frozen=True)
class DocumentRecord:
    file_name: str
    extension: str
    size_bytes: int
    sha256: str


def calculate_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def register_document(file_path: Path) -> DocumentRecord:
    if not file_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемый формат: {extension}. "
            f"Разрешены: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return DocumentRecord(
        file_name=file_path.name,
        extension=extension,
        size_bytes=file_path.stat().st_size,
        sha256=calculate_sha256(file_path),
    )


def record_to_dict(record: DocumentRecord) -> dict[str, str | int]:
    return asdict(record)
