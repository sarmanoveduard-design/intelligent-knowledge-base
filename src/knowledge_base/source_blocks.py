from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knowledge_base.docx_normalizer import normalize_docx
from knowledge_base.pdf_parser import parse_pdf


@dataclass(frozen=True)
class SourceBlock:
    document_id: str
    version_id: str
    block_index: int
    text: str
    source_file: str
    source_format: str

    page_number: int | None = None
    paragraph_index: int | None = None
    line_index: int | None = None
    section_title: str | None = None


def build_docx_source_blocks(
    file_path: Path,
    *,
    document_id: str,
    version_id: str,
) -> tuple[SourceBlock, ...]:
    lines = normalize_docx(file_path)

    return tuple(
        SourceBlock(
            document_id=document_id,
            version_id=version_id,
            block_index=index,
            text=line.text,
            source_file=file_path.name,
            source_format="docx",
            paragraph_index=line.paragraph_index,
            line_index=line.line_index,
        )
        for index, line in enumerate(lines)
        if line.text.strip()
    )


def build_pdf_source_blocks(
    file_path: Path,
    *,
    document_id: str,
    version_id: str,
) -> tuple[SourceBlock, ...]:
    parsed = parse_pdf(file_path)

    return tuple(
        SourceBlock(
            document_id=document_id,
            version_id=version_id,
            block_index=index,
            text=page.text,
            source_file=file_path.name,
            source_format="pdf",
            page_number=page.page_number,
        )
        for index, page in enumerate(parsed.pages)
        if page.text.strip()
    )


def build_source_blocks(
    file_path: Path,
    *,
    document_id: str,
    version_id: str,
) -> tuple[SourceBlock, ...]:
    extension = file_path.suffix.lower()

    if extension == ".docx":
        return build_docx_source_blocks(
            file_path,
            document_id=document_id,
            version_id=version_id,
        )

    if extension == ".pdf":
        return build_pdf_source_blocks(
            file_path,
            document_id=document_id,
            version_id=version_id,
        )

    raise ValueError(
        f"Неподдерживаемый формат документа: {extension}"
    )
