from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PdfPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedPdf:
    file_name: str
    pages: tuple[PdfPage, ...]

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            page.text
            for page in self.pages
            if page.text
        )


def parse_pdf(file_path: Path) -> ParsedPdf:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Файл не найден: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            "PDF-парсер принимает только .pdf файлы"
        )

    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        raise ValueError(
            f"Не удалось прочитать PDF: {file_path.name}"
        ) from exc

    pages: list[PdfPage] = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise ValueError(
                f"Не удалось извлечь текст со страницы {page_index}"
            ) from exc

        clean_text = text.strip()

        if not clean_text:
            continue

        pages.append(
            PdfPage(
                page_number=page_index,
                text=clean_text,
            )
        )

    return ParsedPdf(
        file_name=file_path.name,
        pages=tuple(pages),
    )
