from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.text.run import Run


@dataclass(frozen=True)
class NormalizedLine:
    logical_index: int
    paragraph_index: int
    line_index: int
    text: str
    style: str | None
    bold_percent: int
    font_sizes: tuple[float, ...]
    alignment: str | None
    numbered: bool
    formatting_reliable: bool


def _normalize_for_compare(text: str) -> str:
    return " ".join(text.split())


def _iter_all_runs(paragraph):
    for run_element in paragraph._p.xpath(".//w:r"):
        yield Run(run_element, paragraph)


def _fallback_fragments(
    paragraph_text: str,
) -> list[list[tuple[str, bool, float | None]]]:
    return [
        [(line, False, None)]
        for line in paragraph_text.splitlines()
        if line.strip()
    ]


def _extract_line_fragments(
    paragraph,
) -> tuple[
    list[list[tuple[str, bool, float | None]]],
    bool,
]:
    lines: list[list[tuple[str, bool, float | None]]] = []
    current_line: list[tuple[str, bool, float | None]] = []

    extracted_parts: list[str] = []

    for run in _iter_all_runs(paragraph):
        run_text = (
            run.text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        extracted_parts.append(run_text)

        parts = run_text.split("\n")

        for index, part in enumerate(parts):
            size = (
                round(run.font.size.pt, 1)
                if run.font.size is not None
                else None
            )

            current_line.append(
                (
                    part,
                    run.bold is True,
                    size,
                )
            )

            if index < len(parts) - 1:
                lines.append(current_line)
                current_line = []

    if current_line:
        lines.append(current_line)

    extracted_text = "".join(extracted_parts)
    source_text = paragraph.text

    if (
        _normalize_for_compare(extracted_text)
        != _normalize_for_compare(source_text)
    ):
        return (
            _fallback_fragments(source_text),
            False,
        )

    return lines, True


def _line_metrics(
    fragments: list[tuple[str, bool, float | None]],
) -> tuple[str, int, tuple[float, ...]]:
    text = "".join(
        fragment_text
        for fragment_text, _, _ in fragments
    ).strip()

    total_chars = sum(
        len(fragment_text)
        for fragment_text, _, _ in fragments
        if fragment_text.strip()
    )

    bold_chars = sum(
        len(fragment_text)
        for fragment_text, is_bold, _ in fragments
        if fragment_text.strip() and is_bold
    )

    bold_percent = round(
        bold_chars / max(1, total_chars) * 100
    )

    font_sizes = tuple(
        sorted({
            size
            for fragment_text, _, size in fragments
            if fragment_text.strip() and size is not None
        })
    )

    return text, bold_percent, font_sizes


def normalize_docx(
    file_path: Path,
) -> tuple[NormalizedLine, ...]:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Файл не найден: {file_path}"
        )

    if file_path.suffix.lower() != ".docx":
        raise ValueError(
            "Нормализатор принимает только .docx файлы"
        )

    try:
        document = Document(file_path)
    except Exception as exc:
        raise ValueError(
            f"Не удалось прочитать DOCX: {file_path.name}"
        ) from exc

    result: list[NormalizedLine] = []
    logical_index = 0

    for paragraph_index, paragraph in enumerate(
        document.paragraphs
    ):
        if not paragraph.text.strip():
            continue

        style = (
            paragraph.style.name
            if paragraph.style is not None
            else None
        )

        alignment_value = paragraph.alignment
        alignment = (
            getattr(
                alignment_value,
                "name",
                str(alignment_value),
            )
            if alignment_value is not None
            else None
        )

        numbered = (
            paragraph._p.pPr is not None
            and paragraph._p.pPr.numPr is not None
        )

        (
            fragment_lines,
            formatting_reliable,
        ) = _extract_line_fragments(paragraph)

        line_index = 0

        for fragments in fragment_lines:
            (
                text,
                bold_percent,
                font_sizes,
            ) = _line_metrics(fragments)

            if not text:
                continue

            result.append(
                NormalizedLine(
                    logical_index=logical_index,
                    paragraph_index=paragraph_index,
                    line_index=line_index,
                    text=text,
                    style=style,
                    bold_percent=bold_percent,
                    font_sizes=font_sizes,
                    alignment=alignment,
                    numbered=numbered,
                    formatting_reliable=formatting_reliable,
                )
            )

            logical_index += 1
            line_index += 1

    return tuple(result)
