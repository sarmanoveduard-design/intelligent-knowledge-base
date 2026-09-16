from __future__ import annotations

from dataclasses import dataclass

from knowledge_base.docx_normalizer import NormalizedLine
from knowledge_base.structure_classifier import (
    ClassifiedLine,
    StructuralType,
)


@dataclass(frozen=True)
class DocumentSection:
    section_index: int
    title: str | None
    heading_line_index: int | None
    lines: tuple[NormalizedLine, ...]

    @property
    def body_text(self) -> str:
        return "\n".join(
            line.text
            for line in self.lines
        )

    @property
    def full_text(self) -> str:
        parts: list[str] = []

        if self.title:
            parts.append(self.title)

        if self.body_text:
            parts.append(self.body_text)

        return "\n".join(parts)


def build_sections(
    classified_lines: tuple[ClassifiedLine, ...],
) -> tuple[DocumentSection, ...]:
    sections: list[DocumentSection] = []

    current_title: str | None = None
    current_heading_line_index: int | None = None
    current_lines: list[NormalizedLine] = []

    def flush_current() -> None:
        nonlocal current_title
        nonlocal current_heading_line_index
        nonlocal current_lines

        if (
            current_heading_line_index is None
            and not current_lines
        ):
            return

        sections.append(
            DocumentSection(
                section_index=len(sections),
                title=current_title,
                heading_line_index=current_heading_line_index,
                lines=tuple(current_lines),
            )
        )

        current_title = None
        current_heading_line_index = None
        current_lines = []

    for item in classified_lines:
        if item.structural_type in {
            StructuralType.METADATA,
            StructuralType.TITLE,
            StructuralType.TOC,
        }:
            continue

        if item.structural_type == StructuralType.SECTION:
            flush_current()

            current_title = item.line.text
            current_heading_line_index = (
                item.line.logical_index
            )

            continue

        if item.structural_type == StructuralType.CONTENT:
            current_lines.append(item.line)

    flush_current()

    return tuple(sections)
