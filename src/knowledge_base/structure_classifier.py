from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from knowledge_base.docx_normalizer import NormalizedLine


class StructuralType(str, Enum):
    METADATA = "metadata"
    TITLE = "title"
    TOC = "toc"
    SECTION = "section"
    CONTENT = "content"


@dataclass(frozen=True)
class ClassifiedLine:
    line: NormalizedLine
    structural_type: StructuralType


METADATA_PATTERNS = (
    re.compile(r"^редакция\b", re.IGNORECASE),
    re.compile(r"^версия\b", re.IGNORECASE),
)


ALWAYS_CONTENT_PREFIXES = (
    "федеральный закон",
    "закон ",
    "гражданский кодекс",
    "уголовный кодекс",
    "трудовой кодекс",
    "кодекс российской",
    "национальный стандарт",
    "гост ",
    "статья ",
)


LEGAL_ACT_PREFIXES = (
    "указ ",
    "постановление ",
    "распоряжение ",
    "приказ ",
)


DOCUMENT_NUMBER_PATTERN = re.compile(
    r"(?:№|\bN\b|\bNo\b)\s*[\w/-]+",
    re.IGNORECASE,
)

DOCUMENT_DATE_PATTERN = re.compile(
    r"\bот\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}\b",
    re.IGNORECASE,
)


def _max_font_size(line: NormalizedLine) -> float:
    if not line.font_sizes:
        return 0.0

    return max(line.font_sizes)


def _is_metadata(line: NormalizedLine) -> bool:
    text = line.text.strip()

    return any(
        pattern.search(text)
        for pattern in METADATA_PATTERNS
    )


def _has_legal_act_requisites(text: str) -> bool:
    return bool(
        DOCUMENT_NUMBER_PATTERN.search(text)
        or DOCUMENT_DATE_PATTERN.search(text)
    )


def _looks_like_legal_content(text: str) -> bool:
    lowered = text.strip().lower()

    if lowered.startswith(ALWAYS_CONTENT_PREFIXES):
        return True

    if lowered.startswith(LEGAL_ACT_PREFIXES):
        return _has_legal_act_requisites(text)

    return False


def _is_strong_section_candidate(
    line: NormalizedLine,
) -> bool:
    text = line.text.strip()

    if not text:
        return False

    if len(text) > 180:
        return False

    if _looks_like_legal_content(text):
        return False

    if line.alignment == "CENTER":
        return True

    if (
        line.bold_percent == 100
        and _max_font_size(line) >= 12
        and len(text) <= 100
    ):
        return True

    if (
        line.bold_percent == 100
        and text.endswith(":")
        and len(text) <= 100
    ):
        return True

    return False


def classify_structure(
    lines: tuple[NormalizedLine, ...],
) -> tuple[ClassifiedLine, ...]:
    result: list[ClassifiedLine] = []

    in_toc = False
    body_started = False

    for index, line in enumerate(lines):
        text = line.text.strip()

        if _is_metadata(line):
            result.append(
                ClassifiedLine(
                    line=line,
                    structural_type=StructuralType.METADATA,
                )
            )
            continue

        if (
            not body_started
            and not in_toc
            and index <= 2
        ):
            result.append(
                ClassifiedLine(
                    line=line,
                    structural_type=StructuralType.TITLE,
                )
            )
            continue

        if (
            not body_started
            and text.casefold() == "содержание"
        ):
            in_toc = True

            result.append(
                ClassifiedLine(
                    line=line,
                    structural_type=StructuralType.TOC,
                )
            )
            continue

        if in_toc:
            if (
                line.alignment == "CENTER"
                and line.bold_percent == 100
                and _max_font_size(line) >= 14
            ):
                in_toc = False
                body_started = True

                result.append(
                    ClassifiedLine(
                        line=line,
                        structural_type=StructuralType.SECTION,
                    )
                )
            else:
                result.append(
                    ClassifiedLine(
                        line=line,
                        structural_type=StructuralType.TOC,
                    )
                )

            continue

        body_started = True

        if _is_strong_section_candidate(line):
            structural_type = StructuralType.SECTION
        else:
            structural_type = StructuralType.CONTENT

        result.append(
            ClassifiedLine(
                line=line,
                structural_type=structural_type,
            )
        )

    return tuple(result)
