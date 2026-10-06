"""Versioned document embedding inputs at the benchmark composition boundary."""
from collections.abc import Mapping

DOCUMENT_REPRESENTATIONS = ("plain", "structure_aware_v1")


class DocumentRepresentationError(ValueError):
    """Safe configuration error without source text or supplied environment values."""


def validate_document_representation(mode: str) -> str:
    if mode not in DOCUMENT_REPRESENTATIONS:
        raise DocumentRepresentationError(
            "Invalid document representation; expected plain or structure_aware_v1"
        )
    return mode


def document_embedding_text(text: str, metadata: Mapping[str, str], *, mode: str = "plain") -> str:
    validate_document_representation(mode)
    if mode == "plain":
        return text
    if not isinstance(metadata, Mapping):
        raise DocumentRepresentationError("Invalid document structure metadata")
    lines = []
    for key, label in (("chapter", "Chapter"), ("article_title", "Article")):
        value = metadata.get(key, "")
        if not isinstance(value, str):
            raise DocumentRepresentationError("Invalid document structure metadata")
        value = value.strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines) + "\n\n" + text if lines else text
