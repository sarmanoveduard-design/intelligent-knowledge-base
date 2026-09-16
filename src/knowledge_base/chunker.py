from __future__ import annotations

from dataclasses import dataclass

from knowledge_base.source_blocks import SourceBlock


@dataclass(frozen=True)
class Chunk:
    document_id: str
    version_id: str
    chunk_index: int
    text: str
    source_file: str
    source_format: str
    block_indices: tuple[int, ...]
    page_numbers: tuple[int, ...]
    paragraph_indices: tuple[int, ...]
    section_titles: tuple[str, ...]


@dataclass(frozen=True)
class _Piece:
    text: str
    source_block: SourceBlock


def _ordered_unique(
    values,
) -> tuple:
    seen = set()
    result = []

    for value in values:
        if value is None:
            continue

        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return tuple(result)


def _split_text(
    text: str,
    max_chars: int,
) -> tuple[str, ...]:
    remaining = text.strip()

    if not remaining:
        return ()

    pieces: list[str] = []

    while len(remaining) > max_chars:
        search_area = remaining[: max_chars + 1]

        candidates = [
            search_area.rfind("\n"),
            search_area.rfind(". "),
            search_area.rfind("; "),
            search_area.rfind(", "),
            search_area.rfind(" "),
        ]

        split_at = max(candidates)

        minimum_preferred_split = int(
            max_chars * 0.6
        )

        if split_at < minimum_preferred_split:
            split_at = max_chars

        piece = remaining[:split_at].strip()

        if not piece:
            piece = remaining[:max_chars]
            split_at = max_chars

        pieces.append(piece)

        remaining = remaining[split_at:].strip()

    if remaining:
        pieces.append(remaining)

    return tuple(pieces)


def _expand_blocks(
    blocks: tuple[SourceBlock, ...],
    max_chars: int,
) -> tuple[_Piece, ...]:
    pieces: list[_Piece] = []

    for block in blocks:
        for text_piece in _split_text(
            block.text,
            max_chars,
        ):
            pieces.append(
                _Piece(
                    text=text_piece,
                    source_block=block,
                )
            )

    return tuple(pieces)


def _combined_length(
    pieces: list[_Piece],
    next_piece: _Piece | None = None,
) -> int:
    texts = [
        piece.text
        for piece in pieces
    ]

    if next_piece is not None:
        texts.append(next_piece.text)

    return len("\n\n".join(texts))


def _build_chunk(
    pieces: list[_Piece],
    chunk_index: int,
) -> Chunk:
    first = pieces[0].source_block

    return Chunk(
        document_id=first.document_id,
        version_id=first.version_id,
        chunk_index=chunk_index,
        text="\n\n".join(
            piece.text
            for piece in pieces
        ),
        source_file=first.source_file,
        source_format=first.source_format,
        block_indices=_ordered_unique(
            piece.source_block.block_index
            for piece in pieces
        ),
        page_numbers=_ordered_unique(
            piece.source_block.page_number
            for piece in pieces
        ),
        paragraph_indices=_ordered_unique(
            piece.source_block.paragraph_index
            for piece in pieces
        ),
        section_titles=_ordered_unique(
            piece.source_block.section_title
            for piece in pieces
        ),
    )


def _validate_blocks(
    blocks: tuple[SourceBlock, ...],
) -> None:
    if not blocks:
        return

    first = blocks[0]

    expected_identity = (
        first.document_id,
        first.version_id,
        first.source_file,
        first.source_format,
    )

    for block in blocks[1:]:
        identity = (
            block.document_id,
            block.version_id,
            block.source_file,
            block.source_format,
        )

        if identity != expected_identity:
            raise ValueError(
                "Нельзя смешивать разные документы "
                "или версии в одном вызове chunker"
            )


def chunk_source_blocks(
    blocks: tuple[SourceBlock, ...],
    *,
    max_chars: int = 1800,
    overlap_pieces: int = 1,
) -> tuple[Chunk, ...]:
    if max_chars <= 0:
        raise ValueError(
            "max_chars должен быть больше нуля"
        )

    if overlap_pieces < 0:
        raise ValueError(
            "overlap_pieces не может быть отрицательным"
        )

    if not blocks:
        return ()

    _validate_blocks(blocks)

    pieces = _expand_blocks(
        blocks,
        max_chars,
    )

    chunks: list[Chunk] = []
    current: list[_Piece] = []

    for piece in pieces:
        if (
            current
            and _combined_length(
                current,
                piece,
            ) > max_chars
        ):
            chunks.append(
                _build_chunk(
                    current,
                    len(chunks),
                )
            )

            if overlap_pieces:
                current = current[
                    -overlap_pieces:
                ]
            else:
                current = []

            while (
                current
                and _combined_length(
                    current,
                    piece,
                ) > max_chars
            ):
                current = current[1:]

        current.append(piece)

    if current:
        chunks.append(
            _build_chunk(
                current,
                len(chunks),
            )
        )

    return tuple(chunks)
