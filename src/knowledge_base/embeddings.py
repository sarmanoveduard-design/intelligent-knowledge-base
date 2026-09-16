from __future__ import annotations

from typing import Protocol, Sequence


Vector = tuple[float, ...]


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> tuple[Vector, ...]:
        ...

    def embed_query(
        self,
        text: str,
    ) -> Vector:
        ...
