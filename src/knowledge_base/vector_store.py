from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

from knowledge_base.chunker import Chunk
from knowledge_base.embeddings import Vector


@dataclass(frozen=True)
class IndexedChunk:
    chunk: Chunk
    vector: Vector


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def cosine_similarity(
    left: Vector,
    right: Vector,
) -> float:
    if not left or not right:
        raise ValueError(
            "Векторы не могут быть пустыми"
        )

    if len(left) != len(right):
        raise ValueError(
            "Размерности векторов должны совпадать"
        )

    dot_product = sum(
        a * b
        for a, b in zip(left, right)
    )

    left_norm = sqrt(
        sum(value * value for value in left)
    )

    right_norm = sqrt(
        sum(value * value for value in right)
    )

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (
        left_norm * right_norm
    )


class InMemoryVectorStore:
    def __init__(
        self,
        *,
        dimension: int,
    ) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension должен быть больше нуля"
            )

        self._dimension = dimension
        self._items: list[IndexedChunk] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def count(self) -> int:
        return len(self._items)

    def _validate_vector(
        self,
        vector: Vector,
    ) -> None:
        if len(vector) != self._dimension:
            raise ValueError(
                "Размерность вектора не соответствует "
                "размерности хранилища"
            )

    def add(
        self,
        chunk: Chunk,
        vector: Vector,
    ) -> None:
        self._validate_vector(vector)

        self._items.append(
            IndexedChunk(
                chunk=chunk,
                vector=tuple(vector),
            )
        )

    def add_many(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                "Количество chunks и vectors "
                "должно совпадать"
            )

        for vector in vectors:
            self._validate_vector(vector)

        new_items = [
            IndexedChunk(
                chunk=chunk,
                vector=tuple(vector),
            )
            for chunk, vector in zip(
                chunks,
                vectors,
            )
        ]

        self._items.extend(new_items)

    def search(
        self,
        query_vector: Vector,
        *,
        top_k: int = 5,
    ) -> tuple[SearchResult, ...]:
        self._validate_vector(query_vector)

        if top_k <= 0:
            raise ValueError(
                "top_k должен быть больше нуля"
            )

        results = [
            SearchResult(
                chunk=item.chunk,
                score=cosine_similarity(
                    query_vector,
                    item.vector,
                ),
            )
            for item in self._items
        ]

        results.sort(
            key=lambda result: (
                -result.score,
                result.chunk.chunk_index,
            )
        )

        return tuple(
            results[:top_k]
        )
