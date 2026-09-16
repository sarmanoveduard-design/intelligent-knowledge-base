from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from knowledge_base.chunker import Chunk
from knowledge_base.embeddings import EmbeddingProvider
from knowledge_base.vector_store import (
    InMemoryVectorStore,
    SearchResult,
)


@dataclass
class Retriever:
    embedding_provider: EmbeddingProvider
    vector_store: InMemoryVectorStore

    def __post_init__(self) -> None:
        if (
            self.embedding_provider.dimension
            != self.vector_store.dimension
        ):
            raise ValueError(
                "Размерность embedding provider "
                "и vector store должна совпадать"
            )

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        if not chunks:
            return

        texts = tuple(
            chunk.text
            for chunk in chunks
        )

        vectors = self.embedding_provider.embed_texts(
            texts
        )

        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedding provider вернул "
                "неверное количество векторов"
            )

        self.vector_store.add_many(
            chunks,
            vectors,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[SearchResult, ...]:
        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "Поисковый запрос не может быть пустым"
            )

        query_vector = (
            self.embedding_provider.embed_query(
                clean_query
            )
        )

        return self.vector_store.search(
            query_vector,
            top_k=top_k,
        )
