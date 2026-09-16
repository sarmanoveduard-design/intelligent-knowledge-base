import unittest

from knowledge_base.chunker import Chunk
from knowledge_base.embeddings import Vector
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import (
    InMemoryVectorStore,
)


def make_chunk(
    index: int,
    text: str,
) -> Chunk:
    return Chunk(
        document_id="doc-1",
        version_id="ver-1",
        chunk_index=index,
        text=text,
        source_file="sample.docx",
        source_format="docx",
        block_indices=(index,),
        page_numbers=(),
        paragraph_indices=(index,),
        section_titles=(),
    )


class FakeSemanticEmbeddingProvider:
    @property
    def dimension(self) -> int:
        return 3

    def _embed(
        self,
        text: str,
    ) -> Vector:
        lowered = text.lower()

        if "кошка" in lowered:
            return (1.0, 0.0, 0.0)

        if "автомоб" in lowered:
            return (0.0, 1.0, 0.0)

        if "растен" in lowered:
            return (0.0, 0.0, 1.0)

        return (0.1, 0.1, 0.1)

    def embed_texts(
        self,
        texts,
    ) -> tuple[Vector, ...]:
        return tuple(
            self._embed(text)
            for text in texts
        )

    def embed_query(
        self,
        text: str,
    ) -> Vector:
        return self._embed(text)


class BrokenEmbeddingProvider(
    FakeSemanticEmbeddingProvider
):
    def embed_texts(
        self,
        texts,
    ) -> tuple[Vector, ...]:
        return ()


class RetrieverTests(unittest.TestCase):
    def test_indexes_all_chunks(self):
        provider = FakeSemanticEmbeddingProvider()

        store = InMemoryVectorStore(
            dimension=provider.dimension,
        )

        retriever = Retriever(
            embedding_provider=provider,
            vector_store=store,
        )

        retriever.index_chunks(
            (
                make_chunk(
                    0,
                    "Информация про кошку",
                ),
                make_chunk(
                    1,
                    "Информация про автомобиль",
                ),
            )
        )

        self.assertEqual(
            store.count,
            2,
        )

    def test_semantic_search_returns_best_chunk(self):
        provider = FakeSemanticEmbeddingProvider()

        store = InMemoryVectorStore(
            dimension=provider.dimension,
        )

        retriever = Retriever(
            embedding_provider=provider,
            vector_store=store,
        )

        retriever.index_chunks(
            (
                make_chunk(
                    0,
                    "Информация про кошку",
                ),
                make_chunk(
                    1,
                    "Информация про автомобиль",
                ),
                make_chunk(
                    2,
                    "Информация про растения",
                ),
            )
        )

        results = retriever.search(
            "Что известно про автомобиль?",
            top_k=2,
        )

        self.assertEqual(
            results[0].chunk.chunk_index,
            1,
        )

    def test_empty_query_is_rejected(self):
        provider = FakeSemanticEmbeddingProvider()

        retriever = Retriever(
            embedding_provider=provider,
            vector_store=InMemoryVectorStore(
                dimension=provider.dimension,
            ),
        )

        with self.assertRaises(ValueError):
            retriever.search("   ")

    def test_dimension_mismatch_is_rejected(self):
        provider = FakeSemanticEmbeddingProvider()

        with self.assertRaises(ValueError):
            Retriever(
                embedding_provider=provider,
                vector_store=InMemoryVectorStore(
                    dimension=5,
                ),
            )

    def test_broken_provider_does_not_partially_index(self):
        provider = BrokenEmbeddingProvider()

        store = InMemoryVectorStore(
            dimension=provider.dimension,
        )

        retriever = Retriever(
            embedding_provider=provider,
            vector_store=store,
        )

        with self.assertRaises(ValueError):
            retriever.index_chunks(
                (
                    make_chunk(
                        0,
                        "Первый текст",
                    ),
                    make_chunk(
                        1,
                        "Второй текст",
                    ),
                )
            )

        self.assertEqual(
            store.count,
            0,
        )


if __name__ == "__main__":
    unittest.main()
