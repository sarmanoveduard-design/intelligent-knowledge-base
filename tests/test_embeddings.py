import unittest

from knowledge_base.embeddings import (
    EmbeddingProvider,
    Vector,
)


class FakeEmbeddingProvider:
    @property
    def dimension(self) -> int:
        return 3

    def embed_texts(
        self,
        texts,
    ) -> tuple[Vector, ...]:
        return tuple(
            (float(len(text)), 1.0, 0.0)
            for text in texts
        )

    def embed_query(
        self,
        text: str,
    ) -> Vector:
        return (
            float(len(text)),
            1.0,
            0.0,
        )


class EmbeddingInterfaceTests(unittest.TestCase):
    def test_provider_can_embed_documents_and_query(self):
        provider: EmbeddingProvider = FakeEmbeddingProvider()

        vectors = provider.embed_texts(
            (
                "Первый текст",
                "Второй текст",
            )
        )

        query = provider.embed_query(
            "Вопрос"
        )

        self.assertEqual(
            len(vectors),
            2,
        )

        self.assertEqual(
            provider.dimension,
            3,
        )

        self.assertEqual(
            len(query),
            3,
        )

    def test_provider_returns_immutable_vectors(self):
        provider = FakeEmbeddingProvider()

        vector = provider.embed_query(
            "Текст"
        )

        self.assertIsInstance(
            vector,
            tuple,
        )


if __name__ == "__main__":
    unittest.main()
