import unittest

from knowledge_base.chunker import Chunk
from knowledge_base.vector_store import (
    InMemoryVectorStore,
    cosine_similarity,
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


class VectorStoreTests(unittest.TestCase):
    def test_identical_vectors_have_similarity_one(self):
        score = cosine_similarity(
            (1.0, 2.0, 3.0),
            (1.0, 2.0, 3.0),
        )

        self.assertAlmostEqual(
            score,
            1.0,
        )

    def test_orthogonal_vectors_have_similarity_zero(self):
        score = cosine_similarity(
            (1.0, 0.0),
            (0.0, 1.0),
        )

        self.assertAlmostEqual(
            score,
            0.0,
        )

    def test_zero_vector_returns_zero_similarity(self):
        score = cosine_similarity(
            (0.0, 0.0),
            (1.0, 0.0),
        )

        self.assertEqual(
            score,
            0.0,
        )

    def test_store_returns_most_similar_chunk_first(self):
        store = InMemoryVectorStore(
            dimension=3,
        )

        store.add(
            make_chunk(
                0,
                "Информация про кошек",
            ),
            (1.0, 0.0, 0.0),
        )

        store.add(
            make_chunk(
                1,
                "Информация про автомобили",
            ),
            (0.0, 1.0, 0.0),
        )

        store.add(
            make_chunk(
                2,
                "Информация про растения",
            ),
            (0.0, 0.0, 1.0),
        )

        results = store.search(
            (0.9, 0.1, 0.0),
            top_k=2,
        )

        self.assertEqual(
            results[0].chunk.chunk_index,
            0,
        )

        self.assertEqual(
            len(results),
            2,
        )

    def test_store_rejects_wrong_vector_dimension(self):
        store = InMemoryVectorStore(
            dimension=3,
        )

        with self.assertRaises(ValueError):
            store.add(
                make_chunk(
                    0,
                    "Текст",
                ),
                (1.0, 2.0),
            )

    def test_add_many_is_atomic_on_dimension_error(self):
        store = InMemoryVectorStore(
            dimension=3,
        )

        chunks = (
            make_chunk(
                0,
                "Первый",
            ),
            make_chunk(
                1,
                "Второй",
            ),
        )

        vectors = (
            (1.0, 0.0, 0.0),
            (1.0, 0.0),
        )

        with self.assertRaises(ValueError):
            store.add_many(
                chunks,
                vectors,
            )

        self.assertEqual(
            store.count,
            0,
        )


if __name__ == "__main__":
    unittest.main()
