import unittest

from knowledge_base.chunker import chunk_source_blocks
from knowledge_base.source_blocks import SourceBlock


def make_block(
    index: int,
    text: str,
    *,
    page: int | None = None,
    paragraph: int | None = None,
    document_id: str = "doc-1",
    version_id: str = "ver-1",
) -> SourceBlock:
    return SourceBlock(
        document_id=document_id,
        version_id=version_id,
        block_index=index,
        text=text,
        source_file="sample.docx",
        source_format="docx",
        page_number=page,
        paragraph_index=paragraph,
    )


class ChunkerTests(unittest.TestCase):
    def test_small_blocks_are_grouped(self):
        blocks = (
            make_block(
                0,
                "Первый абзац.",
                paragraph=0,
            ),
            make_block(
                1,
                "Второй абзац.",
                paragraph=1,
            ),
        )

        chunks = chunk_source_blocks(
            blocks,
            max_chars=200,
            overlap_pieces=0,
        )

        self.assertEqual(len(chunks), 1)

        self.assertEqual(
            chunks[0].text,
            "Первый абзац.\n\nВторой абзац.",
        )

        self.assertEqual(
            chunks[0].block_indices,
            (0, 1),
        )

    def test_large_block_is_split(self):
        original = (
            "Это длинный текст для проверки "
            "универсального разбиения документа. "
            * 20
        ).strip()

        blocks = (
            make_block(
                0,
                original,
                paragraph=4,
            ),
        )

        chunks = chunk_source_blocks(
            blocks,
            max_chars=120,
            overlap_pieces=0,
        )

        self.assertGreater(
            len(chunks),
            1,
        )

        for chunk in chunks:
            self.assertLessEqual(
                len(chunk.text),
                120,
            )

        reconstructed = " ".join(
            chunk.text
            for chunk in chunks
        )

        self.assertEqual(
            " ".join(reconstructed.split()),
            " ".join(original.split()),
        )

    def test_source_location_is_preserved(self):
        blocks = (
            make_block(
                0,
                "Страница один.",
                page=1,
            ),
            make_block(
                1,
                "Страница два.",
                page=2,
            ),
        )

        chunks = chunk_source_blocks(
            blocks,
            max_chars=200,
            overlap_pieces=0,
        )

        self.assertEqual(
            chunks[0].page_numbers,
            (1, 2),
        )

        self.assertEqual(
            chunks[0].block_indices,
            (0, 1),
        )

    def test_overlap_reuses_previous_piece(self):
        blocks = (
            make_block(
                0,
                "A" * 60,
            ),
            make_block(
                1,
                "B" * 60,
            ),
            make_block(
                2,
                "C" * 60,
            ),
        )

        chunks = chunk_source_blocks(
            blocks,
            max_chars=125,
            overlap_pieces=1,
        )

        self.assertEqual(
            len(chunks),
            2,
        )

        self.assertEqual(
            chunks[0].block_indices,
            (0, 1),
        )

        self.assertEqual(
            chunks[1].block_indices,
            (1, 2),
        )

    def test_different_documents_cannot_be_mixed(self):
        blocks = (
            make_block(
                0,
                "Документ один",
                document_id="doc-1",
            ),
            make_block(
                1,
                "Документ два",
                document_id="doc-2",
            ),
        )

        with self.assertRaises(ValueError):
            chunk_source_blocks(
                blocks,
            )


if __name__ == "__main__":
    unittest.main()
