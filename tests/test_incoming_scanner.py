import tempfile
import unittest
from pathlib import Path

from knowledge_base.incoming_scanner import (
    find_exact_duplicates,
    scan_incoming_folder,
)


class IncomingScannerTests(unittest.TestCase):
    def test_scans_only_supported_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            (folder / "first.docx").write_bytes(b"docx")
            (folder / "second.pdf").write_bytes(b"pdf")
            (folder / "ignore.txt").write_text("ignore", encoding="utf-8")

            records = scan_incoming_folder(folder)

            self.assertEqual(len(records), 2)
            self.assertEqual(
                {record.file_name for record in records},
                {"first.docx", "second.pdf"},
            )

    def test_finds_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            same_content = b"same document content"

            (folder / "original.docx").write_bytes(same_content)
            (folder / "copy.docx").write_bytes(same_content)
            (folder / "different.pdf").write_bytes(b"different content")

            records = scan_incoming_folder(folder)
            duplicates = find_exact_duplicates(records)

            self.assertEqual(len(duplicates), 1)

            duplicate_group = next(iter(duplicates.values()))

            self.assertEqual(
                {record.file_name for record in duplicate_group},
                {"original.docx", "copy.docx"},
            )


if __name__ == "__main__":
    unittest.main()
