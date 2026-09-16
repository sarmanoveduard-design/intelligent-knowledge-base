import tempfile
import unittest
from pathlib import Path

from knowledge_base.document_registry import (
    calculate_sha256,
    register_document,
)


class DocumentRegistryTests(unittest.TestCase):
    def test_register_docx_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "example.docx"
            file_path.write_bytes(b"test document")

            record = register_document(file_path)

            self.assertEqual(record.file_name, "example.docx")
            self.assertEqual(record.extension, ".docx")
            self.assertEqual(record.size_bytes, 13)
            self.assertEqual(record.sha256, calculate_sha256(file_path))

    def test_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "example.txt"
            file_path.write_text("test", encoding="utf-8")

            with self.assertRaises(ValueError):
                register_document(file_path)


if __name__ == "__main__":
    unittest.main()
