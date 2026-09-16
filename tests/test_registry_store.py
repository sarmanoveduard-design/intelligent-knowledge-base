import tempfile
import unittest
from pathlib import Path

from knowledge_base.incoming_scanner import scan_incoming_folder
from knowledge_base.registry_store import (
    build_registry,
    load_registry,
    save_registry,
)


class RegistryStoreTests(unittest.TestCase):
    def test_builds_registry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            (folder / "first.docx").write_bytes(b"first")
            (folder / "second.pdf").write_bytes(b"second")

            records = scan_incoming_folder(folder)
            registry = build_registry(records)

            self.assertEqual(registry["document_count"], 2)
            self.assertEqual(len(registry["documents"]), 2)
            self.assertEqual(registry["exact_duplicates"], [])
            self.assertIn("generated_at", registry)

    def test_registry_contains_duplicate_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            (folder / "original.docx").write_bytes(b"same")
            (folder / "copy.docx").write_bytes(b"same")

            records = scan_incoming_folder(folder)
            registry = build_registry(records)

            self.assertEqual(len(registry["exact_duplicates"]), 1)
            self.assertEqual(
                registry["exact_duplicates"][0]["file_names"],
                ["copy.docx", "original.docx"],
            )

    def test_saves_and_loads_registry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            output_path = folder / "registry.json"

            registry = {
                "document_count": 1,
                "documents": [{"file_name": "example.docx"}],
            }

            save_registry(registry, output_path)
            loaded = load_registry(output_path)

            self.assertEqual(loaded, registry)


if __name__ == "__main__":
    unittest.main()
