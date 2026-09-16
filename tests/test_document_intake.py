import tempfile
import unittest
from pathlib import Path

from knowledge_base.document_intake import (
    intake_new_document,
    intake_new_version,
)
from knowledge_base.document_versions import (
    VersionStatus,
    create_document,
    create_document_version,
)


VALID_SHA_A = "a" * 64


class DocumentIntakeTests(unittest.TestCase):
    def test_file_becomes_document_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "standard.docx"
            file_path.write_bytes(b"standard content")

            result = intake_new_document(
                file_path=file_path,
                title="Стандарт МОСТ",
                version_label="2026",
                owner="Экспертный совет",
                status=VersionStatus.ACTIVE,
            )

            self.assertEqual(
                result.file_record.file_name,
                "standard.docx",
            )

            self.assertEqual(
                result.document.title,
                "Стандарт МОСТ",
            )

            self.assertEqual(
                result.version.document_id,
                result.document.document_id,
            )

            self.assertEqual(
                result.version.source_sha256,
                result.file_record.sha256,
            )

            self.assertEqual(
                result.version.status,
                VersionStatus.ACTIVE,
            )

    def test_new_document_is_draft_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "lecture.pdf"
            file_path.write_bytes(b"lecture content")

            result = intake_new_document(
                file_path=file_path,
                title="Лекция №1",
                version_label="1.0",
            )

            self.assertEqual(
                result.version.status,
                VersionStatus.DRAFT,
            )


class ExistingDocumentVersionTests(unittest.TestCase):
    def test_new_active_version_supersedes_old_active(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "standard-2026.docx"
            file_path.write_bytes(b"new version content")

            document = create_document("Стандарт МОСТ")

            old_version = create_document_version(
                document=document,
                version_label="2025",
                source_sha256=VALID_SHA_A,
                status=VersionStatus.ACTIVE,
            )

            result = intake_new_version(
                file_path=file_path,
                document=document,
                version_label="2026",
                existing_versions=[old_version],
                status=VersionStatus.ACTIVE,
            )

            statuses = {
                version.version_label: version.status
                for version in result.versions
            }

            self.assertEqual(
                statuses["2025"],
                VersionStatus.SUPERSEDED,
            )

            self.assertEqual(
                statuses["2026"],
                VersionStatus.ACTIVE,
            )

    def test_new_draft_does_not_replace_active_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "standard-draft.docx"
            file_path.write_bytes(b"draft content")

            document = create_document("Стандарт МОСТ")

            active_version = create_document_version(
                document=document,
                version_label="2025",
                source_sha256=VALID_SHA_A,
                status=VersionStatus.ACTIVE,
            )

            result = intake_new_version(
                file_path=file_path,
                document=document,
                version_label="2026-draft",
                existing_versions=[active_version],
            )

            statuses = {
                version.version_label: version.status
                for version in result.versions
            }

            self.assertEqual(
                statuses["2025"],
                VersionStatus.ACTIVE,
            )

            self.assertEqual(
                statuses["2026-draft"],
                VersionStatus.DRAFT,
            )

    def test_same_file_cannot_be_added_as_new_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "same.docx"
            file_path.write_bytes(b"same content")

            first = intake_new_document(
                file_path=file_path,
                title="Стандарт МОСТ",
                version_label="1.0",
            )

            with self.assertRaises(ValueError):
                intake_new_version(
                    file_path=file_path,
                    document=first.document,
                    version_label="2.0",
                    existing_versions=[first.version],
                )


if __name__ == "__main__":
    unittest.main()
