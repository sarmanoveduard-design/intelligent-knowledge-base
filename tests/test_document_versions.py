import unittest

from knowledge_base.document_versions import (
    VersionStatus,
    activate_version,
    create_document,
    create_document_version,
    get_active_version,
    validate_single_active_version,
)


VALID_SHA_A = "a" * 64
VALID_SHA_B = "b" * 64
VALID_SHA_C = "c" * 64


class DocumentVersionTests(unittest.TestCase):
    def test_creates_logical_document(self):
        document = create_document(
            title="Стандарт МОСТ",
            owner="Экспертный совет",
        )

        self.assertEqual(document.title, "Стандарт МОСТ")
        self.assertEqual(document.owner, "Экспертный совет")
        self.assertTrue(document.document_id)

    def test_one_document_can_have_multiple_versions(self):
        document = create_document("Стандарт МОСТ")

        version_1 = create_document_version(
            document=document,
            version_label="1.0",
            source_sha256=VALID_SHA_A,
            status=VersionStatus.SUPERSEDED,
        )

        version_2 = create_document_version(
            document=document,
            version_label="2.0",
            source_sha256=VALID_SHA_B,
            status=VersionStatus.ACTIVE,
        )

        self.assertEqual(version_1.document_id, document.document_id)
        self.assertEqual(version_2.document_id, document.document_id)
        self.assertNotEqual(version_1.version_id, version_2.version_id)
        self.assertNotEqual(
            version_1.source_sha256,
            version_2.source_sha256,
        )

    def test_rejects_invalid_sha256(self):
        document = create_document("Стандарт МОСТ")

        with self.assertRaises(ValueError):
            create_document_version(
                document=document,
                version_label="1.0",
                source_sha256="not-a-valid-sha",
            )


class VersionActivationTests(unittest.TestCase):
    def test_new_active_version_supersedes_previous_active(self):
        document = create_document("Стандарт МОСТ")

        version_1 = create_document_version(
            document=document,
            version_label="1.0",
            source_sha256=VALID_SHA_A,
            status=VersionStatus.ACTIVE,
        )

        version_2 = create_document_version(
            document=document,
            version_label="2.0",
            source_sha256=VALID_SHA_B,
            status=VersionStatus.DRAFT,
        )

        updated = activate_version(
            [version_1, version_2],
            version_2.version_id,
        )

        statuses = {
            version.version_label: version.status
            for version in updated
        }

        self.assertEqual(
            statuses["1.0"],
            VersionStatus.SUPERSEDED,
        )

        self.assertEqual(
            statuses["2.0"],
            VersionStatus.ACTIVE,
        )

    def test_other_document_versions_are_not_changed(self):
        first_document = create_document("Стандарт МОСТ")
        second_document = create_document("Другая инструкция")

        first_version = create_document_version(
            document=first_document,
            version_label="1.0",
            source_sha256=VALID_SHA_A,
            status=VersionStatus.DRAFT,
        )

        second_version = create_document_version(
            document=second_document,
            version_label="1.0",
            source_sha256=VALID_SHA_B,
            status=VersionStatus.ACTIVE,
        )

        updated = activate_version(
            [first_version, second_version],
            first_version.version_id,
        )

        second_updated = next(
            version
            for version in updated
            if version.version_id == second_version.version_id
        )

        self.assertEqual(
            second_updated.status,
            VersionStatus.ACTIVE,
        )

    def test_archived_version_cannot_be_activated(self):
        document = create_document("Стандарт МОСТ")

        archived = create_document_version(
            document=document,
            version_label="1.0",
            source_sha256=VALID_SHA_A,
            status=VersionStatus.ARCHIVED,
        )

        with self.assertRaises(ValueError):
            activate_version(
                [archived],
                archived.version_id,
            )


class ActiveVersionTests(unittest.TestCase):
    def test_returns_active_version(self):
        document = create_document("Стандарт МОСТ")

        old_version = create_document_version(
            document=document,
            version_label="1.0",
            source_sha256=VALID_SHA_A,
            status=VersionStatus.SUPERSEDED,
        )

        active_version = create_document_version(
            document=document,
            version_label="2.0",
            source_sha256=VALID_SHA_B,
            status=VersionStatus.ACTIVE,
        )

        result = get_active_version(
            [old_version, active_version],
            document.document_id,
        )

        self.assertEqual(
            result.version_id,
            active_version.version_id,
        )

    def test_returns_none_when_document_has_no_active_version(self):
        document = create_document("Стандарт МОСТ")

        draft = create_document_version(
            document=document,
            version_label="3.0",
            source_sha256=VALID_SHA_C,
            status=VersionStatus.DRAFT,
        )

        result = get_active_version(
            [draft],
            document.document_id,
        )

        self.assertIsNone(result)

    def test_rejects_multiple_active_versions(self):
        document = create_document("Стандарт МОСТ")

        first_active = create_document_version(
            document=document,
            version_label="1.0",
            source_sha256=VALID_SHA_A,
            status=VersionStatus.ACTIVE,
        )

        second_active = create_document_version(
            document=document,
            version_label="2.0",
            source_sha256=VALID_SHA_B,
            status=VersionStatus.ACTIVE,
        )

        with self.assertRaises(ValueError):
            validate_single_active_version(
                [first_active, second_active],
                document.document_id,
            )


if __name__ == "__main__":
    unittest.main()
