import unittest

from knowledge_base.docx_normalizer import NormalizedLine
from knowledge_base.structure_classifier import (
    StructuralType,
    classify_structure,
)


def make_line(
    index: int,
    text: str,
    *,
    bold: int = 0,
    sizes: tuple[float, ...] = (),
    alignment: str | None = None,
) -> NormalizedLine:
    return NormalizedLine(
        logical_index=index,
        paragraph_index=index,
        line_index=0,
        text=text,
        style="Normal",
        bold_percent=bold,
        font_sizes=sizes,
        alignment=alignment,
        numbered=False,
        formatting_reliable=True,
    )


class StructureClassifierTests(unittest.TestCase):
    def test_classifies_document_header_and_toc(self):
        lines = (
            make_line(
                0,
                "Редакция 22.06.2026",
                bold=100,
                alignment="RIGHT",
            ),
            make_line(
                1,
                "Нормативно-правовая база",
                bold=100,
                sizes=(14.0,),
            ),
            make_line(
                2,
                "обязательных дистанционных ПрМО на 2026 год",
                bold=100,
                sizes=(12.0,),
            ),
            make_line(
                3,
                "Содержание",
                bold=100,
            ),
            make_line(
                4,
                "КОДЕКСЫ",
            ),
            make_line(
                5,
                "Федеральные законы",
            ),
            make_line(
                6,
                "Ссылки на правовые нормативные документы и НСИ",
                bold=100,
                sizes=(14.0,),
                alignment="CENTER",
            ),
        )

        classified = classify_structure(lines)

        self.assertEqual(
            [item.structural_type for item in classified],
            [
                StructuralType.METADATA,
                StructuralType.TITLE,
                StructuralType.TITLE,
                StructuralType.TOC,
                StructuralType.TOC,
                StructuralType.TOC,
                StructuralType.SECTION,
            ],
        )

    def test_centered_body_heading_is_section(self):
        lines = (
            make_line(
                0,
                "Нормативно-правовая база",
                bold=100,
                sizes=(14.0,),
            ),
            make_line(
                1,
                "Содержание",
                bold=100,
            ),
            make_line(
                2,
                "КОДЕКСЫ",
            ),
            make_line(
                3,
                "Ссылки на правовые нормативные документы и НСИ",
                bold=100,
                sizes=(14.0,),
                alignment="CENTER",
            ),
            make_line(
                4,
                "КОДЕКСЫ",
                alignment="CENTER",
            ),
        )

        classified = classify_structure(lines)

        self.assertEqual(
            classified[4].structural_type,
            StructuralType.SECTION,
        )

    def test_legal_document_is_content_even_when_bold(self):
        lines = (
            make_line(
                0,
                "Нормативно-правовая база",
                bold=100,
                sizes=(14.0,),
            ),
            make_line(
                1,
                "Содержание",
                bold=100,
            ),
            make_line(
                2,
                "КОДЕКСЫ",
            ),
            make_line(
                3,
                "Ссылки на правовые нормативные документы и НСИ",
                bold=100,
                sizes=(14.0,),
                alignment="CENTER",
            ),
            make_line(
                4,
                "Приказ Минтруда России / Минздрава России от 31.12.2020 № 988н/1420н",
                bold=100,
            ),
        )

        classified = classify_structure(lines)

        self.assertEqual(
            classified[4].structural_type,
            StructuralType.CONTENT,
        )

    def test_short_centered_heading_is_section(self):
        lines = (
            make_line(
                0,
                "Нормативно-правовая база",
                bold=100,
                sizes=(14.0,),
            ),
            make_line(
                1,
                "Содержание",
            ),
            make_line(
                2,
                "КОДЕКСЫ",
            ),
            make_line(
                3,
                "Ссылки на правовые нормативные документы и НСИ",
                bold=100,
                sizes=(14.0,),
                alignment="CENTER",
            ),
            make_line(
                4,
                "Федеральные законы",
                bold=100,
                sizes=(12.0,),
                alignment="CENTER",
            ),
        )

        classified = classify_structure(lines)

        self.assertEqual(
            classified[4].structural_type,
            StructuralType.SECTION,
        )


if __name__ == "__main__":
    unittest.main()


class LegalHeadingDisambiguationTests(unittest.TestCase):
    def test_ministry_order_without_requisites_can_be_section(self):
        lines = (
            make_line(
                0,
                "Нормативно-правовая база",
                bold=100,
                sizes=(14.0,),
            ),
            make_line(
                1,
                "Содержание",
            ),
            make_line(
                2,
                "КОДЕКСЫ",
            ),
            make_line(
                3,
                "Ссылки на правовые нормативные документы и НСИ",
                bold=100,
                sizes=(14.0,),
                alignment="CENTER",
            ),
            make_line(
                4,
                "Приказ Минэнерго России",
                bold=100,
                sizes=(12.0,),
                alignment="CENTER",
            ),
        )

        classified = classify_structure(lines)

        self.assertEqual(
            classified[4].structural_type,
            StructuralType.SECTION,
        )

    def test_specific_order_with_date_and_number_is_content(self):
        lines = (
            make_line(
                0,
                "Нормативно-правовая база",
                bold=100,
                sizes=(14.0,),
            ),
            make_line(
                1,
                "Содержание",
            ),
            make_line(
                2,
                "КОДЕКСЫ",
            ),
            make_line(
                3,
                "Ссылки на правовые нормативные документы и НСИ",
                bold=100,
                sizes=(14.0,),
                alignment="CENTER",
            ),
            make_line(
                4,
                "Приказ Минтруда России от 31.12.2020 № 988н",
                bold=100,
                sizes=(12.0,),
                alignment="CENTER",
            ),
        )

        classified = classify_structure(lines)

        self.assertEqual(
            classified[4].structural_type,
            StructuralType.CONTENT,
        )
