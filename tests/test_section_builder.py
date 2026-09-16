import unittest

from knowledge_base.docx_normalizer import NormalizedLine
from knowledge_base.section_builder import build_sections
from knowledge_base.structure_classifier import (
    ClassifiedLine,
    StructuralType,
)


def make_line(
    index: int,
    text: str,
) -> NormalizedLine:
    return NormalizedLine(
        logical_index=index,
        paragraph_index=index,
        line_index=0,
        text=text,
        style="Normal",
        bold_percent=0,
        font_sizes=(),
        alignment=None,
        numbered=False,
        formatting_reliable=True,
    )


def classified(
    index: int,
    text: str,
    structural_type: StructuralType,
) -> ClassifiedLine:
    return ClassifiedLine(
        line=make_line(index, text),
        structural_type=structural_type,
    )


class SectionBuilderTests(unittest.TestCase):
    def test_groups_content_under_section(self):
        items = (
            classified(
                0,
                "Нормативно-правовая база",
                StructuralType.TITLE,
            ),
            classified(
                1,
                "Федеральные законы",
                StructuralType.SECTION,
            ),
            classified(
                2,
                "Федеральный закон №1",
                StructuralType.CONTENT,
            ),
            classified(
                3,
                "Федеральный закон №2",
                StructuralType.CONTENT,
            ),
            classified(
                4,
                "Указы Президента РФ",
                StructuralType.SECTION,
            ),
            classified(
                5,
                "Указ №1",
                StructuralType.CONTENT,
            ),
        )

        sections = build_sections(items)

        self.assertEqual(len(sections), 2)

        self.assertEqual(
            sections[0].title,
            "Федеральные законы",
        )

        self.assertEqual(
            len(sections[0].lines),
            2,
        )

        self.assertEqual(
            sections[1].title,
            "Указы Президента РФ",
        )

        self.assertEqual(
            len(sections[1].lines),
            1,
        )

    def test_repeated_section_titles_are_not_merged(self):
        items = (
            classified(
                0,
                "Приказы Минтруда России",
                StructuralType.SECTION,
            ),
            classified(
                1,
                "Первый приказ",
                StructuralType.CONTENT,
            ),
            classified(
                2,
                "Приказы Минтруда России",
                StructuralType.SECTION,
            ),
            classified(
                3,
                "Второй приказ",
                StructuralType.CONTENT,
            ),
        )

        sections = build_sections(items)

        self.assertEqual(len(sections), 2)

        self.assertEqual(
            sections[0].title,
            sections[1].title,
        )

        self.assertNotEqual(
            sections[0].section_index,
            sections[1].section_index,
        )

    def test_content_before_first_section_is_preserved(self):
        items = (
            classified(
                0,
                "Вводный текст",
                StructuralType.CONTENT,
            ),
            classified(
                1,
                "КОДЕКСЫ",
                StructuralType.SECTION,
            ),
            classified(
                2,
                "Трудовой кодекс",
                StructuralType.CONTENT,
            ),
        )

        sections = build_sections(items)

        self.assertEqual(len(sections), 2)

        self.assertIsNone(
            sections[0].title,
        )

        self.assertEqual(
            sections[0].body_text,
            "Вводный текст",
        )

    def test_metadata_title_and_toc_are_not_body_content(self):
        items = (
            classified(
                0,
                "Редакция 22.06.2026",
                StructuralType.METADATA,
            ),
            classified(
                1,
                "Нормативно-правовая база",
                StructuralType.TITLE,
            ),
            classified(
                2,
                "Содержание",
                StructuralType.TOC,
            ),
            classified(
                3,
                "КОДЕКСЫ",
                StructuralType.SECTION,
            ),
            classified(
                4,
                "Трудовой кодекс",
                StructuralType.CONTENT,
            ),
        )

        sections = build_sections(items)

        self.assertEqual(len(sections), 1)

        self.assertEqual(
            sections[0].full_text,
            "КОДЕКСЫ\nТрудовой кодекс",
        )


if __name__ == "__main__":
    unittest.main()
