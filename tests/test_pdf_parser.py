import tempfile
import unittest
from pathlib import Path

from knowledge_base.pdf_parser import parse_pdf


def create_simple_pdf(
    file_path: Path,
    page_texts: list[str],
) -> None:
    objects: list[bytes] = []

    font_object_number = 3 + len(page_texts) * 2

    page_refs = " ".join(
        f"{3 + index * 2} 0 R"
        for index in range(len(page_texts))
    )

    objects.append(
        (
            "<< /Type /Catalog "
            "/Pages 2 0 R >>"
        ).encode("ascii")
    )

    objects.append(
        (
            "<< /Type /Pages "
            f"/Kids [{page_refs}] "
            f"/Count {len(page_texts)} >>"
        ).encode("ascii")
    )

    for index, text in enumerate(page_texts):
        page_object_number = 3 + index * 2
        content_object_number = page_object_number + 1

        objects.append(
            (
                "<< /Type /Page "
                "/Parent 2 0 R "
                "/MediaBox [0 0 612 792] "
                f"/Contents {content_object_number} 0 R "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                ">>"
            ).encode("ascii")
        )

        escaped_text = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

        stream = (
            "BT\n"
            "/F1 12 Tf\n"
            "72 720 Td\n"
            f"({escaped_text}) Tj\n"
            "ET\n"
        ).encode("ascii")

        objects.append(
            (
                f"<< /Length {len(stream)} >>\n"
                "stream\n"
            ).encode("ascii")
            + stream
            + b"endstream"
        )

    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for object_number, content in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(
            f"{object_number} 0 obj\n".encode("ascii")
        )
        pdf.extend(content)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)

    pdf.extend(
        f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    )
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf.extend(
            f"{offset:010d} 00000 n \n".encode("ascii")
        )

    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )

    file_path.write_bytes(pdf)


class PdfParserTests(unittest.TestCase):
    def test_extracts_text_and_page_numbers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "law.pdf"

            create_simple_pdf(
                file_path,
                [
                    "First page text",
                    "Second page text",
                ],
            )

            parsed = parse_pdf(file_path)

            self.assertEqual(
                parsed.file_name,
                "law.pdf",
            )

            self.assertEqual(
                len(parsed.pages),
                2,
            )

            self.assertEqual(
                parsed.pages[0].page_number,
                1,
            )

            self.assertIn(
                "First page text",
                parsed.pages[0].text,
            )

            self.assertEqual(
                parsed.pages[1].page_number,
                2,
            )

            self.assertIn(
                "Second page text",
                parsed.full_text,
            )

    def test_rejects_broken_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "broken.pdf"
            file_path.write_bytes(
                b"this is not a real PDF"
            )

            with self.assertRaises(ValueError):
                parse_pdf(file_path)

    def test_rejects_wrong_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "document.txt"
            file_path.write_text(
                "text",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                parse_pdf(file_path)


if __name__ == "__main__":
    unittest.main()
