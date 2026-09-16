from collections import defaultdict
from pathlib import Path

from docx import Document

from knowledge_base.docx_normalizer import normalize_docx


file_path = next(
    Path("/app/data/incoming").glob("*.docx")
)

document = Document(file_path)
lines = normalize_docx(file_path)

by_paragraph = defaultdict(list)

for line in lines:
    by_paragraph[line.paragraph_index].append(line)


differences = []
fallback_paragraphs = set()

for paragraph_index, paragraph in enumerate(
    document.paragraphs
):
    source = " ".join(
        paragraph.text.split()
    )

    normalized = " ".join(
        line.text
        for line in by_paragraph.get(
            paragraph_index,
            [],
        )
    )

    normalized = " ".join(
        normalized.split()
    )

    if source != normalized:
        differences.append(
            (
                paragraph_index,
                source,
                normalized,
            )
        )

    if any(
        not line.formatting_reliable
        for line in by_paragraph.get(
            paragraph_index,
            [],
        )
    ):
        fallback_paragraphs.add(
            paragraph_index
        )


print(f"Файл: {file_path.name}")
print(
    "Абзацев с потерей текста:",
    len(differences),
)

print(
    "Абзацев с fallback форматирования:",
    len(fallback_paragraphs),
)

if differences:
    print()
    print("--- Расхождения ---")

    for index, source, normalized in differences[:10]:
        print()
        print("PARAGRAPH:", index)
        print("SOURCE:", source[:500])
        print("NORMALIZED:", normalized[:500])
