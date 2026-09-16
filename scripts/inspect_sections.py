from pathlib import Path

from knowledge_base.docx_normalizer import normalize_docx
from knowledge_base.section_builder import build_sections
from knowledge_base.structure_classifier import classify_structure


file_path = next(
    Path("/app/data/incoming").glob("*.docx")
)

lines = normalize_docx(file_path)
classified = classify_structure(lines)
sections = build_sections(classified)

print(f"Файл: {file_path.name}")
print(f"Всего разделов: {len(sections)}")
print()

print("--- РАЗДЕЛЫ ---")

for section in sections:
    print(
        f"{section.section_index:>2}",
        "| title:",
        section.title,
        "| content lines:",
        len(section.lines),
        "| chars:",
        len(section.body_text),
    )
