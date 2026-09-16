from collections import Counter
from pathlib import Path

from knowledge_base.docx_normalizer import normalize_docx
from knowledge_base.structure_classifier import (
    StructuralType,
    classify_structure,
)


file_path = next(
    Path("/app/data/incoming").glob("*.docx")
)

lines = normalize_docx(file_path)
classified = classify_structure(lines)

counts = Counter(
    item.structural_type.value
    for item in classified
)

print(f"Файл: {file_path.name}")
print(f"Всего строк: {len(classified)}")
print()

print("--- СТАТИСТИКА ---")

for structural_type in StructuralType:
    print(
        f"{structural_type.value:>8}:",
        counts.get(structural_type.value, 0),
    )

print()
print("--- SECTION ---")

for item in classified:
    if item.structural_type == StructuralType.SECTION:
        line = item.line

        print(
            f"{line.logical_index:>3}",
            "| align:",
            line.alignment,
            "| bold:",
            line.bold_percent,
            "| size:",
            line.font_sizes,
            "|",
            line.text,
        )

print()
print("--- METADATA ---")

for item in classified:
    if item.structural_type == StructuralType.METADATA:
        print(
            item.line.logical_index,
            "|",
            item.line.text,
        )

print()
print("--- TITLE ---")

for item in classified:
    if item.structural_type == StructuralType.TITLE:
        print(
            item.line.logical_index,
            "|",
            item.line.text,
        )
