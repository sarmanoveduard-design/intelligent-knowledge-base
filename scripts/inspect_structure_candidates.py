from pathlib import Path
from collections import Counter

from knowledge_base.docx_normalizer import normalize_docx


file_path = next(
    Path("/app/data/incoming").glob("*.docx")
)

lines = normalize_docx(file_path)


def is_uppercase_candidate(text: str) -> bool:
    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    return (
        len(letters) >= 3
        and all(
            char.isupper()
            for char in letters
        )
    )


candidates = []

for line in lines:
    short = len(line.text) <= 120

    if (
        short
        and (
            line.alignment == "CENTER"
            or line.bold_percent == 100
            or is_uppercase_candidate(line.text)
        )
    ):
        candidates.append(line)


text_counts = Counter(
    line.text
    for line in lines
)


print(f"Файл: {file_path.name}")
print(f"Всего логических строк: {len(lines)}")
print(f"Кандидатов на структурные элементы: {len(candidates)}")

print()
print("--- КАНДИДАТЫ ---")

for line in candidates:
    repeated = text_counts[line.text]

    print(
        f"{line.logical_index:>3}",
        "| align:", line.alignment,
        "| bold:", line.bold_percent,
        "| size:", line.font_sizes,
        "| repeat:", repeated,
        "|",
        line.text,
    )
