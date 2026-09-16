from pathlib import Path

from docx import Document


file_path = next(
    Path("/app/data/incoming").glob("*.docx")
)

document = Document(file_path)

differences = []

for index, paragraph in enumerate(document.paragraphs):
    source_text = " ".join(
        paragraph.text.split()
    )

    runs_text = " ".join(
        "".join(
            run.text
            for run in paragraph.runs
        ).split()
    )

    if source_text and source_text != runs_text:
        differences.append(
            (
                index,
                source_text,
                runs_text,
            )
        )

print(f"Файл: {file_path.name}")
print(
    "Абзацев с расхождением:",
    len(differences),
)

print()
print("--- Первые 15 расхождений ---")

for index, source_text, runs_text in differences[:15]:
    print()
    print("PARAGRAPH:", index)
    print("SOURCE:", source_text[:500])
    print("RUNS:  ", runs_text[:500])
