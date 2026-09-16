from pathlib import Path

from docx import Document


incoming = Path("/app/data/incoming")
file_path = next(incoming.glob("*.docx"))

document = Document(file_path)
paragraphs = [
    paragraph
    for paragraph in document.paragraphs
    if paragraph.text.strip()
]

print(f"Файл: {file_path.name}")
print()
print("idx | style | bold | sizes | align | numbered | text")
print("-" * 160)

for index, paragraph in enumerate(paragraphs[:45]):
    runs = [
        run
        for run in paragraph.runs
        if run.text.strip()
    ]

    total_chars = sum(
        len(run.text)
        for run in runs
    )

    bold_chars = sum(
        len(run.text)
        for run in runs
        if run.bold is True
    )

    bold_percent = round(
        bold_chars / max(1, total_chars) * 100
    )

    sizes = sorted({
        round(run.font.size.pt, 1)
        for run in runs
        if run.font.size is not None
    })

    numbered = (
        paragraph._p.pPr is not None
        and paragraph._p.pPr.numPr is not None
    )

    style_name = (
        paragraph.style.name
        if paragraph.style is not None
        else "<нет>"
    )

    print(
        index,
        "|",
        style_name,
        "|",
        f"{bold_percent}%",
        "|",
        sizes,
        "|",
        paragraph.alignment,
        "|",
        numbered,
        "|",
        paragraph.text.strip()[:140],
    )
