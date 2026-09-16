from __future__ import annotations

import argparse
from pathlib import Path

from knowledge_base.incoming_scanner import scan_incoming_folder
from knowledge_base.registry_store import build_registry, save_registry


DEFAULT_INCOMING = Path("/app/data/incoming")
DEFAULT_REGISTRY = Path("/app/data/processed/document_registry.json")


def scan_command(
    incoming_path: Path = DEFAULT_INCOMING,
    registry_path: Path = DEFAULT_REGISTRY,
) -> None:
    records = scan_incoming_folder(incoming_path)
    registry = build_registry(records)

    save_registry(registry, registry_path)

    print("Сканирование завершено.")
    print(f"Документов найдено: {registry['document_count']}")
    print(f"Групп точных дубликатов: {len(registry['exact_duplicates'])}")
    print(f"Реестр сохранён: {registry_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Интеллектуальная база знаний"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "scan",
        help="Просканировать входящие PDF/DOCX и создать реестр",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan":
        scan_command()


if __name__ == "__main__":
    main()
