"""Offline automatic comparison of existing retrieval reports."""
import argparse
from pathlib import Path

from knowledge_base.benchmark_config import BenchmarkConfigurationError
from knowledge_base.report_comparison import compare_reports


def main():
    parser = argparse.ArgumentParser(description="Compare reports from one corpus/gold snapshot pair")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--corpus-hash")
    parser.add_argument("--gold-hash")
    args = parser.parse_args()
    try:
        directories = [p for p in args.reports_root.iterdir() if p.is_dir() and (p / "experiment.json").is_file()]
        counts = compare_reports(directories, args.out, corpus_hash=args.corpus_hash, gold_hash=args.gold_hash)
    except BenchmarkConfigurationError as error:
        print(str(error))
        return 1
    except OSError:
        print("Comparison failed; check report files and output access permissions")
        return 1
    print(f"Compared={counts['compared']}; incompatible={counts['incompatible']}; invalid={counts['invalid']}")
    print("Created COMPARISON.md and comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
