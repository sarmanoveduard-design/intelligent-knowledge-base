"""Prepare HOLDOUT snapshots offline; never runs retrieval, models or providers."""
import argparse
from pathlib import Path

from knowledge_base.team_benchmark import TeamDatasetError
from knowledge_base.team_holdout_benchmark import MODES, prepare_team_holdout_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/team/holdout/corpus"))
    parser.add_argument("--gold", type=Path, default=Path("data/team/holdout/normalized/gold.jsonl"))
    parser.add_argument("--mode", choices=MODES, default="approved-only")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or Path("data/team/holdout/benchmark") / ("approved" if args.mode == "approved-only" else "all")
    try:
        manifest = prepare_team_holdout_benchmark(args.corpus_dir, args.gold, out, mode=args.mode)
    except TeamDatasetError as error:
        print(f"HOLDOUT validation failed: {error}")
        return 1
    except OSError:
        print("HOLDOUT preparation failed: check input/output files and permissions.")
        return 1
    print(f"Mode={manifest['mode']}; documents={manifest['document_count']}; chunks={manifest['chunk_count']}; "
          f"selected={manifest['selected_question_count']}/{manifest['source_question_count']}; unresolved refs=0")
    print(f"Corpus SHA256: {manifest['corpus_snapshot_hash']}")
    print(f"Gold SHA256: {manifest['gold_snapshot_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
