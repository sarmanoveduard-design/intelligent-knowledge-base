"""Prepare internal JSON snapshots from team files. Never imports API providers."""
import argparse
from pathlib import Path

from knowledge_base.team_benchmark import TeamDatasetError, prepare_team_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and import team retrieval benchmark files")
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = prepare_team_benchmark(args.corpus_dir, args.gold, args.out)
    except TeamDatasetError as error:
        print(f"Dataset validation failed: {error}")
        return 1
    except OSError:
        print("Dataset preparation failed: check input/output files and access permissions.")
        return 1
    print(f"Prepared documents={manifest['document_count']}, chunks={manifest['chunk_count']}, "
          f"queries={manifest['query_count']}; unresolved refs=0")
    print(f"Corpus SHA256: {manifest['corpus_snapshot_hash']}")
    print(f"Gold SHA256: {manifest['gold_snapshot_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
