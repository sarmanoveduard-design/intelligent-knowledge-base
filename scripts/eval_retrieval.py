"""Explicit real-run entry point; importing this module performs no API calls."""
import argparse
import os
from pathlib import Path

from knowledge_base.embedding_config import provider_from_environment
from knowledge_base.benchmark_config import BenchmarkConfigurationError, config_from_environment
from knowledge_base.document_representation import DocumentRepresentationError, validate_document_representation
from knowledge_base.bge_reranker import RerankerRuntimeError, reranker_from_environment
from knowledge_base.retrieval_benchmark import load_inputs, run_benchmark, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval benchmark, not LLM generation")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    failed = False
    try:
        inputs = load_inputs(args.corpus, args.gold)
        config = config_from_environment(top_k=args.top_k)
        rate = os.environ.get("BENCHMARK_COST_PER_MILLION_TOKENS", "")
        cost = float(rate) if rate else None
        representation = validate_document_representation(
            os.environ.get("BENCHMARK_DOCUMENT_REPRESENTATION", "plain")
        )
        # Validate inputs before constructing any API-backed provider.
        reranker = reranker_from_environment()
        if reranker is not None:
            reranker.prepare()
        provider = provider_from_environment() if config.retrieval_mode != "bm25" else None
        result = run_benchmark(provider, inputs, top_k=config.top_k, cost_per_million_tokens=cost,
                               document_representation=representation, retrieval_mode=config.retrieval_mode,
                               candidate_k=config.candidate_k, rrf_k=config.rrf_k, reranker=reranker)
        directory = write_report(result)
    except (DocumentRepresentationError, BenchmarkConfigurationError, RerankerRuntimeError) as error:
        print(str(error))
        return 1
    except Exception:
        failed = True
    if failed:
        print("Benchmark could not start or write artifacts; check input files, SDK and environment configuration.")
        return 1
    print(f"Report: reports/{directory.name}/REPORT.md")
    return 0 if result["experiment"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
