"""Validated benchmark configuration, without API/provider construction."""
from dataclasses import dataclass
import os
import re


class BenchmarkConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class RetrievalConfig:
    retrieval_mode: str = "dense"
    candidate_k: int = 50
    rrf_k: int = 60
    top_k: int = 10

    def __post_init__(self):
        if self.retrieval_mode not in ("dense", "bm25", "hybrid_rrf"):
            raise BenchmarkConfigurationError("Invalid retrieval mode; expected dense, bm25 or hybrid_rrf")
        if type(self.top_k) is not int or self.top_k < 10:
            raise BenchmarkConfigurationError("top_k must be at least 10 for Recall@10")
        if type(self.candidate_k) is not int or self.candidate_k < self.top_k:
            raise BenchmarkConfigurationError("candidate_k must be an integer at least top_k")
        if type(self.rrf_k) is not int or self.rrf_k <= 0:
            raise BenchmarkConfigurationError("rrf_k must be a positive integer")

    def effective_candidate_k(self, corpus_size):
        return min(self.candidate_k, corpus_size)


def validate_code_overrides():
    sha = os.environ.get("BENCHMARK_CODE_COMMIT_SHA")
    dirty = os.environ.get("BENCHMARK_CODE_DIRTY")
    if sha is not None and not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", sha):
        raise BenchmarkConfigurationError("BENCHMARK_CODE_COMMIT_SHA must be a full hex commit SHA")
    if dirty is not None and dirty not in ("true", "false"):
        raise BenchmarkConfigurationError("BENCHMARK_CODE_DIRTY must be true or false")
    return sha.lower() if sha is not None else None, (dirty == "true") if dirty is not None else None


def config_from_environment(*, top_k=None):
    failed = False
    try:
        config = RetrievalConfig(
            os.environ.get("BENCHMARK_RETRIEVAL_MODE", "dense"),
            int(os.environ.get("BENCHMARK_CANDIDATE_K", "50")),
            int(os.environ.get("BENCHMARK_RRF_K", "60")),
            top_k if top_k is not None else int(os.environ.get("BENCHMARK_TOP_K", "10")),
        )
    except BenchmarkConfigurationError:
        raise
    except (ValueError, OverflowError):
        failed = True
    if failed:
        raise BenchmarkConfigurationError("Invalid numeric benchmark configuration")
    validate_code_overrides()
    return config
