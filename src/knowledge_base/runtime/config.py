"""Runtime configuration validation only; no sufficiency decisions or env reads."""
from __future__ import annotations

from dataclasses import dataclass

from .models import _nonempty, _positive_int, _score


@dataclass(frozen=True)
class ScoreThreshold:
    """An explicit cutoff in a named scorer space, not a probability.

    scorer_identity is an opaque configured provider/model/revision identifier.
    The future policy must match both identity and type before using the cutoff.
    """
    value: float
    score_type: str
    scorer_identity: str

    def __post_init__(self):
        _score(self.value, "threshold value")
        _nonempty(self.score_type, "score_type")
        _nonempty(self.scorer_identity, "scorer_identity")


@dataclass(frozen=True)
class RuntimeConfig:
    """No unset threshold implies permission to generate.

    None means unconfigured; a future policy must handle that explicitly.
    This class validates consistency and never decides evidence sufficiency.
    Request filters may narrow access, but requests cannot override this policy.
    """
    candidate_top_k: int = 20
    final_top_k: int = 5
    context_max_chars: int = 12000
    partial_answers_enabled: bool = False
    minimum_evidence_count: int = 1
    dense_threshold: ScoreThreshold | None = None
    reranker_threshold: ScoreThreshold | None = None

    def __post_init__(self):
        for name in ("candidate_top_k", "final_top_k", "context_max_chars", "minimum_evidence_count"):
            _positive_int(getattr(self, name), name)
        if self.candidate_top_k < self.final_top_k:
            raise ValueError("candidate_top_k must be >= final_top_k")
        if self.minimum_evidence_count > self.final_top_k:
            raise ValueError("minimum_evidence_count cannot exceed final_top_k")
        if type(self.partial_answers_enabled) is not bool:
            raise ValueError("partial_answers_enabled must be boolean")
        for name in ("dense_threshold", "reranker_threshold"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ScoreThreshold):
                raise ValueError(f"{name} must be a ScoreThreshold or None")
