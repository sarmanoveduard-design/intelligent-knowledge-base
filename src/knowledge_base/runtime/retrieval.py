"""Authorized candidate retrieval and neutral reranking, without generation."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Protocol

from knowledge_base.embeddings import EmbeddingProvider

from .config import RuntimeConfig
from .indexing import validate_embedding_identity
from .models import (
    AccessContext, ChunkIdentity, EvidenceChunk, IndexIdentity, RetrievalFilters,
    RuntimeEvidence, _nonempty, _score, compose_retrieval_filters,
)
from .protocols import IndexingRepository, RuntimeReranker, VectorIndex
from .vector_index import chunk_is_visible, validate_vector


class RetrievalError(RuntimeError):
    """Safe diagnostic code; provider exceptions and source text are not exposed."""


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[RuntimeEvidence, ...]
    visible_version_count: int
    candidate_count: int
    query_embedded: bool
    empty_reason: str | None = None


def _validate_reranked(
    original: tuple[RuntimeEvidence, ...], result: tuple[RuntimeEvidence, ...],
    reranker: RuntimeReranker,
) -> None:
    sources = {x.chunk.identity: x for x in original}
    if (len(result) != len(original) or any(not isinstance(x, RuntimeEvidence) for x in result)
            or len({x.chunk.identity for x in result}) != len(result)
            or {x.chunk.identity for x in result} != set(sources)):
        raise RetrievalError('invalid_reranker_candidates')
    for item in result:
        previous = sources[item.chunk.identity]
        if (item.chunk != previous.chunk or item.retrieval_score != previous.retrieval_score
                or item.retrieval_score_type != previous.retrieval_score_type
                or item.retrieval_scorer_identity != previous.retrieval_scorer_identity
                or item.reranker_score_type != reranker.score_type
                or item.reranker_scorer_identity != reranker.scorer_identity):
            raise RetrievalError('invalid_reranker_evidence')
        _score(item.reranker_score, 'reranker_score')
        _nonempty(item.reranker_score_type, 'reranker_score_type')
        _nonempty(item.reranker_scorer_identity, 'reranker_scorer_identity')


@dataclass(frozen=True)
class RetrievalPipeline:
    repository: IndexingRepository
    index: VectorIndex
    embeddings: EmbeddingProvider
    reranker: RuntimeReranker
    identity: IndexIdentity
    config: RuntimeConfig = RuntimeConfig()

    def retrieve(
        self, question: str, access: AccessContext, *, filters: RetrievalFilters | None = None,
        config: RuntimeConfig | None = None,
    ) -> RetrievalResult:
        reason = 'retrieval_validation_failed'
        try:
            _nonempty(question, 'question')
            if self.index.identity != self.identity:
                raise ValueError('incompatible_index_identity')
            validate_embedding_identity(self.embeddings, self.identity)
            effective = compose_retrieval_filters(access, filters or RetrievalFilters())
            config = config if config is not None else self.config
            versions = self.repository.list_visible_versions(access, filters=effective)
            allowed = frozenset(v.identity for v in versions)
            if not allowed:
                return RetrievalResult((), 0, 0, False, 'no_visible_versions')
            authorized = effective.narrow(RetrievalFilters(
                allowed_document_ids=frozenset(v.document_id for v in allowed),
                allowed_version_ids=frozenset(v.version_id for v in allowed),
            ))
            reason = 'query_embedding_failed'
            query = validate_vector(self.embeddings.embed_query(question), self.identity.vector_dimension)
            reason = 'vector_search_failed'
            candidates = tuple(self.index.search(query, access=access, filters=authorized,
                top_k=config.candidate_top_k, expected_identity=self.identity))
            if (len(candidates) > config.candidate_top_k
                    or any(not isinstance(x, RuntimeEvidence) for x in candidates)
                    or len({x.chunk.identity for x in candidates}) != len(candidates)):
                raise RetrievalError('invalid_vector_candidates')
            # A registry change during search must not leak text to the reranker.
            candidates = tuple(x for x in candidates if chunk_is_visible(
                self.repository, x.chunk, access, authorized, allowed, self.identity,
                self.index.get_manifest(x.chunk.identity.snapshot)))
            if not candidates:
                return RetrievalResult((), len(allowed), 0, True, 'no_ready_candidates')
            reason = 'reranking_failed'
            reranked = tuple(self.reranker.rerank(question, candidates))
            _validate_reranked(candidates, reranked, self.reranker)
            # The reranker owns candidate order. Check visibility again before
            # slicing final_top_k, including rights/publication changes mid-call.
            allowed_now = frozenset(v.identity for v in self.repository.list_visible_versions(access, filters=authorized))
            verified = tuple(x for x in reranked if chunk_is_visible(
                self.repository, x.chunk, access, authorized, allowed_now, self.identity,
                self.index.get_manifest(x.chunk.identity.snapshot)))
            evidence = verified[:config.final_top_k]
            return RetrievalResult(evidence, len(allowed), len(candidates), True,
                                   None if evidence else 'visibility_changed')
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError(reason) from None


@dataclass(frozen=True)
class _ScoredChunk:
    chunk_id: ChunkIdentity
    chunk: EvidenceChunk
    score: float


class _ExistingReranker(Protocol):
    """Duck-typed existing BGE API; no benchmark Candidate or config imports."""
    name: str

    @property
    def metadata(self) -> dict: ...

    def rerank(self, query: str, candidates: tuple[_ScoredChunk, ...]) -> tuple[_ScoredChunk, ...]: ...


class BGERuntimeAdapter:
    """Wrap an explicitly constructed existing BGEReranker instance.

    Composition outside runtime supplies its config/backend. This module never
    imports its benchmark configuration or triggers model loading on import.
    Scores are raw logits, never probabilities; full source identity is preserved.
    """
    score_type = 'bge-reranker-logit'

    def __init__(self, reranker: _ExistingReranker):
        self._reranker = reranker

    @property
    def name(self) -> str:
        return self._reranker.name

    @property
    def scorer_identity(self) -> str:
        metadata = self._reranker.metadata
        return json.dumps({
            'name': self.name,
            'model': metadata['reranker_model'],
            'revision': metadata.get('reranker_resolved_revision') or metadata['reranker_revision'],
            'max_length': metadata['reranker_max_length'],
            'precision': metadata['reranker_precision'],
        }, sort_keys=True, separators=(',', ':'))

    def rerank(self, question: str, candidates: tuple[RuntimeEvidence, ...]) -> tuple[RuntimeEvidence, ...]:
        if not candidates:
            return ()
        sources = {x.chunk.identity: x for x in candidates}
        try:
            scored = tuple(self._reranker.rerank(question, tuple(
                _ScoredChunk(x.chunk.identity, x.chunk, x.retrieval_score) for x in candidates)))
            if (len(scored) != len(candidates) or len({x.chunk_id for x in scored}) != len(scored)
                    or {x.chunk_id for x in scored} != set(sources)):
                raise ValueError('invalid_reranker_candidates')
            result = []
            for item in scored:
                original = sources[item.chunk_id]
                if item.chunk != original.chunk:
                    raise ValueError('invalid_reranker_source')
                _score(item.score, 'reranker_score')
                result.append(replace(original, reranker_score=item.score,
                    reranker_score_type=self.score_type, reranker_scorer_identity=self.scorer_identity))
            return tuple(result)
        except Exception:
            raise RetrievalError('reranking_failed') from None
