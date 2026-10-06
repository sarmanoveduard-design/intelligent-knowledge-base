"""Candidate retrieval -> optional reranker -> final top_k, outside Retriever."""
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from knowledge_base.benchmark_config import RetrievalConfig
from knowledge_base.bm25 import BM25Index
from knowledge_base.chunker import Chunk
from knowledge_base.fusion import reciprocal_rank_fusion, reciprocal_rank_fusion_scores
from knowledge_base.retriever import Retriever


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class CandidateResult:
    candidates: tuple[Candidate, ...]
    diagnostic_scores: dict[str, float]
    union_size: int


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def rerank(self, query: str, candidates: tuple[Candidate, ...]) -> tuple[Candidate, ...]: ...


def rerank_candidates(query, result: CandidateResult, *, reranker: Reranker | None = None):
    candidates = result.candidates
    if reranker is not None:
        reordered = tuple(reranker.rerank(query, candidates))
        originals = {c.chunk_id: c for c in candidates}
        if (len(reordered) != len(candidates)
                or {c.chunk_id for c in reordered} != set(originals)
                or any(c.chunk is not originals[c.chunk_id].chunk or not isfinite(c.score) for c in reordered)):
            raise ValueError("Reranker must reorder the candidate pool without adding or removing chunks")
        candidates = reordered
    return candidates


def final_candidates(query, result: CandidateResult, *, top_k: int, reranker: Reranker | None = None):
    return rerank_candidates(query, result, reranker=reranker)[:top_k]


class CandidateRetriever:
    def __init__(self, *, chunks, chunk_ids, config: RetrievalConfig,
                 dense: Retriever | None = None, document_texts=None):
        self.chunks, self.chunk_ids = tuple(chunks), tuple(chunk_ids)
        self.config, self.dense = config, dense
        self.document_texts = document_texts
        self.bm25 = None
        self.limit = config.effective_candidate_k(len(chunks))

    def index(self):
        if self.config.retrieval_mode != "bm25":
            if self.dense is None:
                raise ValueError("Dense retrieval requires an embedding provider")
            if self.document_texts is None:
                self.dense.index_chunks(self.chunks)
            elif self.chunks:
                vectors = self.dense.embedding_provider.embed_texts(self.document_texts)
                self.dense.vector_store.add_many(self.chunks, vectors)
        if self.config.retrieval_mode != "dense":
            texts = self.document_texts if self.document_texts is not None else tuple(c.text for c in self.chunks)
            self.bm25 = BM25Index(texts)  # fixed k1=1.5, b=0.75

    def retrieve(self, query: str) -> CandidateResult:
        rankings, score_maps = [], []
        if self.config.retrieval_mode != "bm25":
            # Preserve legacy full-corpus cosine diagnostics, including hard negatives.
            results = self.dense.search(query, top_k=max(1, len(self.chunks)))
            ranking = [r.chunk.chunk_index for r in results]
            rankings.append(ranking[:self.limit])
            score_maps.append({self.chunk_ids[r.chunk.chunk_index]: r.score for r in results})
        if self.config.retrieval_mode != "dense":
            scores = self.bm25.scores(query)
            ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            rankings.append(ranking[:self.limit])
            score_maps.append(dict(zip(self.chunk_ids, scores)))
        if self.config.retrieval_mode == "hybrid_rrf":
            fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k, limit=max(1, self.limit))
            fusion_scores = reciprocal_rank_fusion_scores(rankings, k=self.config.rrf_k)
            diagnostic = {self.chunk_ids[i]: fusion_scores.get(i, 0.0) for i in range(len(self.chunks))}
            union_size = len(fusion_scores)
        else:
            fused, diagnostic = rankings[0], score_maps[0]
            union_size = len(fused)
        candidates = tuple(Candidate(self.chunk_ids[i], self.chunks[i], diagnostic[self.chunk_ids[i]]) for i in fused)
        return CandidateResult(candidates, diagnostic, union_size)


def candidate_metrics(ranked_ids, positives, *, effective_k: int, corpus_size: int):
    positive = set(positives)
    metrics = {}
    for k in (10, 20, 50):
        metrics[f"Candidate Recall@{k}"] = (
            None if effective_k < k and effective_k < corpus_size
            else len(positive & set(ranked_ids[:k])) / len(positive)
        )
    metrics["Candidate Recall@pool"] = len(positive & set(ranked_ids)) / len(positive)
    return metrics
