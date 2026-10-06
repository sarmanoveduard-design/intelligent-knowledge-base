"""Format-independent benchmark records. Adapters own source format parsing."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    document_id: str
    section_ref: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCase:
    text: str
    positives: tuple[str, ...]
    hard_negatives: tuple[str, ...] = ()
    query_id: str = ""
    difficulty: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
