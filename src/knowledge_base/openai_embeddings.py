"""OpenAI embeddings adapter. No dotenv loading and no response/error logging."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from typing import Any, Sequence

from knowledge_base.embeddings import EmbeddingIdentity, Vector


class OpenAIEmbeddingError(RuntimeError):
    """Safe public error; never contains SDK messages or response bodies."""


@dataclass(frozen=True)
class OpenAIEmbeddingProvider:
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    timeout: float = 60.0
    batch_size: int = 64
    client: Any = field(default=None, repr=False, compare=False)
    _stats: dict = field(default_factory=lambda: {
        "api_calls": 0, "latency_seconds": 0.0,
        "total_tokens": None, "prompt_tokens": None, "error_count": 0,
    }, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        native = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
        if self.model not in native:
            raise ValueError("Unsupported OpenAI embedding model")
        dimension = native[self.model] if self.dimensions is None else self.dimensions
        if type(dimension) is not int or not 1 <= dimension <= native[self.model]:
            raise ValueError("Invalid OpenAI embedding dimensions")
        if not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Embedding timeout must be positive and finite")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 2048:
            raise ValueError("Embedding batch size must be between 1 and 2048")
        object.__setattr__(self, "dimensions", dimension)
        if self.client is None:
            key = os.environ.get("OPENAI_API_KEY")
            if not key or not key.strip():
                raise OpenAIEmbeddingError("OPENAI_API_KEY environment variable is required")
            failed = False
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key, timeout=self.timeout, max_retries=0,
                                base_url="https://api.openai.com/v1")
                object.__setattr__(self, "client", client)
            except Exception:
                failed = True
            # Raise outside the handler: even exception context cannot expose SDK secrets.
            if failed:
                raise OpenAIEmbeddingError("OpenAI embedding client initialization failed")

    @property
    def dimension(self) -> int:
        return self.dimensions

    @property
    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity("openai", self.model, self.dimension)

    @property
    def telemetry(self) -> dict:
        return dict(self._stats)

    def embed_texts(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        if not texts:
            return ()
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding input must contain nonempty strings")
        vectors = []
        for offset in range(0, len(texts), self.batch_size):
            batch = list(texts[offset:offset + self.batch_size])
            started = perf_counter()
            self._stats["api_calls"] += 1
            error = None
            try:
                response = self.client.embeddings.create(
                    model=self.model, dimensions=self.dimension,
                    input=batch, encoding_format="float", timeout=self.timeout,
                )
            except Exception:
                error = "OpenAI embedding API request failed"
            finally:
                self._stats["latency_seconds"] += perf_counter() - started
            if error is not None:
                self._stats["error_count"] += 1
                raise OpenAIEmbeddingError(error)

            error = "Invalid OpenAI embedding response"
            try:
                usage = getattr(response, "usage", None)
                for name in ("prompt_tokens", "total_tokens"):
                    value = getattr(usage, name, None)
                    if type(value) is int and value >= 0:
                        self._stats[name] = (self._stats[name] or 0) + value
                if len(response.data) != len(batch):
                    error = "OpenAI returned incorrect vector count"
                    raise ValueError
                indices = [item.index for item in response.data]
                if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(batch))):
                    error = "OpenAI returned invalid vector indices"
                    raise ValueError
                ordered = sorted(response.data, key=lambda item: item.index)
                batch_vectors = tuple(tuple(float(v) for v in item.embedding) for item in ordered)
                if any(len(v) != self.dimension for v in batch_vectors):
                    error = "OpenAI returned incorrect vector dimension"
                    raise ValueError
                if any(not isfinite(value) for vector in batch_vectors for value in vector):
                    error = "OpenAI returned nonfinite vector values"
                    raise ValueError
            except Exception:
                self._stats["error_count"] += 1
            else:
                vectors.extend(batch_vectors)
                continue
            raise OpenAIEmbeddingError(error)
        return tuple(vectors)

    def embed_query(self, text: str) -> Vector:
        return self.embed_texts((text,))[0]
