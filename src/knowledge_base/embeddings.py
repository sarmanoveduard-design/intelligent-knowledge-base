from __future__ import annotations

from typing import Protocol, Sequence
from dataclasses import dataclass


Vector = tuple[float, ...]


@dataclass(frozen=True)
class EmbeddingIdentity:
    provider: str
    model_name: str
    dimensions: int


def embedding_identity(provider: EmbeddingProvider) -> EmbeddingIdentity:
    """Optional identity extension; legacy providers retain the original protocol."""
    identity = getattr(provider, "identity", None)
    if identity is not None:
        if not isinstance(identity, EmbeddingIdentity) or identity.dimensions != provider.dimension:
            raise ValueError("Invalid embedding identity")
        return identity
    cls = type(provider)
    return EmbeddingIdentity(
        f"{cls.__module__}.{cls.__qualname__}",
        getattr(provider, "model", cls.__qualname__),
        provider.dimension,
    )


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> tuple[Vector, ...]:
        ...

    def embed_query(
        self,
        text: str,
    ) -> Vector:
        ...
