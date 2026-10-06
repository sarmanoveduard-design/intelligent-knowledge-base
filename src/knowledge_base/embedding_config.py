"""Provider registry at the composition boundary; evaluation uses only the protocol."""
import os
from typing import Callable, Mapping

from knowledge_base.embeddings import EmbeddingProvider


def _openai(env: Mapping[str, str]) -> EmbeddingProvider:
    from knowledge_base.openai_embeddings import OpenAIEmbeddingProvider
    dimensions = env.get("OPENAI_EMBEDDING_DIMENSIONS", "")
    return OpenAIEmbeddingProvider(
        model=env.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        dimensions=int(dimensions) if dimensions else None,
        timeout=float(env.get("OPENAI_EMBEDDING_TIMEOUT", "60")),
        batch_size=int(env.get("OPENAI_EMBEDDING_BATCH_SIZE", "64")),
    )


def _ollama(env: Mapping[str, str]) -> EmbeddingProvider:
    from knowledge_base.ollama_embeddings import OllamaEmbeddingProvider
    return OllamaEmbeddingProvider(
        base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=env.get("OLLAMA_EMBEDDING_MODEL", "bge-m3"),
        dimension=int(env.get("OLLAMA_EMBEDDING_DIMENSIONS", "1024")),
        timeout=float(env.get("OLLAMA_EMBEDDING_TIMEOUT", "180")),
    )


PROVIDER_FACTORIES: dict[str, Callable[[Mapping[str, str]], EmbeddingProvider]] = {
    "openai": _openai, "ollama": _ollama,
}


def provider_from_environment() -> EmbeddingProvider:
    name = os.environ.get("EMBEDDING_PROVIDER", "ollama")
    factory = PROVIDER_FACTORIES.get(name)
    if factory is None:
        raise ValueError("Unsupported embedding provider")
    # Never propagate numeric parse errors containing environment values.
    failed = False
    try:
        provider = factory(os.environ)
    except (ValueError, OverflowError):
        failed = True
    if failed:
        raise ValueError("Invalid embedding configuration")
    return provider
