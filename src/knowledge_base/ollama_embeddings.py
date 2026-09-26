from dataclasses import dataclass
from json import dumps, load
from typing import Sequence
from urllib.request import Request, urlopen

from knowledge_base.embeddings import Vector


@dataclass(frozen=True)
class OllamaEmbeddingProvider:
    base_url: str = "http://ollama:11434"
    model: str = "bge-m3"
    dimension: int = 1024
    timeout: int = 180

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> tuple[Vector, ...]:
        if not texts:
            return ()

        if any(not text.strip() for text in texts):
            raise ValueError("Текст не может быть пустым")

        payload = dumps({
            "model": self.model,
            "input": list(texts),
        }).encode("utf-8")

        request = Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urlopen(request, timeout=self.timeout) as response:
            result = load(response)

        embeddings = result.get("embeddings")

        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError("Ollama вернула неверное количество векторов")

        vectors = tuple(
            tuple(float(value) for value in embedding)
            for embedding in embeddings
        )

        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError("Неверная размерность embedding")

        return vectors

    def embed_query(self, text: str) -> Vector:
        return self.embed_texts((text,))[0]
