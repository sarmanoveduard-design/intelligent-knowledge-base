import json
import unittest
from io import BytesIO
from unittest.mock import patch

from knowledge_base.ollama_embeddings import OllamaEmbeddingProvider


class OllamaEmbeddingProviderTests(unittest.TestCase):

    @patch("knowledge_base.ollama_embeddings.urlopen")
    def test_embeds_multiple_texts(self, mock_urlopen):
        response = {
            "embeddings": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        }

        mock_urlopen.return_value.__enter__.return_value = BytesIO(
            json.dumps(response).encode("utf-8")
        )

        provider = OllamaEmbeddingProvider(dimension=3)
        vectors = provider.embed_texts(("Первый", "Второй"))

        self.assertEqual(
            vectors,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data)

        self.assertEqual(payload["model"], "bge-m3")
        self.assertEqual(payload["input"], ["Первый", "Второй"])

    @patch("knowledge_base.ollama_embeddings.urlopen")
    def test_rejects_empty_text(self, mock_urlopen):
        provider = OllamaEmbeddingProvider()

        with self.assertRaises(ValueError):
            provider.embed_texts((" ",))

        mock_urlopen.assert_not_called()

    @patch("knowledge_base.ollama_embeddings.urlopen")
    def test_rejects_wrong_vector_count(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value = BytesIO(
            b'{"embeddings": []}'
        )

        provider = OllamaEmbeddingProvider(dimension=3)

        with self.assertRaises(ValueError):
            provider.embed_texts(("Первый",))

    @patch("knowledge_base.ollama_embeddings.urlopen")
    def test_rejects_wrong_dimension(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value = BytesIO(
            b'{"embeddings": [[1.0, 2.0]]}'
        )

        provider = OllamaEmbeddingProvider(dimension=3)

        with self.assertRaises(ValueError):
            provider.embed_query("Проверка")


if __name__ == "__main__":
    unittest.main()
