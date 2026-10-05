import os
import traceback
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from knowledge_base.embedding_config import provider_from_environment
from knowledge_base.openai_embeddings import OpenAIEmbeddingError, OpenAIEmbeddingProvider
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import InMemoryVectorStore
from test_retriever import make_chunk


def fake_client(dimension, *, reverse=False):
    def create(**kwargs):
        rows = [NS(index=i, embedding=[float(i + 1)] + [0.0] * (dimension - 1))
                for i, _ in enumerate(kwargs["input"])]
        return NS(data=list(reversed(rows)) if reverse else rows,
                  usage=NS(prompt_tokens=7, total_tokens=7))
    return NS(embeddings=NS(create=Mock(side_effect=create)))


class OpenAIEmbeddingTests(unittest.TestCase):
    def test_small_native(self):
        provider = OpenAIEmbeddingProvider(client=fake_client(1536))
        self.assertEqual(provider.dimension, 1536)
        self.assertEqual(len(provider.embed_query("query")), 1536)

    def test_large_native(self):
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large", client=fake_client(3072))
        self.assertEqual(provider.dimension, 3072)
        self.assertEqual(len(provider.embed_query("query")), 3072)

    def test_1024_both_models(self):
        for model in ("text-embedding-3-small", "text-embedding-3-large"):
            with self.subTest(model=model):
                provider = OpenAIEmbeddingProvider(model=model, dimensions=1024, client=fake_client(1024))
                self.assertEqual(len(provider.embed_query("query")), 1024)
                self.assertEqual(provider.client.embeddings.create.call_args.kwargs["dimensions"], 1024)

    def test_batch_ordering_and_timeout(self):
        client = fake_client(2, reverse=True)
        provider = OpenAIEmbeddingProvider(dimensions=2, batch_size=2, timeout=17, client=client)
        self.assertEqual(provider.embed_texts(("a", "b", "c")), ((1., 0.), (2., 0.), (1., 0.)))
        self.assertEqual([c.kwargs["input"] for c in client.embeddings.create.call_args_list], [["a", "b"], ["c"]])
        self.assertEqual(client.embeddings.create.call_args.kwargs["timeout"], 17)
        self.assertEqual(provider.telemetry["total_tokens"], 14)
        self.assertEqual(provider.telemetry["api_calls"], 2)

    def test_embed_query_and_empty_batch(self):
        client = fake_client(2)
        provider = OpenAIEmbeddingProvider(dimensions=2, client=client)
        self.assertEqual(provider.embed_texts(()), ())
        client.embeddings.create.assert_not_called()
        self.assertIsInstance(provider.embed_query("a"), tuple)
        self.assertEqual(client.embeddings.create.call_args.kwargs["input"], ["a"])

    def test_blank_input_rejected_without_call(self):
        client = fake_client(2)
        provider = OpenAIEmbeddingProvider(dimensions=2, client=client)
        with self.assertRaises(ValueError):
            provider.embed_texts(("a", " "))
        client.embeddings.create.assert_not_called()

    def test_wrong_vector_count(self):
        client = fake_client(2)
        client.embeddings.create.side_effect = lambda **kw: NS(data=[], usage=None)
        provider = OpenAIEmbeddingProvider(dimensions=2, client=client)
        with self.assertRaisesRegex(OpenAIEmbeddingError, "vector count"):
            provider.embed_query("a")
        self.assertEqual(provider.telemetry["error_count"], 1)

    def test_wrong_dimension(self):
        provider = OpenAIEmbeddingProvider(dimensions=1024, client=fake_client(2))
        with self.assertRaisesRegex(OpenAIEmbeddingError, "vector dimension"):
            provider.embed_query("a")

    def test_later_batch_failure_does_not_partially_index(self):
        client = fake_client(2)
        original = client.embeddings.create.side_effect
        client.embeddings.create.side_effect = [original(input=["a"]), RuntimeError("private failure")]
        provider = OpenAIEmbeddingProvider(dimensions=2, batch_size=1, client=client)
        store = InMemoryVectorStore(dimension=2)
        retriever = Retriever(provider, store)
        with self.assertRaises(OpenAIEmbeddingError):
            retriever.index_chunks((make_chunk(0, "a"), make_chunk(1, "b")))
        self.assertEqual(store.count, 0)

    def test_invalid_indices(self):
        for indices in ([0, 0], [-1, 1], [0, 2]):
            client = fake_client(2)
            client.embeddings.create.side_effect = lambda **kw: NS(data=[NS(index=i, embedding=[1, 0]) for i in indices])
            with self.assertRaisesRegex(OpenAIEmbeddingError, "indices"):
                OpenAIEmbeddingProvider(dimensions=2, client=client).embed_texts(("a", "b"))

    def test_nonfinite_values(self):
        client = fake_client(2)
        client.embeddings.create.side_effect = lambda **kw: NS(data=[NS(index=0, embedding=[float("nan"), 0])])
        with self.assertRaisesRegex(OpenAIEmbeddingError, "nonfinite"):
            OpenAIEmbeddingProvider(dimensions=2, client=client).embed_query("a")

    def test_sanitized_api_error_no_key_or_context(self):
        secret = "synthetic-secret-do-not-report"
        client = fake_client(2)
        client.embeddings.create.side_effect = RuntimeError(secret + " private input")
        provider = OpenAIEmbeddingProvider(dimensions=2, client=client)
        try:
            provider.embed_query("a")
        except OpenAIEmbeddingError as error:
            self.assertNotIn(secret, str(error))
            self.assertNotIn(secret, traceback.format_exc())
            self.assertIsNone(error.__context__)
            self.assertIsNone(error.__cause__)
        else:
            self.fail("Expected sanitized failure")
        self.assertNotIn(secret, repr(provider))

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_environment_key(self):
        with self.assertRaisesRegex(OpenAIEmbeddingError, "OPENAI_API_KEY"):
            OpenAIEmbeddingProvider()

    def test_official_sdk_receives_only_environment_key(self):
        client = fake_client(1536)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-key"}), patch("openai.OpenAI", return_value=client) as ctor:
            provider = OpenAIEmbeddingProvider()
            ctor.assert_called_once_with(api_key="synthetic-test-key", timeout=60.0, max_retries=0,
                                         base_url="https://api.openai.com/v1")
            client.embeddings.create.assert_not_called()
            self.assertNotIn("synthetic-test-key", repr(provider))

    def test_sanitized_initialization_error(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-key"}), patch("openai.OpenAI", side_effect=RuntimeError("synthetic-test-key")):
            try:
                OpenAIEmbeddingProvider()
            except OpenAIEmbeddingError as error:
                self.assertIsNone(error.__context__)
                self.assertNotIn("synthetic-test-key", traceback.format_exc())
            else:
                self.fail("Expected initialization failure")

    def test_invalid_configuration(self):
        for kw in ({"dimensions": 0}, {"dimensions": 1537}, {"dimensions": True},
                   {"model": "generation-model"}, {"timeout": 0}, {"timeout": float("nan")}, {"batch_size": 0}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                OpenAIEmbeddingProvider(client=fake_client(2), **kw)

    def test_retriever_compatibility(self):
        provider = OpenAIEmbeddingProvider(dimensions=1024, client=fake_client(1024))
        retriever = Retriever(provider, InMemoryVectorStore(dimension=1024))
        retriever.index_chunks((make_chunk(0, "a"),))
        self.assertEqual(retriever.search("a")[0].chunk.chunk_index, 0)

    def test_reject_model_mixing_same_dimension(self):
        store = InMemoryVectorStore(dimension=1024)
        first = OpenAIEmbeddingProvider(dimensions=1024, client=fake_client(1024))
        Retriever(first, store)
        other = OpenAIEmbeddingProvider(model="text-embedding-3-large", dimensions=1024, client=fake_client(1024))
        with self.assertRaises(ValueError):
            Retriever(other, store)

    def test_reject_provider_swap_after_construction(self):
        provider = OpenAIEmbeddingProvider(dimensions=1024, client=fake_client(1024))
        retriever = Retriever(provider, InMemoryVectorStore(dimension=1024))
        retriever.embedding_provider = OpenAIEmbeddingProvider(model="text-embedding-3-large", dimensions=1024, client=fake_client(1024))
        with self.assertRaises(ValueError):
            retriever.search("a")

    def test_environment_factory_large_native(self):
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai", "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
                                   "OPENAI_API_KEY": "synthetic-test-key"}, clear=True), patch("openai.OpenAI", return_value=fake_client(3072)):
            self.assertEqual(provider_from_environment().dimension, 3072)

    def test_factory_sanitizes_bad_environment(self):
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai", "OPENAI_EMBEDDING_DIMENSIONS": "synthetic-secret"}, clear=True):
            with self.assertRaisesRegex(ValueError, "^Invalid embedding configuration$"):
                provider_from_environment()
