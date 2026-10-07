"""Offline production retrieval tests with persisted synthetic sources and fakes."""
from dataclasses import replace
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from knowledge_base.embeddings import EmbeddingIdentity
from knowledge_base.runtime.config import RuntimeConfig
from knowledge_base.runtime.indexing import IndexingError, IndexingService
from knowledge_base.runtime.models import (
    AccessContext, ChunkIdentity, DocumentIdentity, DocumentVersionIdentity,
    DocumentVersionStatus, EvidenceChunk, IndexIdentity, IndexedEvidence, IndexManifest,
    ProcessingSnapshotIdentity, ProcessingState, RetrievalFilters, RuntimeDocument,
    RuntimeDocumentVersion, RuntimeEvidence,
)
from knowledge_base.runtime.protocols import IndexingRepository, RuntimeReranker, VectorIndex
from knowledge_base.runtime.retrieval import BGERuntimeAdapter, RetrievalError, RetrievalPipeline
from knowledge_base.runtime.storage import SQLiteRepository, StorageError
from knowledge_base.runtime.vector_index import MemoryVectorIndex

ROOT = Path(__file__).resolve().parents[1]
CORPORA = {
    'company': ('Cedar company receives supply requests at the desk.',
                'Cedar company checks deliveries in the garage.', 'Cedar keeps a visitor register.'),
    'laboratory': ('The fictional laboratory gives glass containers blue labels.',
                   'The laboratory gives polymer containers silver labels.', 'The laboratory logs samples before storage.'),
}


def identity():
    return IndexIdentity('synthetic', 'space/v1', 2, 'parser/v1', 'chunker/v1', 'text/v1')


class FakeEmbeddings:
    dimension = 2
    revision = None

    def __init__(self):
        self.identity = EmbeddingIdentity('synthetic', 'space/v1', 2)
        self.text_calls = []
        self.query_calls = []
        self.vectors = {texts[0]: (1.0, 0.0) for texts in CORPORA.values()}
        self.vectors.update({texts[1]: (0.8, 0.6) for texts in CORPORA.values()})
        self.vectors.update({texts[2]: (0.0, 1.0) for texts in CORPORA.values()})

    def embed_texts(self, texts):
        self.text_calls.append(tuple(texts))
        return tuple(self.vectors.get(x, (0.6, 0.8)) for x in texts)

    def embed_query(self, text):
        self.query_calls.append(text)
        return (1.0, 0.0)


class FakeReranker:
    name = 'deterministic-reranker'
    score_type = 'synthetic-logit'
    scorer_identity = 'deterministic/revision1'

    def __init__(self):
        self.calls = []
        self.callback = None

    def rerank(self, question, candidates):
        self.calls.append((question, candidates))
        if self.callback:
            return self.callback(candidates)
        return tuple(replace(item, reranker_score=20.0 - i * 7,
            reranker_score_type=self.score_type, reranker_scorer_identity=self.scorer_identity)
            for i, item in enumerate(reversed(candidates)))


class RuntimeRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / 'registry.sqlite'
        self.repository = SQLiteRepository(self.database)
        self.addCleanup(self.repository.close)
        self.embeddings = FakeEmbeddings()
        self.identity = identity()
        self.index = MemoryVectorIndex(self.identity, self.repository)
        self.indexing = IndexingService(self.repository, self.index, self.embeddings, self.identity)
        self.reranker = FakeReranker()
        self.pipeline = RetrievalPipeline(self.repository, self.index, self.embeddings, self.reranker,
                                          self.identity, RuntimeConfig(candidate_top_k=3, final_top_k=2))
        self.access = AccessContext('org/A', 'reader', {'read', 'internal'})
        self.serial = 0

    def prepare(self, texts=None, *, organization='org/A', scopes=frozenset(), document_id=None,
                version_id=None, chunk_ids=None):
        self.serial += 1
        texts = texts if texts is not None else CORPORA['company']
        document_id = document_id or f'документ/自由#{self.serial}'
        document = RuntimeDocument(DocumentIdentity(document_id, organization), 'Synthetic source', scopes,
                                   {'collection': 'arbitrary'})
        self.repository.save_document(document)
        version = RuntimeDocumentVersion(DocumentVersionIdentity(document_id, version_id or f'version Ω/{self.serial}'), 'new')
        self.repository.save_version(version)
        self.repository.save_version(replace(version, processing_state=ProcessingState.PARSED))
        snapshot = ProcessingSnapshotIdentity(version.identity, f'snapshot/opaque/{self.serial}')
        chunks = tuple(EvidenceChunk(
            ChunkIdentity(document_id, version.identity.version_id, snapshot.processing_snapshot_id,
                          chunk_ids[i] if chunk_ids else f'fragment/自由/{self.serial}/{i}'),
            document.identity, version.identity, text, local_chunk_index=0,
            metadata={'ordinal': i}) for i, text in enumerate(texts))
        self.repository.finish_processing(snapshot, chunks, metadata={'processing': 'v1'})
        return snapshot

    def publish(self, snapshot, previous=None):
        self.repository.activate_version(snapshot.version, expected_current_version_id=previous)

    def ready(self, *args, **kwargs):
        snapshot = self.prepare(*args, **kwargs)
        self.indexing.index_version(snapshot.version)
        self.publish(snapshot)
        return snapshot

    def search(self, **kwargs):
        return self.pipeline.retrieve('An arbitrary new question Ω?', self.access, **kwargs)

    def raw_search(self, *, top_k=10, filters=RetrievalFilters(), vector=(1.0, 0.0), expected=None):
        return self.index.search(vector, access=self.access, filters=filters, top_k=top_k,
                                 expected_identity=expected or self.identity)

    def test_chunked_becomes_indexed_with_durable_manifest(self):
        snapshot = self.prepare()
        self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.CHUNKED)
        manifest = self.indexing.index_version(snapshot.version)
        self.assertTrue(manifest.ready)
        self.assertEqual(manifest.chunk_count, 3)
        self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.INDEXED)
        self.assertEqual(self.repository.get_version(snapshot.version).status, DocumentVersionStatus.DRAFT)
        self.assertEqual(self.repository.get_index_manifest(snapshot, self.identity), manifest)
        self.assertEqual(self.index.get_manifest(snapshot), manifest)

    def test_active_chunked_can_be_indexed(self):
        snapshot = self.prepare()
        self.publish(snapshot)
        self.assertEqual(self.search().evidence, ())
        self.indexing.index_version(snapshot.version)
        self.assertEqual(len(self.search().evidence), 2)

    def test_reindex_is_idempotent_and_does_not_embed_again(self):
        snapshot = self.ready()
        first = self.indexing.index_version(snapshot.version)
        second = self.indexing.index_version(snapshot.version)
        self.assertEqual(first, second)
        self.assertEqual(len(self.embeddings.text_calls), 1)
        self.assertEqual(len(self.raw_search()), 3)
        self.assertEqual(self.repository.connection.execute('SELECT COUNT(*) FROM index_manifests').fetchone()[0], 1)

    def test_embedding_failure_keeps_chunked_and_does_not_publish(self):
        snapshot = self.prepare()
        with patch.object(self.embeddings, 'embed_texts', side_effect=RuntimeError('PRIVATE_SOURCE')):
            with self.assertRaises(IndexingError) as error:
                self.indexing.index_version(snapshot.version)
        self.assertEqual(str(error.exception), 'embedding_failed')
        self.assertEqual(error.exception.version, snapshot.version)
        self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.CHUNKED)
        self.assertIsNone(self.repository.get_index_manifest(snapshot, self.identity))
        self.assertIsNone(self.index.get_manifest(snapshot))

    def test_vector_write_failure_keeps_state_and_removes_partial_publication(self):
        snapshot = self.prepare()
        upsert = self.index.upsert
        def partial(manifest, vectors):
            upsert(manifest, vectors)
            raise RuntimeError('PRIVATE_BACKEND')
        with patch.object(self.index, 'upsert', side_effect=partial):
            with self.assertRaises(IndexingError) as error:
                self.indexing.index_version(snapshot.version)
        self.assertEqual(str(error.exception), 'index_write_failed')
        self.assertIsNone(self.index.get_manifest(snapshot))
        self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.CHUNKED)

    def test_completion_failure_rolls_back_manifest_and_state(self):
        snapshot = self.prepare()
        self.repository.connection.execute('''CREATE TEMP TRIGGER reject_indexing BEFORE UPDATE OF processing_state ON versions
            WHEN NEW.processing_state='indexed' BEGIN SELECT RAISE(ABORT,'PRIVATE_SQL'); END''')
        with self.assertRaises(IndexingError) as error:
            self.indexing.index_version(snapshot.version)
        self.assertEqual(str(error.exception), 'index_completion_failed')
        self.assertIsNone(self.repository.get_index_manifest(snapshot, self.identity))
        self.assertIsNone(self.index.get_manifest(snapshot))
        self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.CHUNKED)

    def test_failed_new_version_preserves_old_active_indexed(self):
        old = self.ready()
        new = self.prepare(CORPORA['laboratory'], document_id=old.version.document_id)
        before = self.search().evidence
        with patch.object(self.embeddings, 'embed_texts', side_effect=RuntimeError):
            with self.assertRaises(IndexingError):
                self.indexing.index_version(new.version)
        self.assertEqual(self.search().evidence, before)
        self.assertEqual(self.repository.get_version(old.version).status, DocumentVersionStatus.ACTIVE)
        self.assertEqual(self.repository.get_version(old.version).processing_state, ProcessingState.INDEXED)
        self.assertEqual(self.repository.get_version(new.version).processing_state, ProcessingState.CHUNKED)

    def test_retry_after_failure_succeeds(self):
        snapshot = self.prepare()
        with patch.object(self.embeddings, 'embed_texts', side_effect=RuntimeError):
            with self.assertRaises(IndexingError):
                self.indexing.index_version(snapshot.version)
        self.indexing.index_version(snapshot.version)
        self.publish(snapshot)
        self.assertTrue(self.search().evidence)

    def test_draft_indexed_is_invisible(self):
        snapshot = self.prepare()
        self.indexing.index_version(snapshot.version)
        self.assertEqual(self.raw_search(), ())
        self.assertEqual(self.search().empty_reason, 'no_visible_versions')
        self.assertEqual(self.embeddings.query_calls, [])

    def test_active_chunked_is_invisible_even_with_memory_vectors(self):
        snapshot = self.prepare()
        self.publish(snapshot)
        chunks = self.repository.list_snapshot(snapshot)
        self.index.upsert(IndexManifest(self.identity, snapshot, len(chunks), True),
                          tuple(IndexedEvidence(c, (1.0, 0.0)) for c in chunks))
        self.assertEqual(self.raw_search(), ())
        self.assertEqual(self.search().evidence, ())

    def test_superseded_is_invisible_without_manual_index_deactivation(self):
        old = self.ready()
        new = self.prepare(CORPORA['laboratory'], document_id=old.version.document_id)
        self.indexing.index_version(new.version)
        self.publish(new, old.version.version_id)
        result = self.raw_search()
        self.assertTrue(result)
        self.assertTrue(all(x.chunk.version == new.version for x in result))
        self.assertTrue(self.index.get_manifest(old).ready)

    def test_archived_is_invisible_without_deletion(self):
        snapshot = self.ready()
        self.repository.archive_version(snapshot.version)
        self.assertEqual(self.search().evidence, ())
        self.assertEqual(self.raw_search(), ())
        self.assertTrue(self.repository.list_snapshot(snapshot))

    def test_unapproved_version_is_invisible(self):
        snapshot = self.ready()
        # Deliberately corrupt approval with constraint checks disabled to verify fail-closed reads.
        self.repository.connection.execute('PRAGMA ignore_check_constraints=ON')
        self.repository.connection.execute('UPDATE versions SET approved=0 WHERE version_id=?', (snapshot.version.version_id,))
        self.assertEqual(self.raw_search(), ())
        self.assertEqual(self.search().evidence, ())

    def test_other_organization_does_not_enter_candidate_pool(self):
        allowed = self.ready((CORPORA['company'][1],))
        self.ready((CORPORA['company'][0],), organization='org/B')
        result = self.raw_search(top_k=1)
        self.assertEqual(result[0].chunk.version, allowed.version)
        self.assertAlmostEqual(result[0].retrieval_score, 0.8)

    def test_filter_cannot_expand_organization_access(self):
        self.ready(organization='org/B')
        with self.assertRaises(RetrievalError):
            self.search(filters=RetrievalFilters(organization_id='org/B'))
        with self.assertRaises(ValueError):
            self.raw_search(filters=RetrievalFilters(organization_id='org/B'))
        self.assertEqual(self.embeddings.query_calls, [])

    def test_scopes_exclude_higher_score_before_top_k(self):
        allowed = self.ready((CORPORA['company'][1],), scopes={'read'})
        self.ready((CORPORA['company'][0],), scopes={'secret'})
        result = self.raw_search(top_k=1)
        self.assertEqual(result[0].chunk.version, allowed.version)

    def test_filter_cannot_grant_missing_scopes(self):
        self.ready(scopes={'secret'})
        with self.assertRaises(RetrievalError):
            self.search(filters=RetrievalFilters(required_scopes={'secret'}))
        self.assertEqual(self.search().evidence, ())

    def test_required_scope_filter_narrows_document_metadata(self):
        selected = self.ready((CORPORA['company'][1],), scopes={'read'})
        self.ready((CORPORA['company'][0],))
        result = self.raw_search(top_k=1, filters=RetrievalFilters(required_scopes={'read'}))
        self.assertEqual(result[0].chunk.version, selected.version)

    def test_document_filter_applies_before_candidate_top_k(self):
        selected = self.ready((CORPORA['company'][1],))
        self.ready((CORPORA['company'][0],))
        result = self.raw_search(top_k=1, filters=RetrievalFilters(allowed_document_ids={selected.version.document_id}))
        self.assertEqual(result[0].chunk.version, selected.version)

    def test_version_filter_and_empty_sets_narrow(self):
        selected = self.ready()
        self.ready(CORPORA['laboratory'])
        result = self.search(filters=RetrievalFilters(allowed_version_ids={selected.version.version_id}))
        self.assertTrue(all(x.chunk.version == selected.version for x in result.evidence))
        self.assertEqual(self.search(filters=RetrievalFilters(allowed_document_ids=set())).evidence, ())
        self.assertEqual(self.search(filters=RetrievalFilters(allowed_version_ids=set())).evidence, ())
        self.assertEqual(self.search(filters=RetrievalFilters(allowed_document_ids={'missing'})).evidence, ())

    def test_arbitrary_document_and_chunk_ids_and_local_zero_do_not_collide(self):
        a = self.ready((CORPORA['company'][0],), document_id='合同 Ω/abc', chunk_ids=['fragment/自由 one'])
        b = self.ready((CORPORA['company'][1],), document_id='another::arbitrary', chunk_ids=['opaque chunk#different'])
        result = self.raw_search()
        self.assertEqual(len(result), 2)
        self.assertEqual({x.chunk.local_chunk_index for x in result}, {0})
        self.assertEqual({x.chunk.identity.chunk_id for x in result}, {'fragment/自由 one', 'opaque chunk#different'})
        self.assertEqual({x.chunk.version for x in result}, {a.version, b.version})

    def test_query_embedding_and_vector_ranking_use_arbitrary_question(self):
        self.ready()
        question = '¿Какие правила действуют для нового устройства δ-47?'
        result = self.pipeline.retrieve(question, self.access)
        self.assertEqual(self.embeddings.query_calls, [question])
        self.assertTrue(result.query_embedded)
        candidates = self.reranker.calls[0][1]
        self.assertEqual([x.chunk.text for x in candidates], list(CORPORA['company']))
        self.assertEqual([round(x.retrieval_score, 2) for x in candidates], [1.0, 0.8, 0.0])

    def test_query_vector_changes_candidate_order(self):
        self.ready()
        with patch.object(self.embeddings, 'embed_query', return_value=(0.0, 1.0)):
            self.search()
        self.assertEqual(self.reranker.calls[0][1][0].chunk.text, CORPORA['company'][2])

    def test_reranker_reorders_and_preserves_separate_scores(self):
        self.ready()
        result = self.search()
        self.assertEqual([x.chunk.text for x in result.evidence], list(reversed(CORPORA['company']))[:2])
        self.assertEqual([x.retrieval_score for x in result.evidence], [0.0, 0.8])
        self.assertEqual([x.reranker_score for x in result.evidence], [20.0, 13.0])
        self.assertEqual({x.retrieval_score_type for x in result.evidence}, {'cosine'})
        self.assertEqual({x.reranker_score_type for x in result.evidence}, {'synthetic-logit'})
        for item in result.evidence:
            self.assertNotEqual(item.retrieval_scorer_identity, item.reranker_scorer_identity)

    def test_candidate_top_k_and_final_top_k(self):
        self.ready()
        self.ready(CORPORA['laboratory'])
        result = self.search(config=RuntimeConfig(candidate_top_k=4, final_top_k=1))
        self.assertEqual(len(self.reranker.calls[0][1]), 4)
        self.assertEqual(result.candidate_count, 4)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.visible_version_count, 2)

    def test_reranker_cannot_inject_candidate(self):
        a = self.ready()
        b = self.ready(CORPORA['laboratory'])
        foreign = self.repository.list_snapshot(b)[0]
        self.reranker.callback = lambda items: (replace(items[0], chunk=foreign),) + items[1:]
        with self.assertRaises(RetrievalError):
            self.search(filters=RetrievalFilters(allowed_version_ids={a.version.version_id}))

    def test_reranker_cannot_drop_or_duplicate_candidates(self):
        self.ready()
        for callback in (lambda xs: xs[:-1], lambda xs: (xs[0],) * len(xs)):
            with self.subTest(callback=callback):
                self.reranker.callback = callback
                with self.assertRaises(RetrievalError):
                    self.search()

    def test_reranker_cannot_change_source_or_retrieval_score_metadata(self):
        self.ready()
        mutations = [lambda x: replace(x, chunk=replace(x.chunk, text='fabricated source')),
                     lambda x: replace(x, retrieval_score=0.33),
                     lambda x: replace(x, retrieval_score_type='probability'),
                     lambda x: replace(x, retrieval_scorer_identity='another-space')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.reranker.callback = lambda xs: (mutation(xs[0]),) + xs[1:]
                with self.assertRaises(RetrievalError):
                    self.search()

    def test_reranker_requires_finite_separate_scores_and_declared_identity(self):
        self.ready()
        for field, value in [('reranker_score', float('nan')), ('reranker_score_type', 'probability'),
                             ('reranker_scorer_identity', 'another-reranker')]:
            with self.subTest(field=field):
                def bad(xs):
                    result = tuple(replace(x, reranker_score=1.0, reranker_score_type=self.reranker.score_type,
                                           reranker_scorer_identity=self.reranker.scorer_identity) for x in xs)
                    object.__setattr__(result[0], field, value)
                    return result
                self.reranker.callback = bad
                with self.assertRaises(RetrievalError):
                    self.search()

    def test_reranker_failure_is_sanitized(self):
        self.ready()
        with patch.object(self.reranker, 'rerank', side_effect=RuntimeError('PRIVATE_QUERY')):
            with self.assertRaises(RetrievalError) as error:
                self.search()
        self.assertEqual(str(error.exception), 'reranking_failed')

    def test_archive_during_reranking_is_rechecked(self):
        snapshot = self.ready()
        def archive(question, candidates):
            self.repository.archive_version(snapshot.version)
            return FakeReranker().rerank(question, candidates)
        with patch.object(self.reranker, 'rerank', side_effect=archive):
            result = self.search()
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.empty_reason, 'visibility_changed')

    def test_scope_change_during_reranking_is_rechecked(self):
        snapshot = self.ready()
        def restrict(question, candidates):
            doc = self.repository.get_document(candidates[0].chunk.document)
            self.repository.save_document(replace(doc, required_scopes={'secret'}))
            return FakeReranker().rerank(question, candidates)
        with patch.object(self.reranker, 'rerank', side_effect=restrict):
            self.assertEqual(self.search().evidence, ())

    def test_visibility_change_before_reranking_never_exposes_source(self):
        snapshot = self.ready()
        search = self.index.search
        def archive(*args, **kwargs):
            candidates = search(*args, **kwargs)
            self.repository.archive_version(snapshot.version)
            return candidates
        with patch.object(self.index, 'search', side_effect=archive):
            self.assertEqual(self.search().evidence, ())
        self.assertEqual(self.reranker.calls, [])

    def test_vector_backend_forged_text_is_not_sent_to_reranker(self):
        self.ready()
        forged = self.raw_search()[0]
        forged = replace(forged, chunk=replace(forged.chunk, text='forged confidential source'))
        with patch.object(self.index, 'search', return_value=(forged,)):
            self.assertEqual(self.search().evidence, ())
        self.assertEqual(self.reranker.calls, [])

    def test_vector_backend_cannot_exceed_candidate_limit_or_duplicate(self):
        self.ready()
        values = self.raw_search()
        for result in ((values[0],) * 3, values + (values[0],)):
            with self.subTest(size=len(result)), patch.object(self.index, 'search', return_value=result):
                with self.assertRaises(RetrievalError):
                    self.search()

    def test_index_identity_checks_every_field(self):
        snapshot = self.ready()
        manifest = self.index.get_manifest(snapshot)
        vectors = tuple(IndexedEvidence(c, (1.0, 0.0)) for c in self.repository.list_snapshot(snapshot))
        changes = {'embedding_provider': 'other', 'embedding_model': 'other', 'vector_dimension': 3,
                   'preprocessing_version': 'v2', 'chunking_version': 'v2',
                   'representation_version': 'v2', 'embedding_revision': 'revision2'}
        for field, value in changes.items():
            other = replace(self.identity, **{field: value})
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.raw_search(expected=other)
                with self.assertRaises(ValueError):
                    self.index.upsert(replace(manifest, identity=other), vectors)
                with self.assertRaises(RetrievalError):
                    replace(self.pipeline, identity=other).retrieve('question', self.access)

    def test_same_dimension_different_model_is_rejected(self):
        snapshot = self.prepare()
        self.embeddings.identity = EmbeddingIdentity('synthetic', 'space/v2', 2)
        with self.assertRaises(IndexingError):
            self.indexing.index_version(snapshot.version)
        with self.assertRaises(RetrievalError):
            self.search()
        self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.CHUNKED)

    def test_new_embedding_space_requires_new_index_and_reindex(self):
        snapshot = self.ready()
        new_identity = replace(self.identity, embedding_model='space/v2')
        self.embeddings.identity = EmbeddingIdentity('synthetic', 'space/v2', 2)
        new_index = MemoryVectorIndex(new_identity, self.repository)
        pipeline = replace(self.pipeline, index=new_index, identity=new_identity)
        self.assertEqual(pipeline.retrieve('a new question', self.access).evidence, ())
        IndexingService(self.repository, new_index, self.embeddings, new_identity).index_version(snapshot.version)
        self.assertTrue(pipeline.retrieve('a new question', self.access).evidence)
        self.assertIsNotNone(self.repository.get_index_manifest(snapshot, self.identity))
        self.assertIsNotNone(self.repository.get_index_manifest(snapshot, new_identity))

    def test_embedding_revision_must_match_provider(self):
        snapshot = self.prepare()
        revision_identity = replace(self.identity, embedding_revision='rev1')
        revised_index = MemoryVectorIndex(revision_identity, self.repository)
        service = replace(self.indexing, index=revised_index, identity=revision_identity)
        with self.assertRaises(IndexingError):
            service.index_version(snapshot.version)
        self.embeddings.revision = 'rev1'
        self.assertTrue(service.index_version(snapshot.version).ready)
        self.embeddings.revision = 'rev2'
        with self.assertRaises(RetrievalError):
            replace(self.pipeline, index=revised_index, identity=revision_identity).retrieve('question', self.access)

    def test_invalid_embedding_count_dimension_and_nonfinite_keep_chunked(self):
        for vectors in [(), ((1.0, 0.0),), ((1.0,),) * 3, ((float('nan'), 0.0),) * 3,
                        ((float('inf'), 0.0),) * 3, ((True, 0.0),) * 3]:
            with self.subTest(vectors=vectors):
                snapshot = self.prepare()
                with patch.object(self.embeddings, 'embed_texts', return_value=vectors):
                    with self.assertRaises(IndexingError):
                        self.indexing.index_version(snapshot.version)
                self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.CHUNKED)
                self.assertIsNone(self.index.get_manifest(snapshot))

    def test_invalid_query_vectors_are_rejected(self):
        self.ready()
        for vector in [(1.0,), (float('nan'), 0.0), (float('inf'), 0.0), (False, 0.0)]:
            with self.subTest(vector=vector), patch.object(self.embeddings, 'embed_query', return_value=vector):
                with self.assertRaises(RetrievalError):
                    self.search()
                with self.assertRaises(ValueError):
                    self.raw_search(vector=vector)

    def test_query_embedding_failure_is_sanitized(self):
        self.ready()
        with patch.object(self.embeddings, 'embed_query', side_effect=RuntimeError('PRIVATE_QUERY')):
            with self.assertRaises(RetrievalError) as error:
                self.search()
        self.assertEqual(str(error.exception), 'query_embedding_failed')

    def test_empty_index_returns_empty_with_visible_indexed_registry(self):
        snapshot = self.ready()
        self.index.delete_snapshot(snapshot)
        result = self.search()
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.empty_reason, 'no_ready_candidates')
        self.assertTrue(result.query_embedded)
        self.assertEqual(self.reranker.calls, [])

    def test_no_allowed_documents_returns_empty_without_embedding(self):
        self.ready(organization='org/B')
        result = self.search()
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.visible_version_count, 0)
        self.assertFalse(result.query_embedded)
        self.assertEqual(self.reranker.calls, [])

    def test_both_different_corpora_use_same_runtime(self):
        for label, texts in CORPORA.items():
            with self.subTest(corpus=label):
                snapshot = self.ready(texts, organization='org/' + label)
                result = self.pipeline.retrieve('A previously unseen question about ' + label,
                    AccessContext('org/' + label, 'new principal'))
                self.assertEqual({x.chunk.text for x in result.evidence}, set(texts[1:]))
                self.assertTrue(all(x.chunk.version == snapshot.version for x in result.evidence))

    def test_deactivate_and_delete_snapshots_are_effective_and_rebuildable(self):
        snapshot = self.ready()
        self.index.deactivate_version(snapshot.version)
        self.assertFalse(self.index.get_manifest(snapshot).ready)
        self.assertEqual(self.search().evidence, ())
        self.indexing.index_version(snapshot.version)
        self.assertTrue(self.search().evidence)
        self.index.delete_snapshot(snapshot)
        self.assertIsNone(self.index.get_manifest(snapshot))
        self.assertEqual(self.search().evidence, ())
        self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.INDEXED)

    def test_restart_preserves_manifests_but_requires_memory_rebuild(self):
        snapshot = self.ready()
        with SQLiteRepository(self.database) as reopened:
            manifest = reopened.get_index_manifest(snapshot, self.identity)
            self.assertTrue(manifest.ready)
            memory = MemoryVectorIndex(self.identity, reopened)
            pipeline = replace(self.pipeline, repository=reopened, index=memory)
            self.assertEqual(pipeline.retrieve('new question', self.access).evidence, ())
            IndexingService(reopened, memory, self.embeddings, self.identity).index_version(snapshot.version)
            self.assertTrue(pipeline.retrieve('new question', self.access).evidence)

    def test_schema_v1_migrates_without_changing_saved_chunks(self):
        snapshot = self.prepare()
        self.repository.connection.execute('DROP TABLE index_manifests')
        self.repository.connection.execute('PRAGMA user_version=1')
        with SQLiteRepository(self.database) as reopened:
            self.assertEqual(reopened.connection.execute('PRAGMA user_version').fetchone()[0], 2)
            self.assertEqual(reopened.list_snapshot(snapshot), self.repository.list_snapshot(snapshot))
            self.assertIsNone(reopened.get_index_manifest(snapshot, self.identity))

    def test_invalid_indexing_states_are_rejected(self):
        for state in (ProcessingState.REGISTERED, ProcessingState.PARSED, ProcessingState.FAILED):
            with self.subTest(state=state):
                doc = RuntimeDocument(DocumentIdentity('doc/' + state.value, 'org/A'), 'Unprepared')
                self.repository.save_document(doc)
                version = RuntimeDocumentVersion(DocumentVersionIdentity(doc.identity.document_id, 'v/' + state.value), 'v')
                self.repository.save_version(version)
                if state != ProcessingState.REGISTERED:
                    self.repository.save_version(replace(version, processing_state=state))
                with self.assertRaises(IndexingError):
                    self.indexing.index_version(version.identity)
        with self.assertRaises(IndexingError):
            self.indexing.index_version(DocumentVersionIdentity('missing', 'missing'))

    def test_archived_and_superseded_cannot_be_reindexed(self):
        old = self.ready()
        new = self.ready(CORPORA['laboratory'])
        self.repository.archive_version(new.version)
        replacement = self.prepare(CORPORA['laboratory'], document_id=old.version.document_id)
        self.indexing.index_version(replacement.version)
        self.publish(replacement, old.version.version_id)
        for snapshot in (old, new):
            with self.subTest(snapshot=snapshot), self.assertRaises(IndexingError):
                self.indexing.index_version(snapshot.version)

    def test_archive_during_indexing_prevents_completion(self):
        snapshot = self.prepare()
        embed = self.embeddings.embed_texts
        def archive(texts):
            self.repository.archive_version(snapshot.version)
            return embed(texts)
        with patch.object(self.embeddings, 'embed_texts', side_effect=archive):
            with self.assertRaises(IndexingError):
                self.indexing.index_version(snapshot.version)
        self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.CHUNKED)
        self.assertIsNone(self.repository.get_index_manifest(snapshot, self.identity))
        self.assertIsNone(self.index.get_manifest(snapshot))

    def test_unselected_snapshot_not_visible_and_not_completable(self):
        self.ready()
        # Persist another immutable snapshot on a draft version to respect storage lifecycle.
        draft = self.prepare()
        original = self.repository.list_snapshot(draft)[0]
        other = replace(draft, processing_snapshot_id='unselected/snapshot')
        chunk = replace(original, identity=replace(original.identity, processing_snapshot_id=other.processing_snapshot_id,
                                                  chunk_id='unselected/chunk'))
        self.repository.save_snapshot(other, (chunk,))
        manifest = IndexManifest(self.identity, other, 1, True)
        self.index.upsert(manifest, (IndexedEvidence(chunk, (1.0, 0.0)),))
        with self.assertRaises(StorageError):
            self.repository.complete_indexing(manifest)
        self.indexing.index_version(draft.version)
        self.publish(draft)
        self.assertNotIn(chunk.identity, {x.chunk.identity for x in self.raw_search()})

    def test_complete_indexing_checks_ready_count_and_snapshot(self):
        snapshot = self.prepare()
        for manifest in (IndexManifest(self.identity, snapshot, 3, False),
                         IndexManifest(self.identity, snapshot, 2, True)):
            with self.subTest(manifest=manifest), self.assertRaises(StorageError):
                self.repository.complete_indexing(manifest)
        self.assertEqual(self.repository.get_version(snapshot.version).processing_state, ProcessingState.CHUNKED)
        self.assertIsNone(self.repository.get_index_manifest(snapshot, self.identity))

    def test_upsert_validation_is_atomic_and_snapshot_is_immutable(self):
        snapshot = self.ready()
        manifest = self.index.get_manifest(snapshot)
        chunks = self.repository.list_snapshot(snapshot)
        vectors = tuple(IndexedEvidence(c, v) for c, v in zip(chunks, self.embeddings.embed_texts([c.text for c in chunks])))
        before = self.raw_search()
        wrong = replace(chunks[0], identity=replace(chunks[0].identity, processing_snapshot_id='wrong'))
        invalid = [vectors[:-1], (vectors[0], vectors[0], vectors[2]),
                   (IndexedEvidence(wrong, (1.0, 0.0)),) + vectors[1:],
                   (IndexedEvidence(chunks[0], (1.0,)),) + vectors[1:],
                   (IndexedEvidence(replace(chunks[0], text='changed'), vectors[0].vector),) + vectors[1:]]
        for items in invalid:
            with self.subTest(items=items), self.assertRaises(ValueError):
                self.index.upsert(manifest, items)
            self.assertEqual(self.raw_search(), before)
            self.assertEqual(self.index.get_manifest(snapshot), manifest)
        self.index.upsert(manifest, vectors)
        self.assertEqual(self.raw_search(), before)

    def test_nonready_and_unpersisted_manifests_are_unsearchable(self):
        snapshot = self.ready()
        manifest = self.index.get_manifest(snapshot)
        chunks = self.repository.list_snapshot(snapshot)
        vectors = tuple(IndexedEvidence(c, v) for c, v in zip(chunks, self.embeddings.embed_texts([c.text for c in chunks])))
        self.index.upsert(replace(manifest, ready=False), vectors)
        self.assertEqual(self.raw_search(), ())
        self.index.upsert(manifest, vectors)
        self.repository.connection.execute('DELETE FROM index_manifests')
        self.assertEqual(self.raw_search(), ())

    def test_numerically_large_finite_and_zero_vectors(self):
        snapshot = self.prepare(('large', 'zero'))
        with patch.object(self.embeddings, 'embed_texts', return_value=((1e308, 1e308), (0.0, 0.0))):
            self.indexing.index_version(snapshot.version)
        self.publish(snapshot)
        result = self.raw_search(vector=(1e308, 1e308))
        self.assertAlmostEqual(result[0].retrieval_score, 1.0)
        self.assertEqual(result[1].retrieval_score, 0.0)
        self.assertEqual([x.retrieval_score for x in self.raw_search(vector=(0.0, 0.0))], [0.0, 0.0])

    def test_invalid_top_k_and_empty_question_rejected(self):
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.raw_search(top_k=k)
        with self.assertRaises(RetrievalError):
            self.pipeline.retrieve('   ', self.access)

    def test_structural_protocols_and_alternative_vector_backend(self):
        self.assertIsInstance(self.repository, IndexingRepository)
        self.assertIsInstance(self.index, VectorIndex)
        self.assertIsInstance(self.reranker, RuntimeReranker)
        underlying = self.index
        class DelegatingIndex:
            identity = underlying.identity
            def upsert(self, *args, **kwargs): return underlying.upsert(*args, **kwargs)
            def search(self, *args, **kwargs): return underlying.search(*args, **kwargs)
            def get_manifest(self, *args): return underlying.get_manifest(*args)
            def delete_snapshot(self, *args): return underlying.delete_snapshot(*args)
            def deactivate_version(self, *args): return underlying.deactivate_version(*args)
        backend = DelegatingIndex()
        snapshot = self.prepare()
        replace(self.indexing, index=backend).index_version(snapshot.version)
        self.publish(snapshot)
        self.assertTrue(replace(self.pipeline, index=backend).retrieve('question', self.access).evidence)

    def test_existing_bge_adapter_uses_fake_backend_and_preserves_sources(self):
        # The legacy constructor lives outside runtime. Stub its config error
        # dependency if absent; the benchmark module is not loaded by this test.
        with patch.dict(sys.modules, {'knowledge_base.benchmark_config': SimpleNamespace(BenchmarkConfigurationError=ValueError)}):
            module = importlib.import_module('knowledge_base.bge_reranker')
        class Backend:
            revision = 'a' * 40
            def __init__(self): self.pairs = []
            def score_pairs(self, pairs, *, max_length):
                self.pairs.extend(pairs)
                return [20.0 if 'register' in pair[1] else -5.0 for pair in pairs]
        backend = Backend()
        adapter = BGERuntimeAdapter(module.BGEReranker(backend=backend))
        self.ready()
        result = replace(self.pipeline, reranker=adapter).retrieve('Unseen question', self.access)
        self.assertEqual(result.evidence[0].chunk.text, CORPORA['company'][2])
        self.assertEqual(result.evidence[0].reranker_score, 20.0)
        self.assertEqual(result.evidence[0].retrieval_score, 0.0)
        self.assertEqual(result.evidence[0].reranker_score_type, 'bge-reranker-logit')
        self.assertIn('a' * 40, result.evidence[0].reranker_scorer_identity)
        self.assertEqual(len(backend.pairs), 3)
        self.assertIsInstance(adapter, RuntimeReranker)
        self.assertEqual(adapter.rerank('unused', ()), ())

    def test_bge_adapter_rejects_changed_candidate_identity_and_source(self):
        self.ready()
        candidates = self.raw_search()
        class BadLegacy:
            name = 'test-legacy'
            metadata = {'reranker_model': 'fake', 'reranker_revision': 'v1',
                        'reranker_max_length': 128, 'reranker_precision': 'float32'}
            def rerank(self, query, items):
                return (replace(items[0], chunk_id=replace(items[0].chunk_id, chunk_id='injected')),) + items[1:]
        with self.assertRaises(RetrievalError):
            BGERuntimeAdapter(BadLegacy()).rerank('question', candidates)

    def test_runtime_retrieval_executes_without_benchmark_or_model_imports(self):
        code = r'''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('knowledge_base.benchmark', 'knowledge_base.team', 'benchmarks', 'scripts',
                                'knowledge_base.bge_reranker', 'knowledge_base.candidate_retrieval',
                                'torch', 'transformers', 'openai', 'ollama')):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Reject())
from test_runtime_retrieval import RuntimeRetrievalTests
case = RuntimeRetrievalTests('test_both_different_corpora_use_same_runtime')
case.setUp()
try:
    case.test_both_different_corpora_use_same_runtime()
finally:
    case.doCleanups()
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, 'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(ROOT / 'tests'))),
                 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
