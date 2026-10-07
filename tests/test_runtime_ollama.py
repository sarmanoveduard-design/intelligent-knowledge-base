
"""Offline local adapter and composition tests: no Ollama/model loading."""
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import test_runtime_context as context_tests
import test_runtime_retrieval as retrieval_tests
from knowledge_base.runtime.citations import StructuralCitationValidator
from knowledge_base.runtime.composition import compose_local_runtime
from knowledge_base.runtime.config import RuntimeConfig
from knowledge_base.runtime.context import RankedContextAssembler
from knowledge_base.runtime.generation import GenerationRequestBuilder
from knowledge_base.runtime.models import AccessContext, AskRequest, GenerationLimits, RetrievalFilters
from knowledge_base.runtime.ollama import (
    OllamaGenerationConfig, OllamaGenerationProvider, OllamaProviderError,
    LocalOllamaHTTPTransport, _NoRedirect, check_local_models,
)
from knowledge_base.runtime.sufficiency import SufficiencyPolicy


class FakeTransport:
    def __init__(self, model='synthetic:local'):
        self.model, self.calls, self.output, self.error = model, [], None, None
        self.models = [{'name': model, 'size': 10}, {'name': 'embedding:latest', 'size': 10}]

    def request(self, method, path, *, payload, timeout):
        self.calls.append((method, path, payload, timeout))
        if self.error:
            raise self.error
        if path == '/api/version': return {'version': 'synthetic-server'}
        if path == '/api/tags': return {'models': self.models}
        if path == '/api/show': return {'capabilities': ['completion', 'thinking']}
        if self.output is not None: return self.output
        entry = json.loads(payload['messages'][1]['content'])['context_manifest']['entries'][0]
        wire = {'answer': entry['evidence']['chunk']['text'], 'citations': [{
            'source_handle': entry['source_handle'], 'chunk_identity': entry['evidence']['chunk']['identity']}],
            'declared_limitations': []}
        return {'model': self.model, 'done': True, 'done_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(wire)},
                'total_duration': 100, 'load_duration': 20, 'eval_count': 40}


class RuntimeOllamaTests(unittest.TestCase):
    def setUp(self):
        evidence = context_tests.make_evidence()[:1]
        self.context = RankedContextAssembler().assemble(evidence, max_chars=1000)
        self.config = OllamaGenerationConfig('synthetic:local')
        decision = SufficiencyPolicy(context_tests.FakeRegistry(evidence)).evaluate(
            'Arbitrary question?', self.context, config=RuntimeConfig())
        self.request = GenerationRequestBuilder().build(AskRequest('arbitrary-id', 'Arbitrary question?',
            AccessContext('organization/company', 'reader'), language='ru'),
            self.context, decision, config=RuntimeConfig())
        self.transport = FakeTransport()
        self.provider = OllamaGenerationProvider(self.config, transport=self.transport)

    def wire(self):
        entry = self.context.entries[0]
        return {'answer': 'A supported answer.', 'citations': [{
            'source_handle': entry.source_handle, 'chunk_identity': asdict(entry.evidence.chunk.identity)}],
            'declared_limitations': []}

    def frame(self, content, **extra):
        return {'model': self.config.model, 'done': True,
                'message': {'role': 'assistant', 'content': content}, **extra}


    def test_limitation_only_json_decodes_with_empty_answer(self):
        wire = {'answer': '', 'citations': [], 'declared_limitations': ['Requested fact absent.']}
        self.transport.output = self.frame(json.dumps(wire))
        draft = self.provider.generate(self.request)
        self.assertEqual(draft.answer, '')
        self.assertEqual(draft.citations, ())
        self.assertEqual(draft.declared_limitations, ('Requested fact absent.',))

    def test_empty_wire_without_limitations_or_with_citation_is_malformed(self):
        for wire in ({'answer': '', 'citations': [], 'declared_limitations': []},
                     dict(self.wire(), answer='', declared_limitations=['Absent.']),
                     {'answer': ' ', 'citations': [], 'declared_limitations': ['Absent.']}):
            with self.subTest(wire=wire):
                self.transport.output = self.frame(json.dumps(wire))
                with self.assertRaisesRegex(OllamaProviderError, 'OLLAMA_MALFORMED_OUTPUT'):
                    self.provider.generate(self.request)

    def test_prompt_and_schema_allow_only_structured_missing_fact_response(self):
        self.provider.generate(self.request)
        payload = self.transport.calls[-1][2]
        instructions = payload['messages'][0]['content']
        self.assertIn('answer = ""', instructions)
        self.assertIn('citations = []', instructions)
        self.assertIn('Do not put refusal prose in answer', instructions)
        self.assertIn('For any nonempty factual answer, citations are mandatory', instructions)
        self.assertNotIn('Even limitation responses should reference', instructions)
        self.assertEqual(payload['format']['properties']['answer']['minLength'], 0)

    def test_valid_draft_and_structural_citations(self):
        draft = self.provider.generate(self.request)
        result = StructuralCitationValidator().validate(draft, self.context)
        self.assertEqual(result.structural_validity.name, 'PASS')
        self.assertEqual(result.semantic_support.name, 'NOT_CHECKED')
        self.assertEqual(draft.citations[0].chunk_identity, self.context.entries[0].evidence.chunk.identity)

    def test_manifest_is_preserved_including_nested_provenance(self):
        self.provider.generate(self.request)
        payload = self.transport.calls[-1][2]
        user = json.loads(payload['messages'][1]['content'])
        self.assertEqual(set(user), {'question', 'language', 'context_manifest'})
        self.assertEqual(user['question'], self.request.question)
        self.assertEqual(user['language'], 'ru')
        entry = user['context_manifest']['entries'][0]
        original = self.context.entries[0]
        self.assertEqual(entry['source_handle'], original.source_handle)
        self.assertEqual(entry['evidence']['chunk']['identity'], asdict(original.evidence.chunk.identity))
        self.assertEqual(entry['evidence']['chunk']['text'], original.evidence.chunk.text)
        self.assertEqual(entry['evidence']['chunk']['coordinates'], json.loads(json.dumps(asdict(original.evidence.chunk.coordinates))))
        self.assertEqual(entry['evidence']['reranker_score'], original.evidence.reranker_score)
        self.assertEqual(entry['evidence']['chunk']['metadata']['nested']['tags'], ['synthetic', 'company'])

    def test_schema_and_explicit_generation_settings(self):
        draft = self.provider.generate(self.request)
        payload = self.transport.calls[-1][2]
        self.assertFalse(payload['stream'])
        self.assertIs(payload['think'], False)
        self.assertEqual(payload['options'], {'temperature': 0, 'seed': 0, 'num_predict': 512, 'num_ctx': 8192})
        self.assertEqual(payload['keep_alive'], '5m')
        self.assertFalse(payload['format']['additionalProperties'])
        self.assertIn('Do not use external knowledge', payload['messages'][0]['content'])
        self.assertFalse(draft.provider_metadata['full_model_token_budget_known'])
        self.assertNotIn('answer', draft.provider_metadata)

    def test_request_limits_cap_config(self):
        request = replace(self.request, limits=GenerationLimits(max_output_tokens=17, timeout_seconds=2.5))
        self.provider.generate(request)
        self.assertEqual(self.transport.calls[-1][3], 2.5)
        self.assertEqual(self.transport.calls[-1][2]['options']['num_predict'], 17)

    def test_custom_model_and_settings(self):
        config = replace(self.config, model='other:local', num_ctx=4096, seed=9, keep_alive=None)
        transport = FakeTransport(config.model)
        OllamaGenerationProvider(config, transport=transport).generate(self.request)
        payload = transport.calls[-1][2]
        self.assertEqual(payload['model'], config.model)
        self.assertEqual(payload['options']['num_ctx'], 4096)
        self.assertEqual(payload['options']['seed'], 9)
        self.assertNotIn('keep_alive', payload)

    def test_invalid_json_is_not_repaired(self):
        fence = chr(96) * 3
        for raw in ('bad', fence+'json\n{}\n'+fence,
                    '{"answer":"x","answer":"y"}', '{"answer":NaN}', '[]'):
            with self.subTest(raw=raw):
                self.transport.output = self.frame(raw)
                with self.assertRaisesRegex(OllamaProviderError, 'OLLAMA_MALFORMED_OUTPUT'):
                    self.provider.generate(self.request)

    def test_wrong_output_types_and_extra_fields(self):
        bad = [dict(self.wire(), answer=''), dict(self.wire(), answer=3),
               dict(self.wire(), citations={}), dict(self.wire(), declared_limitations='absent'),
               dict(self.wire(), provider_metadata={}), {'answer': 'x'}]
        bad += [dict(self.wire(), citations=[{'source_handle': 'x'}])]
        for wire in bad:
            with self.subTest(wire=wire):
                self.transport.output = self.frame(json.dumps(wire))
                with self.assertRaises(OllamaProviderError): self.provider.generate(self.request)

    def test_invented_handle_is_left_for_mandatory_validator(self):
        wire = self.wire()
        wire['citations'][0]['source_handle'] = 'invented'
        self.transport.output = self.frame(json.dumps(wire))
        draft = self.provider.generate(self.request)
        self.assertEqual(draft.citations[0].source_handle, 'invented')
        self.assertEqual(StructuralCitationValidator().validate(draft, self.context).structural_validity.name, 'FAIL')

    def test_wrong_full_identity_is_not_repaired(self):
        wire = self.wire()
        wire['citations'][0]['chunk_identity']['version_id'] = 'wrong-version'
        self.transport.output = self.frame(json.dumps(wire))
        draft = self.provider.generate(self.request)
        self.assertEqual(StructuralCitationValidator().validate(draft, self.context).structural_validity.name, 'FAIL')

    def test_empty_incomplete_length_and_model_errors(self):
        for frame, code in [(self.frame(''), 'OLLAMA_EMPTY_OUTPUT'),
                            (self.frame('{}', done=False), 'OLLAMA_INCOMPLETE_RESPONSE'),
                            (self.frame('{}', done_reason='length'), 'OLLAMA_OUTPUT_LIMIT_REACHED'),
                            (self.frame('{}', model='wrong:local'), 'OLLAMA_MODEL_MISMATCH'),
                            ({'error': 'private server body'}, 'OLLAMA_PROVIDER_ERROR')]:
            with self.subTest(code=code):
                self.transport.output = frame
                with self.assertRaisesRegex(OllamaProviderError, code): self.provider.generate(self.request)

    def test_transport_timeout_has_no_retry_or_prompt_leak(self):
        self.transport.error = TimeoutError('private details')
        with self.assertRaisesRegex(OllamaProviderError, '^OLLAMA_TIMEOUT$'):
            self.provider.generate(self.request)
        self.assertEqual(len(self.transport.calls), 1)

    def test_invalid_request_does_not_call_transport(self):
        for request in (replace(self.request, instructions=''), replace(self.request, output_schema_version='other')):
            with self.assertRaisesRegex(OllamaProviderError, 'OLLAMA_INVALID_REQUEST'):
                self.provider.generate(request)
        self.assertEqual(self.transport.calls, [])

    def test_config_rejects_remote_and_invalid_settings(self):
        for values in ({'base_url': 'https://example.com'}, {'base_url': 'http://user:pass@localhost:11434'},
                       {'base_url': 'http://localhost:11434/path'}, {'model': 'synthetic:cloud'},
                       {'num_ctx': 0}, {'think': None}, {'temperature': float('nan')}, {'timeout_seconds': 0}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError): replace(self.config, **values)

    def test_read_only_inventory(self):
        inventory = check_local_models(self.config, 'embedding', transport=self.transport)
        self.assertTrue(inventory.ready)
        self.assertEqual([call[1] for call in self.transport.calls], ['/api/version', '/api/tags', '/api/show'])

    def test_missing_or_remote_model_stops_without_pull(self):
        for models in ([], [{'name': self.config.model, 'size': 1, 'remote_host': 'cloud'}]):
            self.transport.models, self.transport.calls = models, []
            self.assertFalse(check_local_models(self.config, 'embedding', transport=self.transport).ready)
            self.assertEqual([call[1] for call in self.transport.calls], ['/api/version', '/api/tags'])

    def test_http_transport_bounded_json_and_disabled_proxies(self):
        opener = unittest.mock.Mock()
        opener.open.return_value = io.BytesIO(b'{"version":"fake"}')
        with patch('knowledge_base.runtime.ollama.build_opener', return_value=opener) as build:
            result = LocalOllamaHTTPTransport(self.config).request('GET', '/api/version', payload=None, timeout=2)
        self.assertEqual(result, {'version': 'fake'})
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertEqual(opener.open.call_args.kwargs['timeout'], 2)

    def test_http_errors_and_response_bound(self):
        errors = [(TimeoutError(), 'OLLAMA_TIMEOUT'), (URLError(TimeoutError()), 'OLLAMA_TIMEOUT'),
                  (URLError('private'), 'OLLAMA_UNAVAILABLE'),
                  (HTTPError('http://localhost', 500, 'private', {}, None), 'OLLAMA_HTTP_ERROR')]
        for error, code in errors:
            opener = unittest.mock.Mock()
            opener.open.side_effect = error
            with patch('knowledge_base.runtime.ollama.build_opener', return_value=opener):
                with self.assertRaisesRegex(OllamaProviderError, code):
                    LocalOllamaHTTPTransport(self.config).request('GET', '/api/version', payload=None, timeout=2)
        opener = unittest.mock.Mock()
        opener.open.return_value = io.BytesIO(b'12345')
        with patch('knowledge_base.runtime.ollama.build_opener', return_value=opener):
            with self.assertRaisesRegex(OllamaProviderError, 'OLLAMA_RESPONSE_TOO_LARGE'):
                LocalOllamaHTTPTransport(replace(self.config, max_response_bytes=4)).request(
                    'GET', '/api/version', payload=None, timeout=2)

    def test_pull_and_redirect_forbidden(self):
        with self.assertRaisesRegex(OllamaProviderError, 'OLLAMA_ENDPOINT_FORBIDDEN'):
            LocalOllamaHTTPTransport(self.config).request('POST', '/api/pull', payload={}, timeout=2)
        with self.assertRaisesRegex(OllamaProviderError, 'OLLAMA_REDIRECT_FORBIDDEN'):
            _NoRedirect().redirect_request(None, None, 302, '', {}, 'https://external')


class LocalCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.embeddings, self.reranker, self.transport = (
            retrieval_tests.FakeEmbeddings(), retrieval_tests.FakeReranker(), FakeTransport())
        self.access = AccessContext('synthetic/org', 'reader', {'read'})
        self.runtime = self.compose()
        self.addCleanup(lambda: self.runtime.close())

    def compose(self):
        return compose_local_runtime(self.path/'registry.sqlite', embeddings=self.embeddings,
            reranker=self.reranker, generation_config=OllamaGenerationConfig('synthetic:local'),
            transport=self.transport, runtime_config=RuntimeConfig(candidate_top_k=3, final_top_k=2))

    def prepare(self, corpus='company'):
        from docx import Document
        path = self.path/(corpus+'.docx')
        document = Document()
        for text in retrieval_tests.CORPORA[corpus]: document.add_paragraph(text)
        document.save(path)
        result = self.runtime.ingestion.ingest_new_document(path, title='Synthetic source',
            organization_id=self.access.organization_id, version_label='v1', required_scopes=frozenset({'read'}))
        self.runtime.indexing.index_version(result.version.identity)
        self.runtime.repository.activate_version(result.version.identity, expected_current_version_id=None)
        return result

    def request(self, **kwargs):
        return AskRequest('arbitrary-request', 'Arbitrary question?', kwargs.get('access', self.access),
                          retrieval_filters=kwargs.get('filters', RetrievalFilters()))

    def test_construction_has_no_model_io(self):
        self.assertEqual(self.transport.calls, [])
        self.assertEqual(self.embeddings.text_calls, [])
        self.assertEqual(self.runtime.readiness(self.access).status, 'EMPTY')

    def test_two_docx_corpora_use_same_complete_runtime(self):
        for corpus in retrieval_tests.CORPORA:
            with self.subTest(corpus=corpus):
                self.prepare(corpus)
                result = self.runtime.ask(self.request())
                self.assertEqual(result.state.name, 'ANSWER')
                self.assertEqual(result.citation_validation.structural_validity.name, 'PASS')
        self.assertTrue(self.embeddings.text_calls)
        self.assertTrue(self.reranker.calls)

    def test_restart_requires_explicit_rebuild(self):
        self.prepare()
        self.runtime.close()
        self.runtime = self.compose()
        self.assertEqual(self.runtime.readiness(self.access).status, 'INDEX_NOT_READY')
        result = self.runtime.ask(self.request())
        self.assertEqual(result.reason_code, 'INDEX_NOT_READY')
        self.assertEqual(self.transport.calls, [])
        self.runtime.rebuild(self.access)
        self.assertEqual(self.runtime.readiness(self.access).status, 'READY')
        self.assertEqual(self.runtime.ask(self.request()).state.name, 'ANSWER')
        count = len(self.embeddings.text_calls)
        self.runtime.rebuild(self.access)
        self.assertEqual(len(self.embeddings.text_calls), count)

    def test_no_scope_refuses_without_generation(self):
        self.prepare()
        result = self.runtime.ask(self.request(access=AccessContext(self.access.organization_id, 'guest')))
        self.assertEqual(result.state.name, 'REFUSE_INSUFFICIENT_CONTEXT')
        self.assertEqual(self.transport.calls, [])

    def test_malformed_generation_is_ask_error(self):
        self.prepare()
        self.transport.output = {'model': 'synthetic:local', 'done': True,
                                 'message': {'role': 'assistant', 'content': 'bad json'}}
        result = self.runtime.ask(self.request())
        self.assertEqual(result.state.name, 'ERROR')
        self.assertEqual(result.reason_code, 'GENERATION_FAILED')

    def test_declared_missing_fact_is_controlled_refusal(self):
        self.prepare()
        original = self.transport.request
        def missing(method, path, **kwargs):
            output = original(method, path, **kwargs)
            if path == '/api/chat':
                wire = json.loads(output['message']['content'])
                wire['answer'] = 'The supplied context does not contain this fact.'
                wire['declared_limitations'] = ['Requested fact absent']
                output['message']['content'] = json.dumps(wire)
            return output
        self.transport.request = missing
        result = self.runtime.ask(self.request())
        self.assertEqual(result.state.name, 'REFUSE_INSUFFICIENT_CONTEXT')
        self.assertEqual(result.reason_code, 'PARTIAL_ANSWER_DISABLED')


    def test_limitation_only_wire_reaches_final_policy_without_citation_error(self):
        self.prepare()
        self.transport.output = {'model': 'synthetic:local', 'done': True,
            'message': {'role': 'assistant', 'content': json.dumps({
                'answer': '', 'citations': [], 'declared_limitations': ['Requested fact absent.']})}}
        result = self.runtime.ask(self.request())
        self.assertEqual(result.state.name, 'REFUSE_INSUFFICIENT_CONTEXT')
        self.assertEqual(result.reason_code, 'GENERATION_REPORTED_INSUFFICIENT_CONTEXT')
        self.assertIsNone(result.citation_validation)
        self.assertEqual(result.declared_limitations, ('Requested fact absent.',))

    def test_filters_cannot_widen_tenant(self):
        self.prepare()
        filters = RetrievalFilters(organization_id='other')
        with self.assertRaises(ValueError): self.runtime.rebuild(self.access, filters=filters)
        result = self.runtime.ask(self.request(filters=filters))
        self.assertEqual(result.reason_code, 'INDEX_READINESS_FAILED')
        self.assertEqual(self.transport.calls, [])


if __name__ == '__main__':
    unittest.main()
