"""Explicit local Ollama transport and structured generation; no import-time I/O.

API: https://docs.ollama.com/api/chat and /capabilities/structured-outputs.
No model pulling, fallback models, JSON repair, or implicit thinking changes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import ipaddress
import json
from math import isfinite
from time import perf_counter
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .context import validate_evidence
from .generation import OUTPUT_SCHEMA_VERSION, validate_generation_draft
from .models import (
    ChunkIdentity, CitationReference, GenerationDraft, GenerationRequest, Metadata,
    _metadata, _nonempty, _positive_int, _score,
)


class OllamaProviderError(RuntimeError):
    """Safe code only; transport bodies, paths, prompts and exceptions are private."""


def _local_url(value):
    parsed = urlsplit(value)
    host = parsed.hostname
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host in ('localhost', 'ollama')
    if (parsed.scheme not in ('http', 'https') or not local or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ('', '/')):
        raise ValueError('local_ollama_url_required')
    if parsed.port is not None and parsed.port < 1:
        raise ValueError('invalid_ollama_port')
    return value.rstrip('/')


@dataclass(frozen=True)
class OllamaGenerationConfig:
    model: str
    base_url: str = 'http://localhost:11434'
    timeout_seconds: float = 180.0
    num_predict: int = 512
    num_ctx: int = 8192
    temperature: float = 0.0
    seed: int = 0
    think: bool = False
    keep_alive: str | int | None = '5m'
    max_response_bytes: int = 4 * 1024 * 1024

    def __post_init__(self):
        _nonempty(self.model, 'model')
        if self.model.endswith(':cloud'):
            raise ValueError('local_model_required')
        object.__setattr__(self, 'base_url', _local_url(self.base_url))
        _score(self.timeout_seconds, 'timeout_seconds')
        _score(self.temperature, 'temperature')
        if self.timeout_seconds <= 0 or self.temperature < 0:
            raise ValueError('invalid_generation_settings')
        for name in ('num_predict', 'num_ctx', 'max_response_bytes'):
            _positive_int(getattr(self, name), name)
        if type(self.seed) is not int or type(self.think) is not bool:
            raise ValueError('invalid_seed_or_think')
        if self.keep_alive is not None:
            if type(self.keep_alive) is int:
                _positive_int(self.keep_alive, 'keep_alive', minimum=0)
            else:
                _nonempty(self.keep_alive, 'keep_alive')


class OllamaTransport(Protocol):
    def request(self, method: str, path: str, *, payload: dict | None, timeout: float) -> dict: ...


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('nonfinite_json_constant')


def _loads(value):
    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OllamaProviderError('OLLAMA_REDIRECT_FORBIDDEN')


class LocalOllamaHTTPTransport:
    """Local endpoints only; no proxies, redirects, retries, streaming or pulls."""
    def __init__(self, config: OllamaGenerationConfig):
        self.config = config

    def request(self, method: str, path: str, *, payload: dict | None, timeout: float) -> dict:
        if path not in ('/api/chat', '/api/version', '/api/tags', '/api/show'):
            raise OllamaProviderError('OLLAMA_ENDPOINT_FORBIDDEN')
        try:
            data = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf8')
            request = Request(self.config.base_url + path, data=data, method=method,
                              headers={'Content-Type': 'application/json'})
            opener = build_opener(ProxyHandler({}), _NoRedirect())
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(self.config.max_response_bytes + 1)
            if len(raw) > self.config.max_response_bytes:
                raise OllamaProviderError('OLLAMA_RESPONSE_TOO_LARGE')
            value = _loads(raw.decode('utf8'))
            if not isinstance(value, dict):
                raise ValueError
            return value
        except OllamaProviderError:
            raise
        except TimeoutError:
            raise OllamaProviderError('OLLAMA_TIMEOUT') from None
        except HTTPError:
            raise OllamaProviderError('OLLAMA_HTTP_ERROR') from None
        except URLError as error:
            code = 'OLLAMA_TIMEOUT' if isinstance(error.reason, TimeoutError) else 'OLLAMA_UNAVAILABLE'
            raise OllamaProviderError(code) from None
        except (ValueError, UnicodeError):
            raise OllamaProviderError('OLLAMA_INVALID_RESPONSE') from None
        except Exception:
            raise OllamaProviderError('OLLAMA_TRANSPORT_FAILED') from None


def _plain(value):
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _output_schema():
    identity = {'type': 'object', 'additionalProperties': False,
                'properties': {key: {'type': 'string', 'minLength': 1} for key in
                               ('document_id', 'version_id', 'processing_snapshot_id', 'chunk_id')},
                'required': ['document_id', 'version_id', 'processing_snapshot_id', 'chunk_id']}
    return {'type': 'object', 'additionalProperties': False,
            'properties': {
                'answer': {'type': 'string', 'minLength': 0},
                'citations': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                    'properties': {'source_handle': {'type': 'string', 'minLength': 1}, 'chunk_identity': identity},
                    'required': ['source_handle', 'chunk_identity']}},
                'declared_limitations': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}},
            }, 'required': ['answer', 'citations', 'declared_limitations']}


class OllamaGenerationProvider:
    """One configured model, explicit think setting and strictly decoded draft.

    The serialized manifest preserves every entry/text/identity/coordinate/score;
    formatting is additional overhead, never hidden truncation. num_ctx is a
    configured model window, not a proven tokenizer budget. last_call_metadata
    is a per-instance synchronous smoke diagnostic, without response text.
    """
    def __init__(self, config: OllamaGenerationConfig, *, transport: OllamaTransport | None = None):
        self.config = config
        self.transport = transport if transport is not None else LocalOllamaHTTPTransport(config)
        self.last_call_metadata: Metadata = _metadata({})

    def generate(self, request: GenerationRequest) -> GenerationDraft:
        self.last_call_metadata = _metadata({})
        try:
            replace(request)
            replace(request.limits)
            replace(request.context_manifest)
            if request.output_schema_version != OUTPUT_SCHEMA_VERSION:
                raise ValueError
            _nonempty(request.instructions, 'instructions')
            for entry in request.context_manifest.entries:
                validate_evidence(entry.evidence)
            schema = _output_schema()
            instructions = request.instructions + '\nTransport schema takes precedence for output fields. '
            instructions += ('Return only the JSON object specified below. Runtime supplies provider_metadata. '
                'Copy source_handle and all chunk_identity fields exactly from context entries. '
                'If a requested fact is absent, return answer="", citations=[], and nonempty declared_limitations. '
                'Do not write refusal prose in answer or cite a source for an absent fact. '
                'For any nonempty factual answer, citations are mandatory. Do not make unsupported inferences.\n')
            instructions += json.dumps(schema, ensure_ascii=False, sort_keys=True)
            user_data = {'question': request.question, 'language': request.language,
                         'context_manifest': _plain(request.context_manifest)}
            content = json.dumps(user_data, ensure_ascii=False, allow_nan=False, sort_keys=True)
            timeout = min(self.config.timeout_seconds, request.limits.timeout_seconds or self.config.timeout_seconds)
            num_predict = min(self.config.num_predict, request.limits.max_output_tokens or self.config.num_predict)
            payload = {'model': self.config.model, 'stream': False, 'think': self.config.think,
                'format': schema, 'messages': [{'role': 'system', 'content': instructions},
                                               {'role': 'user', 'content': content}],
                'options': {'temperature': self.config.temperature, 'seed': self.config.seed,
                            'num_predict': num_predict, 'num_ctx': self.config.num_ctx}}
            if self.config.keep_alive is not None:
                payload['keep_alive'] = self.config.keep_alive
        except Exception:
            raise OllamaProviderError('OLLAMA_INVALID_REQUEST') from None
        started = perf_counter()
        try:
            response = self.transport.request('POST', '/api/chat', payload=payload, timeout=timeout)
        except OllamaProviderError:
            raise
        except TimeoutError:
            raise OllamaProviderError('OLLAMA_TIMEOUT') from None
        except Exception:
            raise OllamaProviderError('OLLAMA_TRANSPORT_FAILED') from None
        try:
            if not isinstance(response, dict) or response.get('error'):
                raise OllamaProviderError('OLLAMA_PROVIDER_ERROR')
            if response.get('model') != self.config.model:
                raise OllamaProviderError('OLLAMA_MODEL_MISMATCH')
            if response.get('done') is not True:
                raise OllamaProviderError('OLLAMA_INCOMPLETE_RESPONSE')
            if response.get('done_reason') == 'length':
                raise OllamaProviderError('OLLAMA_OUTPUT_LIMIT_REACHED')
            message = response.get('message')
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                raise OllamaProviderError('OLLAMA_INVALID_RESPONSE')
            raw = message.get('content')
            if not isinstance(raw, str) or not raw.strip():
                raise OllamaProviderError('OLLAMA_EMPTY_OUTPUT')
            metadata = {'provider': 'ollama', 'model': self.config.model, 'think': self.config.think,
                'temperature': self.config.temperature, 'seed': self.config.seed, 'num_ctx': self.config.num_ctx,
                'num_predict': num_predict, 'timeout_seconds': timeout, 'keep_alive': self.config.keep_alive,
                'elapsed_seconds': perf_counter() - started, 'serialized_input_chars': len(instructions) + len(content),
                'chunk_text_chars': sum(len(x.evidence.chunk.text) for x in request.context_manifest.entries),
                'full_model_token_budget_known': False, 'thinking_returned': bool(message.get('thinking'))}
            for key in ('total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration',
                        'eval_count', 'eval_duration'):
                if key in response:
                    if type(response[key]) is not int or response[key] < 0:
                        raise ValueError
                    metadata[key] = response[key]
            self.last_call_metadata = _metadata(metadata)
            output = _loads(raw)
            if not isinstance(output, dict) or set(output) != {'answer', 'citations', 'declared_limitations'}:
                raise ValueError
            if not isinstance(output['answer'], str):
                raise ValueError
            if not isinstance(output['citations'], list) or not isinstance(output['declared_limitations'], list):
                raise ValueError
            citations = []
            for reference in output['citations']:
                if not isinstance(reference, dict) or set(reference) != {'source_handle', 'chunk_identity'}:
                    raise ValueError
                identity = reference['chunk_identity']
                if not isinstance(identity, dict) or set(identity) != {
                        'document_id', 'version_id', 'processing_snapshot_id', 'chunk_id'}:
                    raise ValueError
                citations.append(CitationReference(reference['source_handle'], ChunkIdentity(**identity)))
            draft = GenerationDraft(output['answer'], tuple(citations), tuple(output['declared_limitations']), metadata)
            validate_generation_draft(draft)
            return draft
        except OllamaProviderError:
            raise
        except Exception:
            raise OllamaProviderError('OLLAMA_MALFORMED_OUTPUT') from None


@dataclass(frozen=True)
class LocalModelInventory:
    version: str
    models: tuple[str, ...]
    missing_models: tuple[str, ...]
    generation_capabilities: tuple[str, ...]

    @property
    def ready(self):
        return not self.missing_models


def check_local_models(config: OllamaGenerationConfig, embedding_model: str, *, transport=None) -> LocalModelInventory:
    """Read-only health/inventory/show; never pull or load model weights."""
    transport = transport if transport is not None else LocalOllamaHTTPTransport(config)
    try:
        version = transport.request('GET', '/api/version', payload=None, timeout=5)['version']
        _nonempty(version, 'version')
        tags = transport.request('GET', '/api/tags', payload=None, timeout=5)['models']
        names = tuple(sorted(item['name'] for item in tags))
        missing = []
        for required in (config.model, embedding_model):
            canonical = required if ':' in required else required + ':latest'
            item = next((x for x in tags if x['name'] in (required, canonical)), None)
            if item is None or item.get('remote_host') or item.get('remote_model') or item.get('size', 0) <= 0:
                missing.append(required)
        capabilities = ()
        if not missing:
            shown = transport.request('POST', '/api/show', payload={'model': config.model}, timeout=5)
            if shown.get('remote_host') or shown.get('remote_model'):
                missing.append(config.model)
            capabilities = tuple(shown.get('capabilities', ()))
        return LocalModelInventory(version, names, tuple(missing), capabilities)
    except OllamaProviderError:
        raise
    except Exception:
        raise OllamaProviderError('OLLAMA_INVENTORY_INVALID') from None
