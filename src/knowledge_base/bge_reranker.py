"""Optional local cross-encoder runtime. Importing this module never imports torch."""
from dataclasses import dataclass, replace
from math import isfinite
import os
import re
from time import perf_counter

from knowledge_base.benchmark_config import BenchmarkConfigurationError

MODEL = "BAAI/bge-reranker-v2-m3"


class RerankerRuntimeError(RuntimeError):
    """Safe public error without model exception details or input texts."""


@dataclass(frozen=True)
class BGERerankerConfig:
    model: str = MODEL
    device: str = "auto"
    batch_size: int = 4
    max_length: int = 512
    revision: str = "main"
    cache_dir: str = ".model-cache/huggingface"

    def __post_init__(self):
        if self.model != MODEL:
            raise BenchmarkConfigurationError("This reranker adapter requires BAAI/bge-reranker-v2-m3")
        if self.device not in ("auto", "cpu", "cuda"):
            raise BenchmarkConfigurationError("Reranker device must be auto, cpu or cuda")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 1024:
            raise BenchmarkConfigurationError("Reranker batch size must be an integer between 1 and 1024")
        if type(self.max_length) is not int or not 8 <= self.max_length <= 8192:
            raise BenchmarkConfigurationError("Reranker max length must be an integer between 8 and 8192")
        if self.revision != "main" and not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise BenchmarkConfigurationError("Reranker revision must be main or a full model commit SHA")


class _TransformersBackend:
    def __init__(self, config, torch, tokenizer_class, model_class):
        self.torch = torch
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        cuda = torch.cuda.is_available()
        self.device = "cuda" if config.device == "auto" and cuda else ("cpu" if config.device == "auto" else config.device)
        if self.device == "cuda" and not cuda:
            raise ValueError("CUDA unavailable")
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        kwargs = dict(revision=config.revision, cache_dir=config.cache_dir, token=False, trust_remote_code=False)
        self.model = model_class.from_pretrained(config.model, **kwargs, use_safetensors=True,
                                                 torch_dtype=torch.float32, attn_implementation="eager")
        self.model.to(self.device)
        self.model.eval()
        revision = getattr(self.model.config, "_commit_hash", None)
        self.revision = revision.lower() if isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision) else None
        # Keep tokenizer and weights on the same resolved snapshot when available.
        self.tokenizer = tokenizer_class.from_pretrained(config.model, **{**kwargs, "revision": self.revision or config.revision})

    def score_pairs(self, pairs, *, max_length):
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs, return_dict=True).logits
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError("Expected one relevance logit per pair")
        return logits[:, 0].float().cpu().tolist()


def _load_backend(config):
    missing = False
    import_failed = False
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        missing = True
    except Exception:
        import_failed = True
    if missing:
        raise RerankerRuntimeError("Optional reranker dependencies unavailable; install requirements-reranker.txt in the separate runtime")
    if import_failed:
        raise RerankerRuntimeError("Optional reranker runtime import failed; check separate runtime installation")
    failed = False
    try:
        backend = _TransformersBackend(config, torch, AutoTokenizer, AutoModelForSequenceClassification)
    except Exception:
        failed = True
    if failed:
        raise RerankerRuntimeError("Local reranker initialization failed; check optional runtime, model cache and device")
    return backend


class BGEReranker:
    name = "bge-reranker-v2-m3"

    def __init__(self, config: BGERerankerConfig = BGERerankerConfig(), *, backend=None):
        self.config, self._backend = config, backend
        self.load_latency_seconds = 0.0

    def prepare(self):
        if self._backend is None:
            started = perf_counter()
            self._backend = _load_backend(self.config)
            self.load_latency_seconds = perf_counter() - started

    @property
    def metadata(self):
        revision = getattr(self._backend, "revision", None)
        resolved = revision.lower() if isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision) else None
        return {"reranker_model": self.config.model,
                "reranker_device": getattr(self._backend, "device", self.config.device),
                "reranker_batch_size": self.config.batch_size, "reranker_max_length": self.config.max_length,
                "reranker_revision": self.config.revision.lower(),
                "reranker_revision_resolved": resolved,
                "reranker_requested_revision": self.config.revision.lower(),
                "reranker_resolved_revision": resolved,
                "reranker_load_latency_seconds": self.load_latency_seconds,
                "reranker_score_type": "raw relevance logit", "reranker_precision": "float32"}

    def rerank(self, query, candidates):
        if not candidates:
            return ()
        self.prepare()
        failed = False
        try:
            scores = []
            for offset in range(0, len(candidates), self.config.batch_size):
                batch = candidates[offset:offset + self.config.batch_size]
                pairs = [[query, item.chunk.text] for item in batch]
                values = self._backend.score_pairs(pairs, max_length=self.config.max_length)
                values = [float(value) for value in values]
                if len(values) != len(batch) or any(not isfinite(value) for value in values):
                    raise ValueError
                scores.extend(values)
            order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
            result = tuple(replace(candidates[i], score=scores[i]) for i in order)
        except Exception:
            failed = True
        if failed:
            raise RerankerRuntimeError("Local reranker scoring failed")
        return result


def reranker_from_environment():
    name = os.environ.get("BENCHMARK_RERANKER", "none")
    if name == "none":
        return None
    if name != "bge-reranker-v2-m3":
        raise BenchmarkConfigurationError("BENCHMARK_RERANKER must be none or bge-reranker-v2-m3")
    failed = False
    try:
        config = BGERerankerConfig(
            model=os.environ.get("BGE_RERANKER_MODEL", MODEL),
            device=os.environ.get("BGE_RERANKER_DEVICE", "auto"),
            batch_size=int(os.environ.get("BGE_RERANKER_BATCH_SIZE", "4")),
            max_length=int(os.environ.get("BGE_RERANKER_MAX_LENGTH", "512")),
            revision=os.environ.get("BGE_RERANKER_REVISION", "main"),
            cache_dir=os.environ.get("HF_HOME", ".model-cache/huggingface"),
        )
    except BenchmarkConfigurationError:
        raise
    except (ValueError, OverflowError):
        failed = True
    if failed:
        raise BenchmarkConfigurationError("Invalid numeric reranker configuration")
    return BGEReranker(config)
