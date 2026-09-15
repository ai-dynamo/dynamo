# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused benchmark for cumulative TensorRT-LLM completion logprob streaming.

The benchmark supplies a small TensorRT-LLM-shaped async result to the real
``AggregatedHandler.generate`` path.  The import shim keeps this CPU benchmark
independent of the native Dynamo and TensorRT-LLM wheels; it does not replace
the handler or its response loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import os
import sys
import types
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.core,
    pytest.mark.post_merge,
    pytest.mark.benchmark,
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        os.environ.get("DYN_RUN_LOGPROBS_BENCHMARK") != "1",
        reason="run explicitly when collecting the logprob benchmark",
    ),
]

_TOKEN_COUNT = 1024
_SHIM_NAMESPACE_ROOTS = ("torch", "tensorrt_llm", "dynamo")


class _Stub:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


class _Tensor:
    pass


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _package(name: str, path: Path | None = None):
    package = sys.modules.get(name) or types.ModuleType(name)
    package.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = package
    return package


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _is_shim_module(name: str) -> bool:
    return any(
        name == root or name.startswith(f"{root}.") for root in _SHIM_NAMESPACE_ROOTS
    )


@contextmanager
def _isolated_shim_modules():
    """Keep the benchmark's dependency shims out of the pytest process."""
    saved_modules = {
        name: module for name, module in sys.modules.items() if _is_shim_module(name)
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    try:
        yield
    finally:
        for name in list(sys.modules):
            if _is_shim_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


@pytest.fixture
def aggregated_handler():
    with _isolated_shim_modules():
        yield _load_aggregated_handler()


def _load_aggregated_handler():
    """Load the real TRT-LLM handlers with only native dependencies shimmed."""
    source_root = Path(__file__).resolve().parents[1] / "components/src"

    _module("torch", Tensor=_Tensor)
    _package("tensorrt_llm")
    _package("tensorrt_llm.executor")

    class _RequestError(Exception):
        pass

    _module("tensorrt_llm.executor.request", DEFAULT_REQUEST_PRIORITY=0.5)
    _module(
        "tensorrt_llm.executor.result",
        GenerationResult=_Stub,
        Logprob=_Stub,
    )
    _module("tensorrt_llm.executor.utils", RequestError=_RequestError)
    _module(
        "tensorrt_llm.llmapi",
        DisaggregatedParams=_Stub,
        ConversationParams=_Stub,
    )
    _module("tensorrt_llm.llmapi.llm", SamplingParams=_Stub)
    _module(
        "tensorrt_llm.llmapi.disagg_utils",
        get_global_disagg_request_id=lambda *args, **kwargs: 1,
    )
    _module("tensorrt_llm.sampling_params", GuidedDecodingParams=_Stub)
    _module("tensorrt_llm.scheduling_params", SchedulingParams=_Stub)

    _package("dynamo", source_root / "dynamo")
    _package("dynamo.common", source_root / "dynamo/common")
    backend_package = _package("dynamo.common.backend")
    backend_package.logprobs = _load_module(
        "dynamo.common.backend.logprobs",
        source_root / "dynamo/common/backend/logprobs.py",
    )

    class _Context:
        pass

    _module("dynamo._core", Client=_Stub, Context=_Context)

    class _CommonDisaggregationMode(Enum):
        AGGREGATED = "agg"
        PREFILL = "prefill"
        DECODE = "decode"
        ENCODE = "encode"

    _module(
        "dynamo.common.constants",
        DisaggregationMode=_CommonDisaggregationMode,
    )
    _module("dynamo.common.backend.engine", is_generation_stage=lambda value: False)
    _module(
        "dynamo.common.multimodal.cache_uuid",
        reject_unsupported_multimodal_uuids=lambda value: None,
    )
    _package("dynamo.common.utils")
    _module(
        "dynamo.common.utils.structural_tag",
        serialize_structural_tag=lambda value: value,
    )
    _module("dynamo.health_check", HEALTH_CHECK_KEY="health_check")

    class _EngineShutdown(Exception):
        pass

    _package("dynamo.llm")
    _module("dynamo.llm.exceptions", EngineShutdown=_EngineShutdown)
    _module(
        "dynamo.logits_processing.examples",
        HelloWorldLogitsProcessor=_Stub,
    )
    _module("dynamo.nixl_connect", Connector=_Stub)
    _module("dynamo.runtime", DistributedRuntime=_Stub)
    _module("dynamo.runtime.logging", configure_dynamo_logging=lambda: None)

    _package("dynamo.trtllm", source_root / "dynamo/trtllm")
    _load_module(
        "dynamo.trtllm.constants",
        source_root / "dynamo/trtllm/constants.py",
    )
    _package(
        "dynamo.trtllm.request_handlers",
        source_root / "dynamo/trtllm/request_handlers",
    )
    _package("dynamo.trtllm.utils", source_root / "dynamo/trtllm/utils")
    _module(
        "dynamo.trtllm.conversation_affinity",
        CONVERSATION_PARAMS_AVAILABLE=False,
        conversation_params_for=lambda value: None,
        engine_conversation_affinity_enabled=lambda value: False,
        session_id_from_request=lambda value: None,
    )
    _module("dynamo.trtllm.engine", TensorRTLLMEngine=_Stub)
    _module(
        "dynamo.trtllm.logits_processing.adapter",
        create_trtllm_adapters=lambda value: [],
    )
    _module("dynamo.trtllm.metrics", AdditionalMetricsCollector=_Stub)
    _module("dynamo.trtllm.multimodal_processor", MultimodalRequestProcessor=_Stub)
    _module("dynamo.trtllm.publisher", Publisher=_Stub)

    class _BaseGenerativeHandler:
        pass

    _module(
        "dynamo.trtllm.request_handlers.base_generative_handler",
        BaseGenerativeHandler=_BaseGenerativeHandler,
    )
    _module(
        "dynamo.trtllm.utils.disagg_utils",
        DisaggregatedParams=_Stub,
        DisaggregatedParamsCodec=_Stub,
        get_compatible_global_disagg_request_id=lambda value: 1,
    )
    _module(
        "dynamo.trtllm.utils.request_utils",
        apply_stop_conditions_to_sampling_params=lambda *args: None,
        normalize_top_k_for_trtllm=lambda value: value,
        request_cache_salt=lambda value: None,
    )

    handler_base = _load_module(
        "dynamo.trtllm.request_handlers.handler_base",
        source_root / "dynamo/trtllm/request_handlers/handler_base.py",
    )
    sys.modules["dynamo.trtllm.request_handlers"].handler_base = handler_base

    _package("dynamo.common.memory")
    _module(
        "dynamo.common.memory.multimodal_embedding_cache_manager",
        MultimodalEmbeddingCacheManager=_Stub,
    )
    _package("dynamo.common.multimodal")
    _module(
        "dynamo.trtllm.multimodal.embedding_fetcher",
        fetch_embeddings_from_encoder=None,
    )
    _package("dynamo.trtllm.multimodal")
    _module(
        "dynamo.trtllm.request_handlers.push_egress",
        push_egress_capable=lambda function: function,
    )

    aggregated_path = (
        source_root / "dynamo/trtllm/request_handlers/aggregated_handler.py"
    )
    aggregated_handler = _load_module(
        "dynamo.trtllm.request_handlers.aggregated_handler",
        aggregated_path,
    )
    handler_path = source_root / "dynamo/trtllm/request_handlers/handler_base.py"
    if Path(handler_base.__file__).resolve() != handler_path.resolve():
        raise RuntimeError(
            "benchmark loaded HandlerBase outside the checked-out source"
        )
    if Path(aggregated_handler.__file__).resolve() != aggregated_path.resolve():
        raise RuntimeError(
            "benchmark loaded AggregatedHandler outside the checked-out source"
        )
    if handler_base.HandlerBase._generate_locally_impl.__code__.co_filename != str(
        handler_path
    ):
        raise RuntimeError("benchmark did not bind _generate_locally_impl from source")
    if aggregated_handler.AggregatedHandler.generate.__code__.co_filename != str(
        aggregated_path
    ):
        raise RuntimeError(
            "benchmark did not bind AggregatedHandler.generate from source"
        )
    return aggregated_handler.AggregatedHandler


@dataclasses.dataclass
class _SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    n: int | None = None
    best_of: int = 1
    ignore_eos: bool = False
    guided_decoding: object | None = None
    prompt_logprobs: int | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    min_tokens: int | None = None
    stop_token_ids: list[int] | None = None

    def __post_init__(self):
        if self.n is not None and self.best_of < self.n:
            raise ValueError("best_of must be at least n")


@dataclasses.dataclass
class _Logprob:
    logprob: float
    rank: int
    decoded_token: str


@dataclasses.dataclass
class _Output:
    token_ids: list[int]
    logprobs: list[dict[int, _Logprob]]
    finish_reason: str | None
    stop_reason: str | None = None
    prompt_logprobs: list = dataclasses.field(default_factory=list)
    request_perf_metrics: object | None = None
    index: int = 0


@dataclasses.dataclass
class _GenerationChunk:
    outputs: list[_Output]
    finished: bool
    cached_tokens: int = 0
    metrics_dict: object | None = None


class _GenerationResult:
    request_id = "logprob-benchmark"

    def __init__(self, outputs: list[_Output]):
        self._outputs = outputs

    def abort(self):
        pass

    def __aiter__(self):
        async def iterate():
            last = len(self._outputs) - 1
            for index, output in enumerate(self._outputs):
                yield _GenerationChunk([output], index == last)

        return iterate()


class _LLM:
    args = SimpleNamespace()

    def __init__(self, outputs: list[_Output]):
        self._outputs = outputs

    def generate_async(self, **kwargs):
        return _GenerationResult(self._outputs)


class _Engine:
    def __init__(self, outputs: list[_Output]):
        self.llm = _LLM(outputs)


class _Context:
    def id(self):
        return "logprob-benchmark"

    def trace_headers(self):
        return {}


@asynccontextmanager
async def _no_cancellation_monitor(*args, **kwargs):
    yield


def _cumulative_stream() -> list[_Output]:
    return [
        _Output(
            token_ids=list(range(token_count)),
            logprobs=[
                {
                    token_id: _Logprob(-0.1, rank=1, decoded_token="x"),
                    token_id + _TOKEN_COUNT: _Logprob(-1.1, rank=2, decoded_token="y"),
                }
                for token_id in range(token_count)
            ],
            finish_reason="stop" if token_count == _TOKEN_COUNT else None,
        )
        for token_count in range(1, _TOKEN_COUNT + 1)
    ]


def test_cumulative_one_token_stream_logprob_extraction(benchmark, aggregated_handler):
    # These imports must remain after _load_aggregated_handler installs its shims.
    from dynamo.trtllm.constants import DisaggregationMode
    from dynamo.trtllm.request_handlers.handler_base import RequestHandlerConfig

    outputs = _cumulative_stream()
    handler = aggregated_handler(
        RequestHandlerConfig(
            engine=_Engine(outputs),
            default_sampling_params=_SamplingParams(),
            publisher=None,
            disaggregation_mode=DisaggregationMode.AGGREGATED,
        )
    )
    handler._cancellation_monitor = _no_cancellation_monitor
    context = _Context()
    loop = asyncio.new_event_loop()

    async def run_stream() -> int:
        request = {
            "token_ids": [1, 2, 3],
            "stop_conditions": {"max_tokens": _TOKEN_COUNT},
            "sampling_options": {},
            "output_options": {"logprobs": 1},
        }
        extracted_positions = 0
        async for chunk in handler.generate(request, context):
            extracted_positions += len(chunk["token_ids"])
            extracted_positions += len(chunk.get("log_probs", []))
            extracted_positions += len(chunk.get("top_logprobs", []))
        return extracted_positions

    def run_once() -> int:
        return loop.run_until_complete(run_stream())

    try:
        assert benchmark(run_once) == 3 * _TOKEN_COUNT
    finally:
        loop.close()
