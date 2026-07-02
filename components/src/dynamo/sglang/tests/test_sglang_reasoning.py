# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from dynamo.sglang import reasoning

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


class _FakeReasoningParser:
    defaults = {
        "always": "always",
        "nemotron_3": "always",
        "default-on": "enable_thinking",
        "explicit": "explicit_enable_thinking",
        "mistral": "mistral",
    }

    def __init__(self, model_type, stream_reasoning=False):
        del stream_reasoning
        try:
            default = self.defaults[model_type]
        except KeyError as exc:
            raise ValueError("unsupported parser") from exc
        self.detector = SimpleNamespace(
            reasoning_default=default,
            think_end_token="</think>",
        )


class _FakeTokenizer:
    def __init__(self, think_end_ids=None, *, xgrammar_supported=True):
        self.think_end_ids = [13] if think_end_ids is None else think_end_ids
        self.xgrammar_supported = xgrammar_supported

    def encode(self, text, add_special_tokens=False):
        assert text == "</think>"
        assert add_special_tokens is False
        return self.think_end_ids

    def init_xgrammar(self):
        tokenizer_info = object() if self.xgrammar_supported else None
        return tokenizer_info, None


_DEFAULT_TOKENIZER = _FakeTokenizer()


class _Context:
    trace_id = "reasoning-test"

    def id(self):
        return self.trace_id

    def is_stopped(self):
        return False


async def _empty_stream():
    if False:  # pragma: no cover
        yield {}


def _capturing_engine(captured: dict[str, Any]):
    async def async_generate(require_reasoning=False, **kwargs):
        captured.update(kwargs)
        captured["require_reasoning"] = require_reasoning
        return _empty_stream()

    return SimpleNamespace(
        async_generate=async_generate,
        server_args=SimpleNamespace(
            reasoning_parser="nemotron_3", grammar_backend="xgrammar"
        ),
        tokenizer_manager=SimpleNamespace(
            tokenizer=_DEFAULT_TOKENIZER,
            model_config=SimpleNamespace(
                # SGLang sets this only on the scheduler process's ModelConfig.
                think_end_id=None,
                vocab_size=128,
                hf_eos_token_id={2},
            ),
        ),
        template_manager=SimpleNamespace(reasoning_config=None),
    )


def _engine(
    *,
    parser: str = "nemotron_3",
    config=None,
    grammar_backend: str = "xgrammar",
    tokenizer=_DEFAULT_TOKENIZER,
    supports_kwarg: bool = True,
    dllm_algorithm: str | None = None,
    skip_tokenizer_init: bool = False,
):
    if supports_kwarg:

        async def async_generate(require_reasoning=False):
            del require_reasoning

    else:

        async def async_generate(input_ids=None):
            del input_ids

    return SimpleNamespace(
        async_generate=async_generate,
        server_args=SimpleNamespace(
            reasoning_parser=parser,
            grammar_backend=grammar_backend,
            dllm_algorithm=dllm_algorithm,
            skip_tokenizer_init=skip_tokenizer_init,
        ),
        tokenizer_manager=SimpleNamespace(
            tokenizer=tokenizer,
            model_config=SimpleNamespace(
                # The Engine/TokenizerManager copy remains unset in SGLang 0.5.14.
                think_end_id=None,
                vocab_size=128,
                hf_eos_token_id={2},
            ),
        ),
        template_manager=SimpleNamespace(reasoning_config=config),
    )


@pytest.fixture(autouse=True)
def _fake_native_parser(monkeypatch):
    reasoning._reasoning_default.cache_clear()
    monkeypatch.setattr(
        reasoning, "_reasoning_parser_class", lambda: _FakeReasoningParser
    )
    yield
    reasoning._reasoning_default.cache_clear()


@pytest.mark.parametrize(
    ("reasoning_ended", "expected"),
    [(False, True), (True, False)],
)
def test_explicit_reasoning_ended_metadata_wins(reasoning_ended, expected):
    request = {
        "extra_args": {
            "reasoning_ended": reasoning_ended,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"enable_thinking": not expected}
            },
        }
    }

    assert reasoning.resolve_require_reasoning(_engine(), request) is expected


def test_root_reasoning_ended_overrides_extra_args():
    request = {
        "reasoning_ended": False,
        "extra_args": {"reasoning_ended": True},
    }

    assert reasoning.resolve_require_reasoning(_engine(), request) is True


def test_root_reasoning_parser_kwargs_override_extra_args():
    config = SimpleNamespace(
        special_case=None,
        toggle_param="enable_thinking",
        default_enabled=False,
    )
    request = {
        "reasoning_parser_kwargs": {"chat_template_kwargs": {"enable_thinking": True}},
        "extra_args": {
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"enable_thinking": False}
            },
        },
    }

    assert reasoning.resolve_require_reasoning(_engine(config=config), request) is True


def test_root_reasoning_effort_overrides_normalized_template_metadata():
    request = {
        "reasoning_effort": "none",
        "extra_args": {
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            }
        },
    }

    assert not reasoning.resolve_require_reasoning(_engine(parser="mistral"), request)


def test_tokenizer_mode_root_kwargs_disable_default_on_reasoning():
    request = {"chat_template_kwargs": {"enable_thinking": False}}

    assert (
        reasoning.resolve_require_reasoning(_engine(parser="default-on"), request)
        is False
    )


def test_tokenizer_mode_root_kwargs_enable_explicit_reasoning():
    request = {"chat_template_kwargs": {"enable_thinking": True}}

    assert (
        reasoning.resolve_require_reasoning(_engine(parser="explicit"), request) is True
    )


def test_tokenizer_mode_legacy_args_override_root_kwargs():
    request = {
        "chat_template_kwargs": {"enable_thinking": True},
        "chat_template_args": {"enable_thinking": False},
    }

    assert (
        reasoning.resolve_require_reasoning(_engine(parser="default-on"), request)
        is False
    )


@pytest.mark.parametrize(
    ("parser", "chat_template_kwargs", "expected"),
    [
        ("always", {}, True),
        ("default-on", {}, True),
        ("default-on", {"enable_thinking": False}, False),
        ("explicit", {}, False),
        ("explicit", {"enable_thinking": True}, True),
        ("mistral", {"reasoning_effort": "high"}, True),
        ("mistral", {"reasoning_effort": "none"}, False),
    ],
)
def test_detector_defaults_resolve_request_reasoning(
    parser, chat_template_kwargs, expected
):
    request = {
        "extra_args": {
            "reasoning_parser_kwargs": {"chat_template_kwargs": chat_template_kwargs}
        }
    }

    assert (
        reasoning.resolve_require_reasoning(_engine(parser=parser), request) is expected
    )


@pytest.mark.parametrize(
    ("default_enabled", "chat_template_kwargs", "expected"),
    [
        (True, {}, True),
        (True, {"enable_thinking": False}, False),
        (False, {}, False),
        (False, {"enable_thinking": True}, True),
    ],
)
def test_template_reasoning_config_resolves_toggle(
    default_enabled, chat_template_kwargs, expected
):
    config = SimpleNamespace(
        special_case=None,
        toggle_param="enable_thinking",
        default_enabled=default_enabled,
    )
    request = {
        "extra_args": {
            "reasoning_parser_kwargs": {"chat_template_kwargs": chat_template_kwargs}
        }
    }

    assert (
        reasoning.resolve_require_reasoning(_engine(config=config), request) is expected
    )


def test_request_reasoning_kwarg_is_filtered_for_older_sglang():
    request = {"extra_args": {"reasoning_ended": False}}

    assert reasoning.request_reasoning_kwargs(_engine(), request) == {
        "require_reasoning": True
    }
    assert (
        reasoning.request_reasoning_kwargs(_engine(supports_kwarg=False), request) == {}
    )


@pytest.mark.parametrize(
    "engine_changes,dyn_parser,expected",
    [
        ({}, "nemotron_v3", True),
        ({}, None, False),
        ({"grammar_backend": "outlines"}, "nemotron_v3", True),
        ({"grammar_backend": "llguidance"}, "nemotron_v3", True),
        ({"grammar_backend": "none"}, "nemotron_v3", False),
        ({"grammar_backend": "custom"}, "nemotron_v3", False),
        ({"tokenizer": None}, "nemotron_v3", False),
        ({"tokenizer": _FakeTokenizer(think_end_ids=[])}, "nemotron_v3", False),
        (
            {"tokenizer": _FakeTokenizer(think_end_ids=[13, 14])},
            "nemotron_v3",
            False,
        ),
        (
            {"tokenizer": _FakeTokenizer(xgrammar_supported=False)},
            "nemotron_v3",
            False,
        ),
        ({"supports_kwarg": False}, "nemotron_v3", False),
        ({"dllm_algorithm": "low_confidence"}, "nemotron_v3", False),
        ({"skip_tokenizer_init": True}, "nemotron_v3", False),
        ({"parser": "unknown"}, "nemotron_v3", False),
    ],
)
def test_capability_detection_fails_closed(engine_changes, dyn_parser, expected):
    engine = _engine(**engine_changes)
    dynamo_args = SimpleNamespace(dyn_reasoning_parser=dyn_parser)

    live_engine_result = reasoning.supports_reasoning_aware_guided_decoding(
        engine, engine.server_args, dynamo_args
    )
    component_result = reasoning.supports_reasoning_aware_guided_decoding_components(
        native_parser=engine.server_args.reasoning_parser,
        dynamo_parser=dyn_parser,
        server_args=engine.server_args,
        tokenizer=engine.tokenizer_manager.tokenizer,
        model_config=engine.tokenizer_manager.model_config,
        async_generate=engine.async_generate,
    )

    assert live_engine_result is expected
    assert component_result is expected


def test_capability_rejects_incompatible_native_and_dynamo_parsers():
    engine = _engine(parser="default-on")

    assert not reasoning.supports_reasoning_aware_guided_decoding(
        engine,
        engine.server_args,
        SimpleNamespace(dyn_reasoning_parser="gpt_oss"),
    )


def test_hyphenated_minimax_nom_alias_is_compatible():
    assert reasoning.reasoning_parsers_compatible("minimax-m3", "minimax-m3-nom")


def test_custom_override_of_builtin_grammar_backend_fails_closed(monkeypatch):
    from sglang.srt.constrained.base_grammar_backend import GRAMMAR_BACKEND_REGISTRY

    monkeypatch.setitem(GRAMMAR_BACKEND_REGISTRY, "xgrammar", object())
    engine = _engine()

    assert not reasoning.supports_reasoning_aware_guided_decoding(
        engine, engine.server_args, SimpleNamespace(dyn_reasoning_parser=None)
    )


def test_capability_probe_is_cached_on_engine(monkeypatch):
    calls = 0

    def compatible(*_args):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(reasoning, "_xgrammar_tokenizer_supported", compatible)
    engine = _engine()
    args = SimpleNamespace(dyn_reasoning_parser="nemotron_v3")

    assert reasoning.supports_reasoning_aware_guided_decoding(
        engine, engine.server_args, args
    )
    assert reasoning.supports_reasoning_aware_guided_decoding(
        engine, engine.server_args, args
    )
    assert calls == 1


def test_runtime_data_and_json_publication():
    engine = _engine()
    dynamo_args = SimpleNamespace(dyn_reasoning_parser="nemotron_v3")
    calls = []
    runtime_config = SimpleNamespace(
        set_engine_specific=lambda key, value: calls.append((key, value))
    )

    assert reasoning.reasoning_aware_guided_decoding_runtime_data(
        engine, engine.server_args, dynamo_args
    ) == {reasoning.REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY: True}
    assert (
        reasoning.publish_reasoning_aware_guided_decoding(
            runtime_config, engine, engine.server_args, dynamo_args
        )
        is True
    )
    assert calls == [(reasoning.REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY, "true")]


def test_surface_without_generation_engine_fails_publication_closed():
    server_args = _engine().server_args
    dynamo_args = SimpleNamespace(dyn_reasoning_parser="nemotron_v3")
    calls = []
    runtime_config = SimpleNamespace(
        set_engine_specific=lambda key, value: calls.append((key, value))
    )

    assert not reasoning.publish_reasoning_aware_guided_decoding(
        runtime_config, None, server_args, dynamo_args
    )
    assert calls == []


def test_unified_runtime_data_advertises_capability_only_for_generation(
    monkeypatch,
):
    from dynamo.common.constants import DisaggregationMode
    from dynamo.sglang import llm_engine

    monkeypatch.setattr(llm_engine, "get_sglang_worker_group_id", lambda _args: None)
    monkeypatch.setattr(
        llm_engine,
        "reasoning_aware_guided_decoding_runtime_data",
        lambda *_args: {reasoning.REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY: True},
    )
    kwargs = {
        "engine": object(),
        "dynamo_args": object(),
    }

    assert (
        llm_engine._get_runtime_data(
            object(), serving_mode=DisaggregationMode.PREFILL, **kwargs
        )
        is None
    )
    assert llm_engine._get_runtime_data(
        object(), serving_mode=DisaggregationMode.DECODE, **kwargs
    ) == {reasoning.REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY: True}


@pytest.mark.asyncio
async def test_unified_engine_forwards_require_reasoning():
    from dynamo.common.constants import DisaggregationMode
    from dynamo.sglang.llm_engine import SglangLLMEngine

    captured = {}
    handler = SglangLLMEngine.__new__(SglangLLMEngine)
    handler.engine = _capturing_engine(captured)
    handler.serving_mode = DisaggregationMode.AGGREGATED
    handler.enable_trace = False
    handler._logits_processor_spec = None
    handler._dp_start = 0
    handler._dp_size = 1
    handler._build_sampling_params = lambda _request: {}
    handler._get_input_param = lambda _request: {"input_ids": [1, 2, 3]}

    request = {"extra_args": {"reasoning_ended": False}}
    assert [chunk async for chunk in handler.generate(request, _Context())] == []
    assert captured["require_reasoning"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("serving_mode", ["aggregated", "decode"])
async def test_legacy_decode_forwards_require_reasoning(serving_mode):
    from dynamo.common.constants import DisaggregationMode
    from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

    captured = {}
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.engine = _capturing_engine(captured)
    handler.serving_mode = (
        DisaggregationMode.AGGREGATED
        if serving_mode == "aggregated"
        else DisaggregationMode.DECODE
    )
    handler.use_sglang_tokenizer = False
    handler.enable_trace = False
    handler.config = SimpleNamespace(dynamo_args=SimpleNamespace(enable_rl=False))
    handler._routed_experts_kwargs = {}
    handler._enable_frontend_decoding = False
    handler._image_loader = None
    handler._mm_hashes_supported = False
    handler._get_input_param = lambda _request: {"input_ids": [1, 2, 3]}
    handler._resolve_lora = lambda _request: None
    handler._session_kwargs = lambda _request: {}
    handler._priority_kwargs = lambda _priority: {}

    async def process_stream(*_args, **_kwargs):
        if False:  # pragma: no cover
            yield {}

    handler._process_token_stream = process_stream
    request = {
        "token_ids": [1, 2, 3],
        "extra_args": {"reasoning_ended": False},
        "bootstrap_info": {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": 1234,
            "bootstrap_room": 1,
        },
    }

    assert [chunk async for chunk in handler.generate(request, _Context())] == []
    assert captured["require_reasoning"] is True


@pytest.mark.asyncio
async def test_legacy_prefill_forwards_require_reasoning():
    from dynamo.common.constants import DisaggregationMode
    from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler

    captured = {}
    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = _capturing_engine(captured)
    handler.serving_mode = DisaggregationMode.PREFILL
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False
    handler._consume_tasks = set()
    handler._get_input_param = lambda _request: {"input_ids": [1, 2, 3]}
    handler._resolve_lora = lambda _request: None
    handler._session_kwargs = lambda _request: {}
    handler._priority_kwargs = lambda _priority: {}

    async def consume_results(*_args, **_kwargs):
        return None

    handler._consume_results = consume_results
    inner_request = {
        "token_ids": [1, 2, 3],
        "extra_args": {"reasoning_ended": False},
        "bootstrap_info": {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": 1234,
            "bootstrap_room": 1,
        },
    }
    request = {"request": inner_request, "sampling_params": {}}

    chunks = [chunk async for chunk in handler.generate(request, _Context())]
    assert len(chunks) == 1
    assert captured["require_reasoning"] is True


@pytest.mark.asyncio
async def test_diffusion_llm_forwards_require_reasoning():
    from dynamo.sglang.request_handlers.llm.diffusion_handler import (
        DiffusionWorkerHandler,
    )

    captured = {}
    handler = DiffusionWorkerHandler.__new__(DiffusionWorkerHandler)
    handler.engine = _capturing_engine(captured)
    handler.use_sglang_tokenizer = False
    handler.enable_trace = False
    handler._get_input_param = lambda _request: {"input_ids": [1, 2, 3]}
    handler._build_sampling_params = lambda _request: {}

    async def process_stream(*_args, **_kwargs):
        if False:  # pragma: no cover
            yield {}

    handler._process_token_stream = process_stream
    request = {
        "token_ids": [1, 2, 3],
        "extra_args": {"reasoning_ended": False},
    }

    assert [chunk async for chunk in handler.generate(request, _Context())] == []
    assert captured["require_reasoning"] is True


def _multimodal_request():
    from dynamo.sglang.protocol import PreprocessedRequest, SglangMultimodalRequest

    return SglangMultimodalRequest(
        request=PreprocessedRequest(
            token_ids=[1, 2, 3],
            stop_conditions={"max_tokens": 8},
            sampling_options={"guided_decoding": {"json": {"type": "array"}}},
            extra_args={"reasoning_ended": False},
        )
    )


def test_multimodal_sampling_preserves_guided_decoding():
    from dynamo.sglang.request_handlers.multimodal.worker_handler import SglangUtils

    sampling_params = SglangUtils.build_sampling_params(_multimodal_request())

    assert json.loads(sampling_params["json_schema"]) == {"type": "array"}


def test_multimodal_protocol_preserves_reasoning_metadata():
    from dynamo.sglang.protocol import PreprocessedRequest

    request = PreprocessedRequest.model_validate(
        {
            "token_ids": [1],
            "stop_conditions": {},
            "sampling_options": {},
            "extra_args": {
                "reasoning_parser_kwargs": {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            },
            "reasoning_ended": True,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"enable_thinking": True}
            },
            "reasoning_effort": "high",
        }
    )

    assert request.extra_args["reasoning_parser_kwargs"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert request.reasoning_ended is True
    assert request.reasoning_parser_kwargs == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert request.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_multimodal_aggregated_forwards_require_reasoning(monkeypatch):
    from dynamo.sglang.request_handlers.multimodal import worker_handler

    async def no_multimodal_items(*_args):
        return [], [], None, None

    monkeypatch.setattr(worker_handler, "_build_mm_items", no_multimodal_items)
    captured = {}
    handler = worker_handler.MultimodalWorkerHandler.__new__(
        worker_handler.MultimodalWorkerHandler
    )
    handler.engine = _capturing_engine(captured)
    handler.embeddings_processor = object()
    handler.enable_trace = False

    chunks = [
        chunk
        async for chunk in handler._generate_aggregated(
            _multimodal_request(), lambda: None
        )
    ]

    assert chunks == []
    assert captured["require_reasoning"] is True


@pytest.mark.asyncio
async def test_multimodal_decode_forwards_require_reasoning():
    from dynamo.sglang.request_handlers.multimodal import worker_handler

    captured = {}
    handler = worker_handler.MultimodalWorkerHandler.__new__(
        worker_handler.MultimodalWorkerHandler
    )
    handler.engine = _capturing_engine(captured)
    handler.enable_trace = False

    async def bootstrap(*_args, **_kwargs):
        return {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": 1234,
            "bootstrap_room": 7,
        }

    handler._get_bootstrap_from_prefill = bootstrap
    chunks = [
        chunk
        async for chunk in handler._generate_disaggregated(
            _multimodal_request(), lambda: None
        )
    ]

    assert chunks == []
    assert captured["require_reasoning"] is True


@pytest.mark.asyncio
async def test_multimodal_prefill_forwards_require_reasoning(monkeypatch):
    from dynamo.sglang.protocol import DisaggSglangMultimodalRequest
    from dynamo.sglang.request_handlers.multimodal import worker_handler

    async def no_multimodal_items(*_args):
        return [], [], None, None

    monkeypatch.setattr(worker_handler, "_build_mm_items", no_multimodal_items)
    captured = {}
    handler = worker_handler.MultimodalPrefillWorkerHandler.__new__(
        worker_handler.MultimodalPrefillWorkerHandler
    )
    handler.engine = _capturing_engine(captured)
    handler.embeddings_processor = object()
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler.enable_trace = False

    request = DisaggSglangMultimodalRequest(
        request=_multimodal_request(), sampling_params={}
    )
    await handler._process_prefill_generation(request, bootstrap_room=7)
    await asyncio.sleep(0)

    assert captured["require_reasoning"] is True
