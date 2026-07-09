# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import (
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)

from dynamo.vllm.engine_generate import (
    EngineGenerateRequest,
    build_prompt,
    merge_kv_transfer_params,
    priority,
    serialize_routed_experts,
)
from dynamo.vllm.handlers import BaseWorkerHandler, build_sampling_params
from dynamo.vllm.response_adapters import (
    GenerationResponseContext,
    LegacyResponseAdapter,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _request(sampling_params=None, *, provided=None, **top_level):
    sampling_params = sampling_params or {}
    payload = {
        "request_id": "request-1",
        "model": "test-model",
        "token_ids": [11, 22, 33],
        "sampling_params": sampling_params,
        **top_level,
    }
    return {
        "model": "test-model",
        "token_ids": [11, 22, 33],
        "generate_request": payload,
        "generate_sampling_fields": (
            list(sampling_params) if provided is None else provided
        ),
        "routing": {},
    }


def test_sampling_params_preserve_native_fields_and_nested_types():
    request = _request(
        {
            "temperature": 0.25,
            "max_tokens": 17,
            "logprobs": 2,
            "logit_bias": {"11": -1.5},
            "structured_outputs": {"regex": "[0-9]+"},
            "repetition_detection": {
                "max_pattern_size": 8,
                "min_pattern_size": 2,
                "min_count": 3,
            },
            "flat_logprobs": True,
            "skip_reading_prefix_cache": True,
        }
    )

    params = build_sampling_params(request, default_sampling_params={})

    assert params.temperature == 0.25
    assert params.max_tokens == 17
    assert params.logprobs == 2
    assert params.logit_bias == {11: -1.5}
    assert isinstance(params.structured_outputs, StructuredOutputsParams)
    assert params.structured_outputs.regex == "[0-9]+"
    assert isinstance(params.repetition_detection, RepetitionDetectionParams)
    assert params.repetition_detection.min_count == 3
    assert params.flat_logprobs is True
    assert params.skip_reading_prefix_cache is True
    assert params.output_kind is RequestOutputKind.DELTA


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("temperature", 1.0),
        ("n", 1),
        ("top_p", 1.0),
        ("min_tokens", 0),
        ("ignore_eos", False),
        ("repetition_penalty", 1.0),
        ("detokenize", True),
    ],
)
def test_explicit_null_nonnullable_sampling_fields_use_vllm_defaults(field, expected):
    request = _request({field: None})

    params = build_sampling_params(request, default_sampling_params={})

    assert getattr(params, field) == expected


def test_omitted_max_tokens_uses_server_default_with_context_cap():
    request = _request({"temperature": 0.5})
    params = build_sampling_params(
        request,
        default_sampling_params={"max_tokens": 100},
        model_max_len=64,
    )
    assert params.max_tokens == 61


def test_omitted_max_tokens_uses_dynamic_context_default_without_config():
    request = _request({"temperature": 0.5})

    params = build_sampling_params(
        request,
        default_sampling_params={},
        model_max_len=64,
    )

    assert params.max_tokens == 61


def test_explicit_null_max_tokens_is_not_replaced_by_server_default():
    request = _request({"max_tokens": None})
    params = build_sampling_params(
        request,
        default_sampling_params={"max_tokens": 100},
        model_max_len=64,
    )
    assert params.max_tokens is None


@pytest.mark.parametrize(
    ("wire_value", "expected"),
    [
        (None, None),
        (-1, None),
        (0, 0),
        (8, 8),
    ],
)
def test_thinking_token_budget_uses_vllm_normalization(wire_value, expected):
    request = _request({"thinking_token_budget": wire_value})

    params = build_sampling_params(request, default_sampling_params={})

    assert params.thinking_token_budget == expected


def test_omitted_thinking_token_budget_remains_unset():
    params = build_sampling_params(_request({}), default_sampling_params={})

    assert params.thinking_token_budget is None


def test_backend_extension_maps_are_combined_without_overwrite():
    request = _request(
        {
            "extra_args": {"caller": 1},
            "vllm_xargs": {"backend": 2},
        },
        kv_transfer_params={"connector": "mooncake"},
    )
    params = build_sampling_params(request, default_sampling_params={})
    assert params.extra_args == {
        "caller": 1,
        "backend": 2,
        "kv_transfer_params": {"connector": "mooncake"},
    }


def test_backend_extension_collision_is_rejected():
    request = _request(
        {
            "extra_args": {"duplicate": 1},
            "vllm_xargs": {"duplicate": 2},
        }
    )
    with pytest.raises(ValueError, match="duplicate backend sampling extension"):
        build_sampling_params(request, default_sampling_params={})


def test_framework_kv_metadata_merges_without_replacing_caller_fields():
    assert merge_kv_transfer_params({"caller": 1}, {"framework": 2}) == {
        "caller": 1,
        "framework": 2,
    }
    with pytest.raises(ValueError, match="collide"):
        merge_kv_transfer_params({"same": 1}, {"same": 2})


def test_text_prompt_preserves_exact_tokens_and_cache_salt():
    request = _request({}, cache_salt="checkpoint-7")
    prompt = build_prompt(request)
    assert prompt["prompt_token_ids"] == [11, 22, 33]
    assert prompt["prompt_token_ids"] is request["token_ids"]
    assert prompt["cache_salt"] == "checkpoint-7"


def test_multimodal_cache_hit_prompt_preserves_hashes_and_placeholders():
    request = _request(
        {},
        cache_salt="checkpoint-mm",
        features={
            "mm_hashes": {"image": ["hash-1"]},
            "mm_placeholders": {"image": [{"offset": 1, "length": 1}]},
            "kwargs_data": None,
        },
    )
    prompt = build_prompt(request)
    assert prompt["type"] == "multimodal"
    assert prompt["prompt_token_ids"] == [11, 22, 33]
    assert prompt["mm_hashes"] == {"image": ["hash-1"]}
    assert prompt["mm_placeholders"]["image"][0].offset == 1
    assert prompt["mm_kwargs"]["image"] == [None]
    assert prompt["cache_salt"] == "checkpoint-mm"


def test_multimodal_cache_miss_uses_scale_out_serde(monkeypatch: pytest.MonkeyPatch):
    decoded_item = object()
    module = ModuleType("vllm.entrypoints.scale_out.token_in_token_out.mm_serde")
    module.decode_mm_kwargs_item = lambda value: decoded_item  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    request = _request(
        {},
        features={
            "mm_hashes": {"image": ["hash-1"]},
            "mm_placeholders": {"image": [{"offset": 1, "length": 1}]},
            "kwargs_data": {"image": ["encoded-item"]},
        },
    )

    prompt = build_prompt(request)

    assert prompt["mm_kwargs"]["image"] == [decoded_item]


def test_priority_is_native_for_engine_generate_and_legacy_for_chat():
    request = _request({})
    request["generate_request"]["priority"] = -(2**31)
    assert priority(request, {"priority": 2**31 - 1}) == -(2**31)
    assert priority({}, {"priority": -4}) == 4


def test_request_adapter_owns_engine_prompt_sampling_and_priority():
    request = _request(
        {"temperature": 0.25, "max_tokens": 17}, cache_salt="checkpoint-7"
    )
    adapter = EngineGenerateRequest.from_request(request)

    assert adapter is not None
    assert adapter.build_prompt()["prompt_token_ids"] == [11, 22, 33]
    assert adapter.build_prompt()["cache_salt"] == "checkpoint-7"
    assert adapter.build_sampling_params({}, model_max_len=64).max_tokens == 17
    request["generate_request"]["priority"] = -4
    assert adapter.priority({"priority": 4}) == -4


def test_request_adapter_ignores_legacy_requests():
    assert EngineGenerateRequest.from_request({"token_ids": [11, 22, 33]}) is None


def test_routed_experts_have_one_vllm_compatible_base64_numpy_encoding():
    expected = np.array([[1, 2], [3, 4]], dtype=np.int32)
    encoded = serialize_routed_experts(expected)
    assert encoded is not None
    decoded = np.load(io.BytesIO(base64.b64decode(encoded)))
    np.testing.assert_array_equal(decoded, expected)


def test_legacy_response_adapter_preserves_finish_only_metadata():
    adapter = LegacyResponseAdapter()
    request_output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[41])],
        prompt_token_ids=[11, 22, 33],
        num_cached_tokens=2,
    )
    context = GenerationResponseContext(
        request_output=request_output,
        output_index=0,
        finished=False,
        sampling_params=SamplingParams(),
        embedding_sequence_length=None,
        completion_token_counts={0: 1},
        prompt_logprobs=None,
        routed_experts=None,
        kv_transfer_params=None,
    )
    output = {"token_ids": [41]}

    adapter.adapt(output, context)
    assert "completion_usage" not in output

    adapter.adapt(output, replace(context, finished=True))
    assert output["completion_usage"]["prompt_tokens_details"] == {"cached_tokens": 2}


class _FakeEngineClient:
    tokenizer = None

    def __init__(self, responses):
        self.responses = responses

    def generate(self, *args, **kwargs):
        async def _stream():
            for response in self.responses:
                yield response

        return _stream()


def test_engine_generate_worker_emits_usage_on_every_delta():
    responses = [
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=[41],
                    finish_reason=None,
                    stop_reason=None,
                    logprobs=None,
                    routed_experts=None,
                )
            ],
            prompt_token_ids=[11, 22, 33],
            prompt_logprobs=None,
            num_cached_tokens=0,
            kv_transfer_params=None,
        ),
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=[],
                    finish_reason="length",
                    stop_reason=None,
                    logprobs=None,
                    routed_experts=None,
                )
            ],
            prompt_token_ids=[11, 22, 33],
            prompt_logprobs=None,
            num_cached_tokens=0,
            kv_transfer_params=None,
        ),
    ]
    handler = SimpleNamespace(
        engine_client=_FakeEngineClient(responses),
        _extract_logprobs=BaseWorkerHandler._extract_logprobs,
        _log_with_lora_context=lambda *args, **kwargs: None,
    )

    async def collect_chunks():
        chunks = []
        async for chunk in BaseWorkerHandler.generate_tokens(
            handler,
            prompt={"prompt_token_ids": [11, 22, 33]},
            sampling_params=SamplingParams(max_tokens=2),
            request_id="request-1",
            response_adapter=EngineGenerateRequest.from_request(
                _request({})
            ).response_adapter(),
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect_chunks())

    assert chunks[0]["token_ids"] == [41]
    assert chunks[0]["completion_usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
        "prompt_tokens_details": None,
    }
    assert chunks[1]["token_ids"] == []
    assert chunks[1]["finish_reason"] == "length"
    assert chunks[1]["completion_usage"]["completion_tokens"] == 1
