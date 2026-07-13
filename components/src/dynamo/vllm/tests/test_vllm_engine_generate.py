# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from dynamo.vllm.handlers import (
    BaseWorkerHandler,
    PrefillWorkerHandler,
    _build_engine_generate_prompt,
    _engine_generate_priority,
    _merge_kv_transfer_params,
    _serialize_routed_experts_vllm,
    _use_prefill_prompt_logprobs,
    build_sampling_params,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _request(sampling_params=None, **top_level):
    sampling_params = sampling_params or {}
    envelope = {
        "request_id": "request-1",
        "model": "test-model",
        "sampling_params": sampling_params,
        **top_level,
    }
    return {
        "model": "test-model",
        "token_ids": [11, 22, 33],
        "extra_args": {"vllm_tito": envelope},
        "routing": {"priority": 4},
    }


def test_sampling_params_use_vllm_validation_and_internal_delta():
    request = _request(
        {
            "temperature": 0,
            "max_tokens": 17,
            "logprobs": 2,
            "some_future_field": {"enabled": True},
        }
    )

    params = build_sampling_params(request, default_sampling_params={})

    assert params.temperature == 0
    assert params.max_tokens == 17
    assert params.logprobs == 2
    assert params.output_kind is RequestOutputKind.DELTA


def test_omitted_max_tokens_uses_server_default_with_context_cap():
    params = build_sampling_params(
        _request({"temperature": 0}),
        default_sampling_params={"max_tokens": 100},
        model_max_len=64,
    )
    assert params.max_tokens == 61


def test_top_level_kv_metadata_merges_with_sampling_extensions():
    params = build_sampling_params(
        _request(
            {"extra_args": {"caller": 1}},
            kv_transfer_params={"connector": "mooncake"},
        ),
        default_sampling_params={},
    )
    assert params.extra_args == {
        "caller": 1,
        "kv_transfer_params": {"connector": "mooncake"},
    }


def test_framework_kv_metadata_merges_without_replacing_caller_fields():
    assert _merge_kv_transfer_params({"caller": 1}, {"framework": 2}) == {
        "caller": 1,
        "framework": 2,
    }
    with pytest.raises(ValueError, match="collide"):
        _merge_kv_transfer_params({"same": 1}, {"same": 2})


def test_pd_prompt_logprobs_are_composed_from_prefill():
    payload = [None, {"11": {"logprob": -0.25, "rank": 1}}]
    disaggregated = PrefillWorkerHandler._build_disaggregated_params(
        SimpleNamespace(),
        {"remote_engine_id": "prefill-1"},
        prompt_logprobs=payload,
    )
    assert disaggregated == {
        "kv_transfer_params": {"remote_engine_id": "prefill-1"},
        "prompt_logprobs": payload,
    }

    decode_params = SamplingParams(prompt_logprobs=1)
    assert _use_prefill_prompt_logprobs(decode_params, disaggregated, True) is payload
    assert decode_params.prompt_logprobs is None

    legacy_params = SamplingParams(prompt_logprobs=1)
    assert _use_prefill_prompt_logprobs(legacy_params, disaggregated, False) is None
    assert legacy_params.prompt_logprobs == 1


def test_text_prompt_preserves_exact_tokens_and_cache_salt():
    prompt = _build_engine_generate_prompt(_request({}, cache_salt="checkpoint-7"))
    assert prompt["prompt_token_ids"] == [11, 22, 33]
    assert prompt["cache_salt"] == "dynamo-cache-salt:checkpoint-7"


def test_prompt_rejects_envelope_core_token_mismatch():
    request = _request({})
    request["extra_args"]["vllm_tito"]["token_ids"] = [11, 22, 33]
    request["token_ids"] = [11, 22]
    with pytest.raises(ValueError, match="do not match"):
        _build_engine_generate_prompt(request)


def test_sampling_params_reconstruct_canonical_token_ids():
    request = _request({"temperature": 0, "max_tokens": 2})
    assert "token_ids" not in request["extra_args"]["vllm_tito"]

    params = build_sampling_params(request, default_sampling_params={})

    assert params.temperature == 0
    assert params.max_tokens == 2


def test_multimodal_cache_hit_prompt_preserves_hashes_and_placeholders():
    request = _request(
        {},
        cache_salt="checkpoint-mm",
        features={
            "mm_hashes": {"image": ["hash-1"]},
            "mm_placeholders": {
                "image": [
                    {
                        "offset": 0,
                        "length": 3,
                        "is_embed": [False, True, False],
                    },
                ]
            },
            "kwargs_data": None,
        },
    )
    prompt = _build_engine_generate_prompt(request)
    assert prompt["type"] == "multimodal"
    assert prompt["prompt_token_ids"] == [11, 22, 33]
    assert prompt["mm_hashes"] == {"image": ["hash-1"]}
    placeholder = prompt["mm_placeholders"]["image"][0]
    assert (placeholder.offset, placeholder.length) == (0, 3)
    assert placeholder.is_embed.tolist() == [False, True, False]
    assert prompt["mm_kwargs"]["image"] == [None]
    assert prompt["cache_salt"] == "dynamo-cache-salt:checkpoint-mm"


def test_multimodal_prompt_rejects_invalid_sparse_placeholder_mask():
    request = _request(
        {},
        features={
            "mm_hashes": {"audio": ["hash-1"]},
            "mm_placeholders": {
                "audio": [
                    {"offset": 1, "length": 2, "is_embed": [True]},
                ]
            },
            "kwargs_data": None,
        },
    )

    with pytest.raises(ValueError, match="is_embed"):
        _build_engine_generate_prompt(request)


def test_priority_uses_envelope_for_generate_and_routing_for_legacy():
    assert _engine_generate_priority(_request({}, priority=-4)) == -4
    assert _engine_generate_priority({"routing": {"priority": -4}}) == 4


def test_routed_experts_use_vllm_base64_numpy_encoding():
    expected = np.array([[1, 2], [3, 4]], dtype=np.int32)
    encoded = _serialize_routed_experts_vllm(expected)
    assert encoded is not None
    decoded = np.load(io.BytesIO(base64.b64decode(encoded)))
    np.testing.assert_array_equal(decoded, expected)


class _FakeEngineClient:
    tokenizer = None

    def __init__(self, responses):
        self.responses = responses

    def generate(self, *args, **kwargs):
        async def _stream():
            for response in self.responses:
                yield response

        return _stream()


@pytest.mark.asyncio
async def test_engine_generate_worker_emits_native_terminal_metadata():
    routed_experts = np.array([[1, 2]], dtype=np.uint8)
    responses = [
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=[41],
                    finish_reason="length",
                    stop_reason=None,
                    logprobs=None,
                    routed_experts=routed_experts,
                )
            ],
            prompt_token_ids=[11, 22, 33],
            encoder_prompt_token_ids=None,
            prompt_logprobs=None,
            num_cached_tokens=0,
            kv_transfer_params={"connector": "test"},
        )
    ]
    handler = SimpleNamespace(
        engine_client=_FakeEngineClient(responses),
        _extract_logprobs=BaseWorkerHandler._extract_logprobs,
        _log_with_lora_context=lambda *args, **kwargs: None,
    )

    chunks = []
    async for chunk in BaseWorkerHandler.generate_tokens(
        handler,
        prompt={"prompt_token_ids": [11, 22, 33]},
        sampling_params=SamplingParams(max_tokens=1),
        request_id="request-1",
        engine_generate=True,
    ):
        chunks.append(chunk)

    assert chunks[0]["token_ids"] == [41]
    assert chunks[0]["finish_reason"] == "length"
    assert chunks[0]["completion_usage"]["completion_tokens"] == 1
    assert chunks[0]["engine_data"]["kv_transfer_params"] == {"connector": "test"}
    decoded = np.load(
        io.BytesIO(base64.b64decode(chunks[0]["engine_data"]["routed_experts"]))
    )
    np.testing.assert_array_equal(decoded, routed_experts)
