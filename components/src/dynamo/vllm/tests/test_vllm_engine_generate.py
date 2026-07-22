# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from dynamo.vllm.engine_generate import (
    build_prompt,
    build_sampling_params,
    merge_kv_transfer_params,
    serialize_routed_experts,
)
from dynamo.vllm.handlers import (
    BaseWorkerHandler,
    PrefillWorkerHandler,
    _use_prefill_prompt_logprobs,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _request(sampling_params=None, **top_level):
    return {
        "model": "test-model",
        "token_ids": [11, 22, 33],
        "extra_args": {
            "vllm_tito": {
                "request_id": "request-1",
                "model": "test-model",
                "sampling_params": sampling_params or {},
                **top_level,
            }
        },
        "routing": {"priority": 4},
    }


def test_text_prompt_and_sampling_preserve_native_generate_contract():
    request = _request(
        {"temperature": 0, "max_tokens": 17, "logprobs": 2},
        cache_salt="checkpoint-7",
    )

    prompt = build_prompt(request)
    params = build_sampling_params(
        request, default_sampling_params={}, model_max_len=64
    )

    assert prompt["prompt_token_ids"] == [11, 22, 33]
    assert prompt["cache_salt"] == "dynamo-cache-salt:checkpoint-7"
    assert params.temperature == 0
    assert params.max_tokens == 17
    assert params.logprobs == 2
    assert params.output_kind is RequestOutputKind.DELTA


def test_multimodal_prompt_reconstructs_hashes_placeholders_and_embed_mask():
    request = _request(
        features={
            "mm_hashes": {"image": ["opaque-renderer-image-0"]},
            "mm_placeholders": {
                "image": [
                    {
                        "offset": 0,
                        "length": 3,
                        "is_embed": [False, True, False],
                    }
                ]
            },
            "kwargs_data": None,
        }
    )

    prompt = build_prompt(request)

    assert prompt["type"] == "multimodal"
    assert prompt["prompt_token_ids"] == [11, 22, 33]
    assert prompt["mm_hashes"] == {"image": ["opaque-renderer-image-0"]}
    placeholder = prompt["mm_placeholders"]["image"][0]
    assert (placeholder.offset, placeholder.length) == (0, 3)
    assert placeholder.is_embed.tolist() == [False, True, False]


def test_multimodal_prompt_rejects_mismatched_embed_mask():
    request = _request(
        features={
            "mm_hashes": {"image": ["opaque-renderer-image-0"]},
            "mm_placeholders": {
                "image": [{"offset": 0, "length": 3, "is_embed": [True]}]
            },
            "kwargs_data": None,
        }
    )

    with pytest.raises(ValueError, match="one-dimensional and match length"):
        build_prompt(request)


def test_native_generate_rejects_conflicting_canonical_state():
    request = _request({})
    request["extra_args"]["vllm_tito"]["token_ids"] = [11, 22, 33]
    request["token_ids"] = [11, 22]
    with pytest.raises(ValueError, match="do not match"):
        build_prompt(request)

    with pytest.raises(ValueError, match="collide"):
        merge_kv_transfer_params({"same": 1}, {"same": 2})


def test_legacy_pd_kv_transfer_keeps_framework_state():
    assert merge_kv_transfer_params(
        {"remote_host": "caller"},
        {"remote_host": "prefill", "remote_port": 1234},
        reject_collisions=False,
    ) == {"remote_host": "prefill", "remote_port": 1234}


def test_routed_experts_use_vllm_base64_numpy_encoding():
    expected = np.array([[1, 2], [3, 4]], dtype=np.int32)

    encoded = serialize_routed_experts(expected)

    assert encoded is not None
    decoded = np.load(io.BytesIO(base64.b64decode(encoded)))
    np.testing.assert_array_equal(decoded, expected)


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
