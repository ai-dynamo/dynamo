# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import (
    RepetitionDetectionParams,
    RequestOutputKind,
    StructuredOutputsParams,
)

from dynamo.vllm.engine_generate import (
    EngineGenerateRequest,
    build_prompt,
    merge_kv_transfer_params,
    priority,
)
from dynamo.vllm.handlers import BaseWorkerHandler, build_sampling_params

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKER_FIXTURE = (
    REPO_ROOT / "lib/llm/tests/data/inference_generate/vllm-tito-worker.json"
)


def _request(sampling_params=None, **top_level):
    sampling_params = sampling_params or {}
    payload = {
        "request_id": "request-1",
        "model": "test-model",
        "token_ids": [11, 22, 33],
        "sampling_params": sampling_params,
        "priority": 0,
        **top_level,
    }
    return {
        "model": "test-model",
        "token_ids": [11, 22, 33],
        "extra_args": {"vllm_tito": payload},
        "routing": {},
    }


def test_worker_fixture_is_consumed_by_python_adapter():
    request = json.loads(WORKER_FIXTURE.read_text())
    adapter = EngineGenerateRequest.from_request(request)

    assert adapter is not None
    assert adapter.priority(request["routing"]) == 7
    assert adapter.build_sampling_params({}, model_max_len=64).temperature == 0.25
    prompt = adapter.build_prompt()
    assert prompt["prompt_token_ids"] == [11, 22, 33]
    assert prompt["mm_hashes"] == {"image": ["hash-1"]}


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


def test_omitted_max_tokens_uses_server_default_with_context_cap():
    request = _request({"temperature": 0.5})
    params = build_sampling_params(
        request,
        default_sampling_params={"max_tokens": 100},
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
    prompt = build_prompt(_request({}, cache_salt="checkpoint-7"))
    assert prompt["prompt_token_ids"] == [11, 22, 33]
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


def test_priority_is_native_for_engine_generate_and_legacy_for_chat():
    assert priority(_request({}, priority=7), {"priority": -7}) == 7
    assert priority(_request({}, priority=-4), {"priority": 4}) == -4
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
    assert adapter.priority({"priority": 0}) == 0


def test_request_adapter_ignores_legacy_requests():
    assert EngineGenerateRequest.from_request({"token_ids": [11, 22, 33]}) is None


def test_request_adapter_rejects_parallel_generate_request_envelope():
    request = _request({})
    request["generate_request"] = request["extra_args"]["vllm_tito"]
    request["extra_args"] = {}

    assert EngineGenerateRequest.from_request(request) is None


def test_request_adapter_emits_generate_response_metadata():
    adapter = EngineGenerateRequest.from_request(_request({}))
    assert adapter is not None
    routed_experts = np.array([[1, 2], [3, 4]], dtype=np.int32)
    output = {"engine_data": {"prompt_logprobs": [{"11": {"logprob": -0.1}}]}}

    adapter.adapt_response_metadata(
        output,
        routed_experts,
        {"remote_block_ids": [1, 2]},
    )

    engine_data = output["engine_data"]
    decoded = np.load(io.BytesIO(base64.b64decode(engine_data["routed_experts"])))
    np.testing.assert_array_equal(decoded, routed_experts)
    assert engine_data["kv_transfer_params"] == {"remote_block_ids": [1, 2]}
    assert "prompt_logprobs" in engine_data


def test_worker_stream_uses_generate_response_metadata():
    routed_experts = np.array([[1, 2]], dtype=np.int32)
    response = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                index=0,
                token_ids=[41],
                finish_reason="stop",
                stop_reason=None,
                logprobs=None,
                routed_experts=routed_experts,
            )
        ],
        prompt_token_ids=[11, 22, 33],
        prompt_logprobs=None,
        num_cached_tokens=0,
        kv_transfer_params={"remote_block_ids": [1, 2]},
    )

    class FakeEngineClient:
        tokenizer = None

        def generate(self, *args, **kwargs):
            async def stream():
                yield response

            return stream()

    handler = SimpleNamespace(
        engine_client=FakeEngineClient(),
        _extract_logprobs=BaseWorkerHandler._extract_logprobs,
        _log_with_lora_context=lambda *args, **kwargs: None,
    )
    adapter = EngineGenerateRequest.from_request(_request({}))
    assert adapter is not None

    async def collect():
        return [
            chunk
            async for chunk in BaseWorkerHandler.generate_tokens(
                handler,
                prompt={"prompt_token_ids": [11, 22, 33]},
                sampling_params=adapter.build_sampling_params({}, 64),
                request_id="request-1",
                engine_request=adapter,
            )
        ]

    chunks = asyncio.run(collect())
    assert chunks[0]["engine_data"]["kv_transfer_params"] == {
        "remote_block_ids": [1, 2]
    }
    assert isinstance(chunks[0]["engine_data"]["routed_experts"], str)
