# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import numpy as np
import pytest
from vllm.sampling_params import (
    RepetitionDetectionParams,
    RequestOutputKind,
    StructuredOutputsParams,
)

from dynamo.vllm.handlers import (
    _build_engine_generate_prompt,
    _engine_generate_priority,
    _merge_kv_transfer_params,
    _serialize_routed_experts_vllm,
    build_sampling_params,
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
    assert _merge_kv_transfer_params(
        {"caller": 1}, {"framework": 2}
    ) == {"caller": 1, "framework": 2}
    with pytest.raises(ValueError, match="collide"):
        _merge_kv_transfer_params({"same": 1}, {"same": 2})


def test_text_prompt_preserves_exact_tokens_and_cache_salt():
    prompt = _build_engine_generate_prompt(_request({}, cache_salt="checkpoint-7"))
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
    prompt = _build_engine_generate_prompt(request)
    assert prompt["type"] == "multimodal"
    assert prompt["prompt_token_ids"] == [11, 22, 33]
    assert prompt["mm_hashes"] == {"image": ["hash-1"]}
    assert prompt["mm_placeholders"]["image"][0].offset == 1
    assert prompt["mm_kwargs"]["image"] == [None]
    assert prompt["cache_salt"] == "checkpoint-mm"


def test_priority_is_native_for_engine_generate_and_legacy_for_chat():
    assert _engine_generate_priority(_request({}), {"priority": -4}) == -4
    assert _engine_generate_priority({}, {"priority": -4}) == 4


def test_routed_experts_have_one_vllm_compatible_base64_numpy_encoding():
    expected = np.array([[1, 2], [3, 4]], dtype=np.int32)
    encoded = _serialize_routed_experts_vllm(expected)
    assert encoded is not None
    decoded = np.load(io.BytesIO(base64.b64decode(encoded)))
    np.testing.assert_array_equal(decoded, expected)
