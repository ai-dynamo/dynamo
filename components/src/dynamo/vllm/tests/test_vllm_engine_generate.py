# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.sampling_params import RequestOutputKind

from dynamo.vllm.engine_generate import (
    build_prompt,
    build_sampling_params,
    merge_kv_transfer_params,
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
