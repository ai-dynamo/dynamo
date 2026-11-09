# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import TypeAdapter
from vllm.sampling_params import SamplingParams

from dynamo.vllm.multimodal_utils.protocol import vLLMMultimodalRequest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    "max_tokens": 8,
    "seed": 17,
    "stop": ["END"],
    "stop_token_ids": [42],
}


def test_vllm_sampling_params_json_after_dynamo_import():
    # Build a fresh schema after the Dynamo protocol module has been imported.
    adapter = TypeAdapter(SamplingParams)
    encoded = adapter.dump_json(SamplingParams(**PARAMS))
    restored = adapter.validate_json(encoded)

    assert isinstance(restored, SamplingParams)
    assert restored == SamplingParams(**PARAMS)


@pytest.mark.parametrize("input_form", ["object", "dict", "legacy_json"])
def test_dynamo_multimodal_sampling_params_json_roundtrip(input_form):
    raw = {
        "object": SamplingParams(**PARAMS),
        "dict": PARAMS,
        "legacy_json": json.dumps(PARAMS),
    }[input_form]
    request = vLLMMultimodalRequest(
        engine_prompt={"prompt_token_ids": [1, 2, 3]},
        sampling_params=raw,
        request_id="sampling-params",
    )
    payload = request.model_dump_json()
    serialized_params = json.loads(payload)["sampling_params"]
    assert {key: serialized_params[key] for key in PARAMS} == PARAMS

    restored = vLLMMultimodalRequest.model_validate_json(payload)
    assert restored.sampling_params == SamplingParams(**PARAMS)
    assert restored.engine_prompt == request.engine_prompt
