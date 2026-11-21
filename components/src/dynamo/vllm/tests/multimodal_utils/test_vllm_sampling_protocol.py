# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

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
    """Preserve vLLM JSON serialization when Dynamo is imported first."""
    adapter = TypeAdapter(SamplingParams)
    encoded = adapter.dump_json(SamplingParams(**PARAMS))
    restored = adapter.validate_json(encoded)

    assert isinstance(restored, SamplingParams)
    assert restored == SamplingParams(**PARAMS)


@pytest.mark.parametrize("input_form", ["object", "dict", "legacy_json"])
def test_dynamo_multimodal_sampling_params_json_roundtrip(input_form):
    """Preserve explicit sampling values through every supported wire input."""
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


@pytest.mark.parametrize("import_order", ["dynamo-first", "vllm-first"])
def test_native_render_routes_preserve_sampling_params(import_order):
    """Exercise both native render routes with isolated module import orders."""
    # Each interpreter builds its own Pydantic schemas, independent of collection.
    probe = """
import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

if sys.argv[1] == "dynamo-first":
    importlib.import_module("dynamo.vllm.multimodal_utils.protocol")

api = importlib.import_module("vllm.entrypoints.scale_out.render.api_router")
serving = importlib.import_module("vllm.entrypoints.scale_out.render.serving")
protocol = importlib.import_module(
    "vllm.entrypoints.scale_out.token_in_token_out.protocol"
)
sampling = importlib.import_module("vllm.sampling_params")

if sys.argv[1] == "vllm-first":
    importlib.import_module("dynamo.vllm.multimodal_utils.protocol")

params = json.loads(sys.argv[2])
tokens = [1, 2, 3]
engine_input = {"type": "token", "prompt_token_ids": tokens}
model_config = SimpleNamespace(
    max_model_len=128,
    is_encoder_decoder=False,
    generation_config="vllm",
    override_generation_config={},
    get_diff_sampling_param=lambda: {},
)
online_renderer = SimpleNamespace(
    model_config=model_config,
    render_chat=AsyncMock(return_value=([], [engine_input])),
    render_completion=AsyncMock(return_value=[engine_input]),
    effective_chat_template_kwargs=lambda request: {},
    parser=None,
    renderer=None,
    chat_template=None,
    use_harmony=False,
)
app = FastAPI()
app.state.serving_render = serving.ServingRender(
    models=SimpleNamespace(is_base_model=lambda name: name == "fixture"),
    online_renderer=online_renderer,
)
app.include_router(api.router)

with TestClient(app) as client:
    for path, content in [
        ("/v1/chat/completions/render",
         {"messages": [{"role": "user", "content": "hello"}]}),
        ("/v1/completions/render", {"prompt": "hello"}),
    ]:
        response = client.post(path, json={"model": "fixture", **content, **params})
        assert response.status_code == 200, response.text
        payload = response.json()
        wire = payload[0] if isinstance(payload, list) else payload
        assert {key: wire["sampling_params"][key] for key in params} == params
        restored = protocol.GenerateRequest.model_validate(wire)
        assert restored.token_ids == tokens
        assert isinstance(restored.sampling_params, sampling.SamplingParams)
        assert {key: getattr(restored.sampling_params, key) for key in params} == params

online_renderer.render_chat.assert_awaited_once()
online_renderer.render_completion.assert_awaited_once()
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, import_order, json.dumps(PARAMS)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
