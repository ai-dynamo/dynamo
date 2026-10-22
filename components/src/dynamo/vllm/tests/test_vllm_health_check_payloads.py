# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vllm health-check payload shape tests.

Asserts the canary HEALTH_CHECK_KEY marker is layered onto each Vllm probe
payload via the to_dict() override and survives DYN_HEALTH_CHECK_PAYLOAD env
overrides. No vllm handler branches on the marker today; this is wire-format
parity with trtllm/sglang for any future marker-gated behavior.
"""

import json
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.vllm.health_check import (
    VllmEmbeddingHealthCheckPayload,
    VllmHealthCheckPayload,
    VllmOmniHealthCheckPayload,
    VllmPrefillHealthCheckPayload,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

PAYLOAD_FACTORIES = [
    pytest.param(VllmHealthCheckPayload, id="VllmHealthCheckPayload"),
    pytest.param(VllmPrefillHealthCheckPayload, id="VllmPrefillHealthCheckPayload"),
    pytest.param(VllmOmniHealthCheckPayload, id="VllmOmniHealthCheckPayload"),
    pytest.param(
        VllmEmbeddingHealthCheckPayload,
        id="VllmEmbeddingHealthCheckPayload",
    ),
]


@pytest.mark.parametrize("make", PAYLOAD_FACTORIES)
def test_payload_has_marker(make):
    assert make().to_dict().get(HEALTH_CHECK_KEY) is True


@pytest.mark.parametrize("make", PAYLOAD_FACTORIES)
def test_env_override_preserves_marker(monkeypatch, make):
    """DYN_HEALTH_CHECK_PAYLOAD must not drop the canary marker."""
    monkeypatch.setenv(
        "DYN_HEALTH_CHECK_PAYLOAD",
        json.dumps(
            {
                "token_ids": [1],
                "sampling_options": {"temperature": 0.0},
                "stop_conditions": {"max_tokens": 1},
            }
        ),
    )
    assert make().to_dict().get(HEALTH_CHECK_KEY) is True


def test_embedding_payload_shape_matches_handler_contract():
    """``VllmEmbeddingHealthCheckPayload`` must produce the
    ``{model, input}`` shape that ``EmbeddingWorkerHandler.generate``
    expects. The chat payload (``token_ids`` + ``sampling_options``)
    would be rejected by that handler with "missing required 'input'
    field" and leave the canary stuck unhealthy forever -- regression
    pin for the bug PR #9765's canary readiness fix targets."""
    payload = VllmEmbeddingHealthCheckPayload(
        model_name="Qwen/Qwen3-Embedding-0.6B"
    ).to_dict()
    assert payload.get("model") == "Qwen/Qwen3-Embedding-0.6B"
    assert payload.get("input") == "probe"
    assert payload.get(HEALTH_CHECK_KEY) is True
    # Must NOT carry chat-shaped keys -- those would make the embedding
    # handler's request shape validation fail.
    assert "token_ids" not in payload
    assert "sampling_options" not in payload
    assert "stop_conditions" not in payload


def test_embedding_payload_omits_model_when_no_name():
    """``model_name`` is optional. When omitted, the ``model`` key is
    absent from the payload and the handler falls back to
    ``config.served_model_name`` (see ``EmbeddingWorkerHandler.generate``).
    """
    payload = VllmEmbeddingHealthCheckPayload().to_dict()
    assert "model" not in payload
    assert payload.get("input") == "probe"
    assert payload.get(HEALTH_CHECK_KEY) is True


@pytest.mark.parametrize(
    "make", [VllmHealthCheckPayload, VllmPrefillHealthCheckPayload]
)
@pytest.mark.parametrize("use_text_input", [False, True])
@pytest.mark.parametrize("is_diffusion", [False, True])
def test_generation_canary_uses_effective_model_temperature(
    make, use_text_input, is_diffusion
):
    engine = SimpleNamespace(model_config=SimpleNamespace(is_diffusion=is_diffusion))
    payload = make(engine, use_text_input=use_text_input).to_dict()
    sampling = payload if use_text_input else payload["sampling_options"]
    assert sampling["temperature"] == (1.0 if is_diffusion else 0.0)
    assert payload[HEALTH_CHECK_KEY] is True
    if use_text_input:
        assert payload["prompt"] == "Test"
        assert payload["max_tokens"] == 1
    else:
        assert payload["token_ids"] == [1]
        assert payload["stop_conditions"]["max_tokens"] == 1


@pytest.mark.parametrize("use_text_input", [False, True])
def test_diffusion_canary_preserves_explicit_override(monkeypatch, use_text_input):
    override = (
        {"prompt": "override", "temperature": 0.0, "max_tokens": 2}
        if use_text_input
        else {
            "token_ids": [42],
            "sampling_options": {"temperature": 0.0},
            "stop_conditions": {"max_tokens": 2},
        }
    )
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", json.dumps(override))
    engine = SimpleNamespace(model_config=SimpleNamespace(is_diffusion=True))
    payload = VllmHealthCheckPayload(engine, use_text_input=use_text_input).to_dict()
    assert payload == {**override, HEALTH_CHECK_KEY: True}


@pytest.mark.parametrize("use_text_input", [False, True])
def test_installed_main_accepts_default_diffusion_canary_sampling(use_text_input):
    try:
        installed = version("vllm")
    except PackageNotFoundError:
        pytest.skip("Optional vLLM runtime is not installed")
    if installed != "0.32.1rc1.dev172+g9da5f2d31":
        pytest.skip("Exact pinned-main native validation API is not installed")
    from vllm.sampling_params import SamplingParams

    model = SimpleNamespace(is_diffusion=True)
    payload = VllmHealthCheckPayload(
        SimpleNamespace(model_config=model), use_text_input=use_text_input
    ).to_dict()
    sampling = (
        {"temperature": payload["temperature"], "max_tokens": payload["max_tokens"]}
        if use_text_input
        else {**payload["sampling_options"], **payload["stop_conditions"]}
    )
    params = SamplingParams(**sampling)
    params._validate_diffusion(model)
