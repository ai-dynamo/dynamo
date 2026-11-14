# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vllm health-check payload shape tests.

Asserts the canary HEALTH_CHECK_KEY marker is layered onto each Vllm probe
payload via the to_dict() override and survives DYN_HEALTH_CHECK_PAYLOAD env
overrides. No vllm handler branches on the marker today; this is wire-format
parity with trtllm/sglang for any future marker-gated behavior.
"""

import base64
import io
import json
import wave
from types import SimpleNamespace

import pytest

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.vllm.handlers import build_sampling_params, build_sampling_params_openai
from dynamo.vllm.health_check import (
    VllmEmbeddingHealthCheckPayload,
    VllmHealthCheckPayload,
    VllmOmniHealthCheckPayload,
    VllmPrefillHealthCheckPayload,
    VllmTranscriptionHealthCheckPayload,
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
        VllmTranscriptionHealthCheckPayload,
        id="VllmTranscriptionHealthCheckPayload",
    ),
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


def test_transcription_payload_contains_valid_short_wav():
    payload = VllmTranscriptionHealthCheckPayload(model_name="whisper").to_dict()

    assert payload.get("model") == "whisper"
    assert payload.get("language") == "en"
    assert payload.get("response_format") == "json"
    assert payload.get(HEALTH_CHECK_KEY) is True

    audio = base64.b64decode(payload["audio_b64"], validate=True)
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 1_600


GENERATE_PAYLOAD_CLASSES = [
    pytest.param(VllmHealthCheckPayload, id="aggregated"),
    pytest.param(VllmPrefillHealthCheckPayload, id="prefill"),
]


def _engine(return_sampling_mask: bool) -> SimpleNamespace:
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(return_sampling_mask=return_sampling_mask)
        ),
        tokenizer=None,
    )


def _sampling_values(payload: dict, use_text_input: bool) -> dict:
    return payload if use_text_input else payload["sampling_options"]


@pytest.mark.parametrize("use_text_input", [False, True], ids=["tokens", "text"])
@pytest.mark.parametrize("cls", GENERATE_PAYLOAD_CLASSES)
def test_mask_mode_canary_is_sampled(cls, use_text_input):
    """With --return-sampling-mask, vLLM rejects requests unless temperature > 0
    and top_k > 0; a greedy canary keeps /live at 503 forever."""
    payload = cls(_engine(True), use_text_input=use_text_input).to_dict()
    values = _sampling_values(payload, use_text_input)
    assert values["temperature"] > 0
    assert values["top_k"] > 0


@pytest.mark.parametrize("cls", GENERATE_PAYLOAD_CLASSES)
def test_mask_mode_canary_passes_vllm_sampling_checks(cls):
    """The canary values must survive the request path into vLLM's
    SamplingParams, where input_processor enforces the mask-mode checks."""
    token_payload = cls(_engine(True), use_text_input=False).to_dict()
    params = build_sampling_params(token_payload, {})
    assert params.temperature > 0
    assert params.top_k > 0

    text_payload = cls(_engine(True), use_text_input=True).to_dict()
    params = build_sampling_params_openai(text_payload, {})
    assert params.temperature > 0
    assert params.top_k > 0
