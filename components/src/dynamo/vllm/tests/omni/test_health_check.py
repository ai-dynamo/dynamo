# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dynamo.common.utils.output_modalities import RequestType
from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.vllm.omni.health_check import (
    build_omni_health_check_payload,
    has_completed_output,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "value", ["{bad", "[]", "null", '"prompt"', "", "@/missing/canary.json"]
)
def test_invalid_override_does_not_downgrade_to_native_health(monkeypatch, value):
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", value)
    with pytest.raises(ValueError, match="refusing to fall back"):
        build_omni_health_check_payload(["image"])


def test_token_payload_is_rejected_for_image_canary(monkeypatch):
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", '{"token_ids":[1]}')
    with pytest.raises(ValueError, match="configured Omni request protocol"):
        build_omni_health_check_payload(["image"])


@pytest.mark.parametrize("modality", ["text", "image"])
@pytest.mark.parametrize(
    "messages",
    [
        [],
        "hello",
        [{}],
        [{"role": "user", "content": 42}],
        [{"role": "assistant", "content": "hello"}],
        [{"role": "user", "content": "   "}],
    ],
)
def test_invalid_chat_canary_is_rejected(monkeypatch, modality, messages):
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", json.dumps({"messages": messages}))
    with pytest.raises(ValueError, match="configured Omni request protocol"):
        build_omni_health_check_payload([modality])


@pytest.mark.parametrize("modality", ["text", "image"])
def test_valid_chat_canary_preserves_request(monkeypatch, modality):
    payload = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "A teapot."},
        ],
        "max_tokens": 1,
        "extra_body": {"height": 256, "width": 256},
    }
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", json.dumps(payload))
    assert build_omni_health_check_payload([modality]) == {
        **payload,
        HEALTH_CHECK_KEY: True,
    }


@pytest.mark.parametrize("modality", ["text", "image"])
@pytest.mark.asyncio
async def test_invalid_chat_canary_fails_before_model_loading(monkeypatch, modality):
    from dynamo.vllm.omni.main import init_omni

    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", '{"messages":[]}')
    runtime = MagicMock()
    config = SimpleNamespace(output_modalities=[modality])
    with patch("dynamo.vllm.omni.OmniHandler") as handler:
        with pytest.raises(ValueError, match="configured Omni request protocol"):
            await init_omni(runtime, config, MagicMock())
        handler.assert_not_called()
    runtime.endpoint.assert_not_called()


@pytest.mark.parametrize("value", [None, "{}"])
def test_native_check_is_selected_only_without_inference_payload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    else:
        monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", value)
    assert build_omni_health_check_payload(["image"]) == {HEALTH_CHECK_KEY: True}


def test_file_override_preserves_model_specific_fields(monkeypatch, tmp_path):
    payload = {
        "input": "Hello.",
        "voice": "Ryan",
        "task_type": "CustomVoice",
        HEALTH_CHECK_KEY: False,
    }
    path = tmp_path / "canary.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", "@" + str(path))
    assert build_omni_health_check_payload(["audio"]) == {
        **payload,
        HEALTH_CHECK_KEY: True,
    }


@pytest.mark.parametrize(
    "request_type,chunk",
    [
        (RequestType.IMAGE_GENERATION, {"created": 1, "data": [{"b64_json": "image"}]}),
        (
            RequestType.IMAGE_GENERATION,
            {"created": 1, "data": [{"url": "file:///image.png"}]},
        ),
        (
            RequestType.AUDIO_GENERATION,
            {"status": "completed", "data": [{"b64_json": "audio"}]},
        ),
        (
            RequestType.VIDEO_GENERATION,
            {"status": "completed", "data": [{"url": "file:///video.mp4"}]},
        ),
        (
            RequestType.CHAT_COMPLETION,
            {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]},
        ),
    ],
)
def test_formatted_terminal_results_are_recognized(request_type, chunk):
    assert has_completed_output(chunk, request_type)


@pytest.mark.parametrize(
    "chunk",
    [{}, {"choices": None}, {"choices": []}],
    ids=["missing", "null", "empty"],
)
def test_chat_without_choices_is_not_complete(chunk):
    assert has_completed_output(chunk, RequestType.CHAT_COMPLETION) is False
