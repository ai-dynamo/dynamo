# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from unittest.mock import Mock

import pytest

from tests.serve.conftest import (
    MULTIMODAL_IMG_URL,
    MULTIMODAL_VIDEO_URL,
    get_multimodal_test_image_bytes,
)
from tests.utils.engine_process import (
    EngineProcess,
    EngineResponseError,
    ResponseValidationError,
)
from tests.utils.multimodal import (
    UuidPassthroughChatPayload,
    make_mixed_image_video_payload,
    make_qwen35_custom_encoder_multi_image_payload,
    make_sidecar_media_payload,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _image_part(body: dict) -> dict:
    return body["messages"][0]["content"][1]


def test_uuid_passthrough_payload_sends_fill_then_uuid_only() -> None:
    payload = UuidPassthroughChatPayload(expected_response=["green"])

    first_body = payload.body_for_iteration(0)
    fill = _image_part(first_body)
    assert payload.body_for_iteration(0) is first_body
    reuse = _image_part(payload.body_for_iteration(1))

    assert fill == {
        "type": "image_url",
        "image_url": {"url": MULTIMODAL_IMG_URL},
        "uuid": "dynamo-mm-cache-image-1",
    }
    assert reuse == {
        "type": "image_url",
        "image_url": None,
        "uuid": "dynamo-mm-cache-image-1",
    }
    assert payload.expected_log == []
    payload.final_validation()


def test_uuid_embedding_cache_payload_checks_hit_after_gpu_eviction() -> None:
    payload = UuidPassthroughChatPayload(
        expected_response=["green"],
        exercise_embedding_cache=True,
    )

    first_fill = _image_part(payload.body_for_iteration(0))
    assert first_fill["uuid"] == "dynamo-mm-cache-image-1"
    assert first_fill["image_url"] == {"url": MULTIMODAL_IMG_URL}
    assert payload.expected_log == []

    eviction_fill = _image_part(payload.body_for_iteration(1))
    assert eviction_fill["uuid"] == "dynamo-mm-cache-image-1-eviction"
    assert eviction_fill["image_url"] == {"url": MULTIMODAL_IMG_URL}
    assert payload.expected_log == []

    reuse = _image_part(payload.body_for_iteration(2))
    assert reuse["uuid"] == "dynamo-mm-cache-image-1"
    assert reuse["image_url"] is None
    assert payload.expected_log == [
        "Dynamo multimodal embedding cache hit: "
        r"identifier='dynamo\-mm\-cache\-image\-1'"
    ]
    payload.final_validation()


def test_qwen35_multi_image_payload_is_order_sensitive() -> None:
    payload = make_qwen35_custom_encoder_multi_image_payload()
    content = payload.body["messages"][0]["content"]

    assert payload.expected_response == ["green then red"]
    assert "green" not in content[0]["text"].lower()
    assert "red" not in content[0]["text"].lower()
    assert [part["image_url"]["url"] for part in content[1:]] == [
        "data:image/png;base64,"
        + base64.b64encode(get_multimodal_test_image_bytes(color)).decode()
        for color in ("green", "red")
    ]


def test_mixed_image_video_payload_requires_video_context() -> None:
    payload = make_mixed_image_video_payload(["triangle"], frontend_decoding=True)
    content = payload.body["messages"][0]["content"]

    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": MULTIMODAL_IMG_URL},
    }
    assert content[2] == {
        "type": "video_url",
        "video_url": {"url": MULTIMODAL_VIDEO_URL},
    }
    assert payload.expected_response == ["triangle"]


def _sidecar_image_response(content: str, cached: int, hit_rate: float) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 256,
            "completion_tokens": 1,
            "total_tokens": 257,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
        "nvext": {"prompt_token_ids": [1, 2], "timing": {"kv_hit_rate": hit_rate}},
    }
    return response


@pytest.mark.parametrize(
    "retry_iteration,failed_content", [(0, "green"), (0, ""), (2, "green")]
)
def test_image_content_retry_preserves_first_cache_observation(
    monkeypatch, retry_iteration, failed_content
) -> None:
    payload = make_sidecar_media_payload("image-routing")
    assert payload.body["seed"] == 0
    assert payload.max_attempts == 3
    monkeypatch.setattr(payload, "_wait_for_complete_prompt", Mock())
    process = object.__new__(EngineProcess)
    for iteration, (cached, hit_rate) in enumerate(
        [(0, 0.0), (256, 1.0), (64, 0.25), (64, 0.25)]
    ):
        payload.body_for_iteration(iteration)
        content = "blue red" if iteration == 3 else "red blue"
        if iteration == retry_iteration:
            with pytest.raises(ResponseValidationError):
                process.check_response(
                    payload, _sidecar_image_response(failed_content, cached, hit_rate)
                )
            process.check_response(payload, _sidecar_image_response(content, 256, 1.0))
        else:
            process.check_response(
                payload, _sidecar_image_response(content, cached, hit_rate)
            )
    payload.final_validation()
    assert payload._observed == [
        (256, 0, 0.0),
        (256, 256, 1.0),
        (256, 64, 0.25),
        (256, 64, 0.25),
    ]


@pytest.mark.parametrize("invalid", ["usage", "cache", "probe_worker"])
def test_sidecar_usage_and_cache_failures_are_not_content_retries(
    invalid, monkeypatch
) -> None:
    payload = make_sidecar_media_payload("image-routing")
    payload.body_for_iteration(0)
    process = object.__new__(EngineProcess)
    process.check_response(payload, _sidecar_image_response("red blue", 0, 0.0))
    payload.body_for_iteration(1)
    response = _sidecar_image_response("green", 0, 0.0)
    if invalid == "usage":
        response.json.return_value["usage"]["total_tokens"] = 1
    elif invalid == "probe_worker":
        process.check_response(payload, _sidecar_image_response("red blue", 256, 1.0))
        payload.body_for_iteration(2)
        response = _sidecar_image_response("green", 64, 0.25)
        monkeypatch.setattr(
            "tests.utils.payloads.requests.post", Mock(return_value=response)
        )
    with pytest.raises(EngineResponseError) as failure:
        process.check_response(payload, response)
    assert not isinstance(failure.value, ResponseValidationError)
    assert payload._observed[0] == (256, 0, 0.0)
    assert len(payload._observed) == (3 if invalid == "probe_worker" else 1)
