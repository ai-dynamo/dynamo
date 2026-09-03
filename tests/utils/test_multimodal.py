# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest

from tests.serve.conftest import MULTIMODAL_IMG_URL, get_multimodal_test_image_bytes
from tests.utils.multimodal import (
    UuidPassthroughChatPayload,
    make_qwen35_custom_encoder_multi_image_payload,
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
