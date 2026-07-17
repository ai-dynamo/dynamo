# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

from tests.serve.conftest import MULTIMODAL_IMG_URL
from tests.utils.multimodal import UuidPassthroughChatPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _image_part(body: dict) -> dict:
    return body["messages"][0]["content"][1]


def test_uuid_passthrough_payload_sends_fill_then_uuid_only() -> None:
    payload = UuidPassthroughChatPayload(expected_response=["green"])

    fill = _image_part(payload.body)
    reuse = _image_part(payload.body)

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

    first_fill = _image_part(payload.body)
    assert first_fill["uuid"] == "dynamo-mm-cache-image-1"
    assert first_fill["image_url"] == {"url": MULTIMODAL_IMG_URL}
    assert payload.expected_log == []

    eviction_fill = _image_part(payload.body)
    assert eviction_fill["uuid"] == "dynamo-mm-cache-image-1-eviction"
    assert eviction_fill["image_url"] == {"url": MULTIMODAL_IMG_URL}
    assert payload.expected_log == []

    reuse = _image_part(payload.body)
    assert reuse["uuid"] == "dynamo-mm-cache-image-1"
    assert reuse["image_url"] is None
    assert payload.expected_log == [
        "Dynamo multimodal embedding cache hit: "
        r"identifier='dynamo\-mm\-cache\-image\-1'"
    ]
    hit_pattern = re.compile(payload.expected_log[0])
    assert hit_pattern.search(
        "Dynamo multimodal embedding cache hit: " "identifier='dynamo-mm-cache-image-1'"
    )
    assert not hit_pattern.search(
        "Dynamo multimodal embedding cache hit: "
        "identifier='dynamo-mm-cache-image-1-eviction'"
    )
    payload.final_validation()
