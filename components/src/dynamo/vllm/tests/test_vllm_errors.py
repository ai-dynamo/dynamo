# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.v1.engine.exceptions import EngineGenerateError

from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.errors import uuid_image_cache_miss_error

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_direct_image_cache_miss_identifies_the_aligned_uuid_only_slot():
    request = {
        "multi_modal_data": {
            "image_url": [
                {"Url": "https://example.com/image?token=SECRET"},
                {"UuidOnly": "SECRET-UUID"},
            ]
        },
        "multi_modal_uuids": {"image_url": ["other-image", "SECRET-UUID"]},
    }
    error = uuid_image_cache_miss_error(
        ValueError("Cache miss for image at index 1 but data is not provided."), request
    )
    assert isinstance(error, InvalidArgument)
    assert "Resend the request" in str(error)
    assert "same UUID" in str(error)
    assert "SECRET" not in str(error)


@pytest.mark.parametrize(
    "message,slot,uuid",
    [
        ("Invalid sampling parameters", {"UuidOnly": "image-key"}, "image-key"),
        (
            "Cache miss for image at index 0 but data is not provided.",
            {"Url": "https://example.com/image"},
            "image-key",
        ),
        (
            "Cache miss for image at index 1 but data is not provided.",
            {"UuidOnly": "image-key"},
            "image-key",
        ),
        (
            "Cache miss for image at index 0 but data is not provided.",
            {"UuidOnly": "image-key"},
            "different-key",
        ),
    ],
    ids=["unrelated-validation", "url-backed", "wrong-index", "different-uuid"],
)
def test_unrelated_errors_are_not_reported_as_uuid_cache_misses(message, slot, uuid):
    request = {
        "multi_modal_data": {"image_url": [slot]},
        "multi_modal_uuids": {"image_url": [uuid]},
    }
    cause = ValueError(message)
    wrapped = EngineGenerateError()
    wrapped.__cause__ = cause
    for error in (cause, wrapped):
        assert uuid_image_cache_miss_error(error, request) is None


def test_wrapped_server_failure_is_not_reported_as_a_cache_miss():
    request = {
        "multi_modal_data": {"image_url": [{"UuidOnly": "image-key"}]},
        "multi_modal_uuids": {"image_url": ["image-key"]},
    }
    error = EngineGenerateError()
    error.__cause__ = RuntimeError(
        "Cache miss for image at index 0 but data is not provided."
    )
    assert uuid_image_cache_miss_error(error, request) is None
