# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.multimodal.cache_uuid import (
    reject_unsupported_json_multimodal_data,
    reject_unsupported_multimodal_uuids,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "multi_modal_uuids",
    [
        None,
        {},
        {"image_url": [None, None]},
        {"image_url": [None], "video_url": [None]},
    ],
)
def test_allows_requests_without_cache_uuids(multi_modal_uuids: object) -> None:
    reject_unsupported_multimodal_uuids(multi_modal_uuids)


def test_rejects_request_with_cache_uuid() -> None:
    with pytest.raises(
        ValueError,
        match="supported only by the vLLM backend",
    ):
        reject_unsupported_multimodal_uuids({"image_url": [None, "cached-image"]})


def test_json_guard_rejects_opaque_payloads_but_allows_existing_media() -> None:
    reject_unsupported_json_multimodal_data({"image_url": [{"Url": "https://image"}]})
    reject_unsupported_json_multimodal_data(None)
    with pytest.raises(ValueError, match="supported only by the vLLM backend"):
        reject_unsupported_json_multimodal_data({"image": [{"Json": None}]})


@pytest.mark.parametrize(
    "multi_modal_uuids",
    [
        [],
        "cached-image",
        {"image_url": None},
        {"image_url": "cached-image"},
        {"image_url": 123},
    ],
)
def test_rejects_malformed_cache_uuid_metadata(multi_modal_uuids: object) -> None:
    with pytest.raises(
        ValueError,
        match="supported only by the vLLM backend",
    ):
        reject_unsupported_multimodal_uuids(multi_modal_uuids)
