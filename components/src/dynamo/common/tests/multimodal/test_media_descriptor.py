# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.multimodal.media_descriptor import decoded_content_hash_key

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_accepts_canonical_decoded_content_hash() -> None:
    assert (
        decoded_content_hash_key({"content_hash": "0123456789abcdef"})
        == "0123456789abcdef"
    )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "0123456789abcdef",
        {},
        {"content_hash": "0x12345678901234"},
        {"content_hash": "+123456789abcdef"},
        {"content_hash": "01_23456789abcde"},
        {"content_hash": "0123456789abcdeF"},
        {"content_hash": ""},
        {"content_hash": None},
    ],
)
def test_rejects_noncanonical_decoded_content_hash(metadata: object) -> None:
    assert decoded_content_hash_key(metadata) is None
