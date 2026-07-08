# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.trtllm.utils.request_utils import (
    request_cache_salt,
    stored_event_cache_salt,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    ("request_body", "expected"),
    [
        (
            {
                "routing": {"cache_salt": "tenant-routing"},
                "extra_args": {"nvext": {"cache_salt": "tenant-nvext"}},
            },
            "tenant-routing",
        ),
        (
            {
                "routing": {"cache_salt": ""},
                "extra_args": {"nvext": {"cache_salt": "tenant-nvext"}},
            },
            "tenant-nvext",
        ),
        (
            {
                "routing": {"cache_salt": ""},
                "extra_args": {"nvext": {"cache_salt": ""}},
            },
            None,
        ),
        ({}, None),
    ],
)
def test_request_cache_salt_precedence_and_empty_fallback(
    request_body, expected
) -> None:
    assert request_cache_salt(request_body) == expected


def test_stored_event_cache_salt_uses_per_block_schema() -> None:
    data = {
        "blocks": [
            {"cache_salt": "tenant-a"},
            {"cache_salt": "tenant-a"},
        ]
    }

    assert stored_event_cache_salt(data) == "tenant-a"


def test_stored_event_cache_salt_supports_parent_fallback() -> None:
    assert stored_event_cache_salt({"cache_salt": "tenant-a", "blocks": []}) == (
        "tenant-a"
    )


def test_stored_event_cache_salt_allows_unsalted_blocks() -> None:
    assert stored_event_cache_salt({"blocks": [{}, {}]}) is None


def test_stored_event_cache_salt_rejects_conflicting_blocks() -> None:
    data = {
        "blocks": [
            {"cache_salt": "tenant-a"},
            {"cache_salt": "tenant-b"},
        ]
    }

    with pytest.raises(ValueError, match="conflicting cache_salt"):
        stored_event_cache_salt(data)
