# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.client import _sanitize_payload_for_logging

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def test_sanitize_payload_preserves_uuid_only_image_part() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": "cached-image",
                    }
                ],
            }
        ]
    }

    assert _sanitize_payload_for_logging(payload) == payload
