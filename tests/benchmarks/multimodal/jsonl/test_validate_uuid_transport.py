# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.multimodal.jsonl.validate_uuid_transport import count_uuid_image_parts

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_count_uuid_image_parts() -> None:
    payload = {
        "data": [
            {
                "payloads": [
                    {
                        "messages": [
                            {
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,abc"
                                        },
                                        "uuid": "uuid-a",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": ""},
                                        "uuid": "uuid-a",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "missing.png"},
                                    },
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    assert count_uuid_image_parts(payload) == {
        "content": 2,
        "stripped": 1,
        "missing_uuid": 1,
    }
