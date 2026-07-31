# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from unittest.mock import MagicMock

import pytest

from tests.utils.payload_builder import classify_payload, pooling_payload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _response(
    *,
    json_body: dict | None = None,
    content: bytes = b"",
    headers: dict[str, str] | None = None,
) -> MagicMock:
    response = MagicMock()
    response.json.return_value = json_body
    response.content = content
    response.headers = headers or {}
    return response


def test_classify_payload_validates_truncated_usage():
    payload = classify_payload(
        input_data=[0, 1, 2, 3],
        expected_prompt_tokens=2,
        extra_body={"truncate_prompt_tokens": 2},
    )
    response = _response(
        json_body={
            "object": "list",
            "data": [
                {
                    "index": 0,
                    "label": "entailment",
                    "probs": [0.1, 0.8, 0.1],
                    "num_classes": 3,
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "total_tokens": 2,
                "completion_tokens": 0,
            },
        }
    )

    assert payload.process_response(response) == "Classified 1 inputs"


def test_pooling_payload_validates_nested_float_output():
    payload = pooling_payload(
        input_data=[[0, 1, 2], [3, 4, 5]],
        expected_prompt_tokens=4,
        extra_body={"truncate_prompt_tokens": 2},
    )
    response = _response(
        json_body={
            "object": "list",
            "data": [
                {"index": 0, "object": "pooling", "data": [[0.1], [0.2]]},
                {"index": 1, "object": "pooling", "data": [[0.3], [0.4]]},
            ],
            "usage": {
                "prompt_tokens": 4,
                "total_tokens": 4,
                "completion_tokens": 0,
            },
        }
    )

    assert payload.process_response(response) == "Pooled 2 inputs as float"


def test_pooling_payload_validates_base64_dtype_width():
    payload = pooling_payload(
        input_data="text",
        extra_body={"encoding_format": "base64", "embed_dtype": "float16"},
    )
    response = _response(
        json_body={
            "object": "list",
            "data": [
                {
                    "index": 0,
                    "object": "pooling",
                    "data": base64.b64encode(b"\x00\x01\x02\x03").decode(),
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "total_tokens": 1,
                "completion_tokens": 0,
            },
        }
    )

    assert payload.process_response(response) == "Pooled 1 inputs as base64"


def test_pooling_payload_validates_bytes_metadata():
    payload = pooling_payload(
        input_data="text",
        expected_response=["Pooled 1 binary tensors"],
        expected_prompt_tokens=3,
        extra_body={
            "encoding_format": "bytes",
            "embed_dtype": "float16",
            "endianness": "big",
        },
    )
    metadata = {
        "data": [
            {
                "index": 0,
                "embed_dtype": "float16",
                "endianness": "big",
                "start": 0,
                "end": 6,
                "shape": [3],
            }
        ],
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
    response = _response(
        content=b"\x00\x01\x02\x03\x04\x05",
        headers={
            "content-type": "application/octet-stream",
            "metadata": json.dumps(metadata),
        },
    )

    assert payload.process_response(response) == "Pooled 1 binary tensors"


def test_pooling_payload_validates_bytes_only_without_metadata():
    payload = pooling_payload(
        input_data="text",
        expected_response=["Pooled bytes_only response"],
        extra_body={"encoding_format": "bytes_only"},
    )
    response = _response(
        content=b"\x00\x01\x02\x03",
        headers={"content-type": "application/octet-stream"},
    )

    assert payload.process_response(response) == (
        "Pooled bytes_only response with 4 bytes"
    )
