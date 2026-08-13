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


def test_pooling_payload_validates_flat_float_output():
    """A batch of pre-tokenized prompts returns one flat vector per input."""
    payload = pooling_payload(
        input_data=[[0, 1, 2], [3, 4, 5]],
        expected_prompt_tokens=4,
        extra_body={"truncate_prompt_tokens": 2},
    )
    response = _response(
        json_body={
            "object": "list",
            "data": [
                {"index": 0, "object": "pooling", "data": [0.1, 0.2]},
                {"index": 1, "object": "pooling", "data": [0.3, 0.4]},
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


def _token_wise_response(data, prompt_tokens: int = 2):
    return _response(
        json_body={
            "object": "list",
            "data": [{"index": 0, "object": "pooling", "data": data}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
                "completion_tokens": 0,
            },
        }
    )


def _token_wise_payload():
    return pooling_payload(input_data="text", task="token_classify")


def test_pooling_payload_accepts_token_wise_matrix():
    payload = _token_wise_payload()
    response = _token_wise_response([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])

    assert payload.process_response(response) == "Pooled 1 inputs as float"


def test_pooling_payload_rejects_flattened_token_wise_output():
    """A flattened vector loses the token-by-class structure this case exists
    to prove; validating only data[0] used to accept it."""
    payload = _token_wise_payload()
    response = _token_wise_response([0.1, 0.2, 0.7, 0.3, 0.4, 0.3])

    with pytest.raises(AssertionError, match="must be a list of rows"):
        payload.process_response(response)


def test_pooling_payload_rejects_ragged_token_wise_rows():
    payload = _token_wise_payload()
    response = _token_wise_response([[0.1, 0.2, 0.7], [0.3, 0.4]])

    with pytest.raises(AssertionError, match="consistent width"):
        payload.process_response(response)


def test_pooling_payload_rejects_empty_token_wise_rows():
    payload = _token_wise_payload()
    response = _token_wise_response([[], []])

    with pytest.raises(AssertionError, match="non-empty"):
        payload.process_response(response)


def test_pooling_payload_rejects_nonnumeric_token_wise_rows():
    payload = _token_wise_payload()
    response = _token_wise_response([[0.1, 0.2], [0.3, "oops"]])

    with pytest.raises(AssertionError, match="non-numeric"):
        payload.process_response(response)


def _sequence_wise_payload():
    return pooling_payload(input_data="text", task="classify")


def test_pooling_payload_rejects_nested_sequence_wise_output():
    """Sequence classification returns one vector per input, which the worker
    enforces. A matrix used to pass because only data[0] was validated, so a
    rank regression stayed invisible here."""
    payload = _sequence_wise_payload()
    response = _token_wise_response([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])

    with pytest.raises(AssertionError, match="flat numeric vector"):
        payload.process_response(response)


def test_pooling_payload_rejects_nonnumeric_sequence_wise_values():
    """Every element is checked, not only the first."""
    payload = _sequence_wise_payload()
    response = _token_wise_response([0.1, 0.2, "oops"])

    with pytest.raises(AssertionError, match="flat numeric vector"):
        payload.process_response(response)


def _bytes_payload():
    return pooling_payload(
        input_data="text",
        expected_response=["Pooled 1 binary tensors"],
        extra_body={
            "encoding_format": "bytes",
            "embed_dtype": "float16",
            "endianness": "big",
        },
    )


def _bytes_response(entries, content: bytes):
    return _response(
        content=content,
        headers={
            "content-type": "application/octet-stream",
            "metadata": json.dumps(
                {"data": entries, "usage": {"prompt_tokens": 3, "total_tokens": 3}}
            ),
        },
    )


def test_pooling_payload_rejects_shape_span_mismatch():
    """float16 with shape=[999] and a 2-byte span used to pass on bounds alone,
    leaving clients unable to reconstruct the tensor."""
    payload = _bytes_payload()
    response = _bytes_response(
        [
            {
                "index": 0,
                "embed_dtype": "float16",
                "endianness": "big",
                "start": 0,
                "end": 2,
                "shape": [999],
            }
        ],
        content=b"\x00\x01",
    )

    with pytest.raises(AssertionError, match="needs 1998"):
        payload.process_response(response)


def test_pooling_payload_rejects_overlapping_spans():
    payload = _bytes_payload()
    response = _bytes_response(
        [
            {
                "index": 0,
                "embed_dtype": "float16",
                "endianness": "big",
                "start": 0,
                "end": 4,
                "shape": [2],
            },
            {
                "index": 1,
                "embed_dtype": "float16",
                "endianness": "big",
                "start": 2,
                "end": 6,
                "shape": [2],
            },
        ],
        content=b"\x00\x01\x02\x03\x04\x05",
    )

    with pytest.raises(AssertionError, match="contiguous and non-overlapping"):
        payload.process_response(response)


def test_pooling_payload_rejects_incomplete_body_coverage():
    payload = _bytes_payload()
    response = _bytes_response(
        [
            {
                "index": 0,
                "embed_dtype": "float16",
                "endianness": "big",
                "start": 0,
                "end": 4,
                "shape": [2],
            }
        ],
        content=b"\x00\x01\x02\x03\x04\x05",
    )

    with pytest.raises(AssertionError, match="cover 4 bytes but the body is 6"):
        payload.process_response(response)


def test_pooling_payload_rejects_wrong_rank_binary_shape():
    """shape=[2, 2] and shape=[4] span the same bytes, so the byte-count check
    alone cannot tell them apart; sequence classification must stay rank 1."""
    payload = _bytes_payload()
    response = _bytes_response(
        [
            {
                "index": 0,
                "embed_dtype": "float16",
                "endianness": "big",
                "start": 0,
                "end": 8,
                "shape": [2, 2],
            }
        ],
        content=b"\x00\x01\x02\x03\x04\x05\x06\x07",
    )

    with pytest.raises(AssertionError, match="has rank 2"):
        payload.process_response(response)
