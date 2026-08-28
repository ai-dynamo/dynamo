# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

from dynamo.common.external_encoder import (
    ExternalEncoderResult,
    decode_request_plane_tensor,
    encode_request_plane_tensor,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_request_plane_tensor_round_trip(dtype: torch.dtype) -> None:
    tensor = torch.arange(12, dtype=dtype).reshape(3, 4)

    restored = decode_request_plane_tensor(encode_request_plane_tensor(tensor))

    assert restored.is_contiguous()
    assert restored.data_ptr() != tensor.data_ptr()
    torch.testing.assert_close(restored, tensor)


@pytest.mark.parametrize(
    "payload,match",
    [
        (
            {
                "transport": "request_plane_msgpack",
                "shape": [1, 2],
                "dtype": "float32",
                "data": b"short",
            },
            "byte count",
        ),
        (
            {
                "transport": "request_plane_msgpack",
                "shape": [1, 2],
                "dtype": "float64",
                "data": b"",
            },
            "dtype",
        ),
        (
            {
                "transport": "request_plane_msgpack",
                "shape": [1, 2],
                "dtype": "float32",
                "data": [0, 1],
            },
            "must be bytes",
        ),
    ],
)
def test_request_plane_tensor_rejects_malformed_payload(
    payload: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        decode_request_plane_tensor(payload)


def test_request_plane_tensor_rejects_noncontiguous_input() -> None:
    tensor = torch.ones((2, 3), dtype=torch.float32).transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous"):
        encode_request_plane_tensor(tensor)


def test_external_encoder_result_round_trip() -> None:
    result = ExternalEncoderResult(
        features=encode_request_plane_tensor(torch.ones((3, 4))),
        row_splits=(0, 2, 3),
        image_token_id=99,
    )

    restored = ExternalEncoderResult.from_dict(result.to_dict())

    assert restored.row_splits == (0, 2, 3)
    assert restored.image_token_id == 99
    assert restored.embedding_format == "linear_embeddings"


def test_external_encoder_result_covers_packed_rows() -> None:
    features = encode_request_plane_tensor(torch.ones((3, 4)))

    with pytest.raises(ValueError, match="do not cover"):
        ExternalEncoderResult(
            features=features,
            row_splits=(0, 2),
            image_token_id=99,
        )
