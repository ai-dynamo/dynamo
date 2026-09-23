# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Dynamo's external custom-encoder handoff contract."""

from copy import deepcopy
from typing import Any

import pytest
import torch

from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder import (
    ExternalEncoderHandoff,
    VisionEncoderBackend,
)
from dynamo.vllm.multimodal_utils.custom_encoder.handoff import (
    ExternalEncoderResult,
    decode_request_plane_tensor,
    encode_request_plane_tensor,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
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


def test_external_encoder_result_packs_and_restores_artifacts_in_order() -> None:
    artifacts = [
        torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        torch.full((1, 4), 12, dtype=torch.bfloat16),
    ]

    result = ExternalEncoderResult.from_artifacts(artifacts, image_token_id=99)
    restored = result.to_artifacts()

    assert result.row_splits == (0, 2, 3)
    assert len(restored) == 2
    for actual, expected in zip(restored, artifacts):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "artifacts,match",
    [
        (
            [torch.ones((1, 4)), torch.ones((1, 5))],
            "hidden size and dtype",
        ),
        (
            [
                torch.ones((1, 4), dtype=torch.float16),
                torch.ones((1, 4), dtype=torch.float32),
            ],
            "hidden size and dtype",
        ),
    ],
)
def test_external_encoder_result_rejects_incompatible_artifacts(
    artifacts: list[torch.Tensor], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ExternalEncoderResult.from_artifacts(artifacts, image_token_id=99)


def test_external_encoder_result_covers_packed_rows() -> None:
    features = encode_request_plane_tensor(torch.ones((3, 4)))

    with pytest.raises(ValueError, match="do not cover"):
        ExternalEncoderResult(
            features=features,
            row_splits=(0, 2),
            image_token_id=99,
        )


def test_external_encoder_result_rejects_malformed_features() -> None:
    class RowSplitsThatMustNotBeRead(list[int]):
        def __iter__(self):
            raise AssertionError("row_splits read before features were validated")

    with pytest.raises(ValueError, match="features missing fields"):
        ExternalEncoderResult.from_dict(
            {
                "schema": "dynamo.external_encoder_result",
                "version": 0,
                "format": "linear_embeddings",
                "features": {"unexpected": "field"},
                "row_splits": RowSplitsThatMustNotBeRead([0, 1]),
                "image_token_id": 99,
            }
        )


def test_external_encoder_result_rejects_empty_image() -> None:
    features = encode_request_plane_tensor(torch.ones((3, 4)))

    with pytest.raises(ValueError, match="strictly increasing"):
        ExternalEncoderResult(
            features=features,
            row_splits=(0, 0, 3),
            image_token_id=99,
        )


class _Backend(VisionEncoderBackend):
    image_token_id = 99

    def __init__(self) -> None:
        self.model_id: str | None = None
        self.forward_calls: list[list[str]] = []
        self.closed = False

    def build(self, model_id: str) -> None:
        self.model_id = model_id

    def forward_batch(self, items, target_bucket=None):
        self.forward_calls.append(list(items))
        return [
            torch.full(
                (2 if item == "first" else 1, 4),
                float(len(item)),
                dtype=torch.bfloat16,
            )
            for item in items
        ]

    def close(self) -> None:
        self.closed = True


async def test_handoff_encodes_and_sanitizes_downstream_request() -> None:
    backend = _Backend()
    handoff = ExternalEncoderHandoff(backend)
    request = {
        "model": "encoder-model",
        "token_ids": [1, 99, 2, 99],
        "multi_modal_data": {"image_url": [{"Url": "first"}, {"Url": "second"}]},
        "multi_modal_uuids": {"image": ["one", "two"]},
        "mm_processor_kwargs": {"unused": True},
        "extra_args": {
            "keep": "value",
            "mm_hashes": ["one", "two"],
            "expanded_token_ids": [1, 99, 2, 99],
        },
        "sampling_options": {"max_tokens": 7},
    }
    original = deepcopy(request)

    handoff.load("encoder-model")
    try:
        downstream = await handoff.prepare_request(
            request,
            target_model="decoder-model",
        )
    finally:
        handoff.shutdown()

    assert request == original
    assert backend.model_id == "encoder-model"
    assert backend.closed is True
    assert [item for batch in backend.forward_calls for item in batch] == [
        "first",
        "second",
    ]
    assert downstream["model"] == "decoder-model"
    assert downstream["token_ids"] == [1, 99, 2, 99]
    assert downstream["sampling_options"] == {"max_tokens": 7}
    assert downstream["extra_args"] == {"keep": "value"}
    assert "multi_modal_data" not in downstream
    assert "multi_modal_uuids" not in downstream
    assert "mm_processor_kwargs" not in downstream

    result = ExternalEncoderResult.from_dict(downstream["encoder_result"])
    assert result.row_splits == (0, 2, 3)
    assert result.image_token_id == 99


@pytest.mark.parametrize(
    "request_value,match",
    [
        ({"encoder_result": {}}, "already contains encoder_result"),
        ({"prompt_embeds": []}, "cannot be combined with prompt_embeds"),
        ({}, "at least one image"),
        (
            {"multi_modal_data": {"audio_url": [{"Url": "audio"}]}},
            "image inputs only",
        ),
        (
            {"multi_modal_data": {"image_url": ["image"]}},
            "item 0 must be an object",
        ),
    ],
)
async def test_handoff_rejects_invalid_source_request(
    request_value: dict[str, Any], match: str
) -> None:
    handoff = ExternalEncoderHandoff(_Backend())

    with pytest.raises(InvalidArgument, match=match):
        await handoff.prepare_request(request_value, target_model="decoder-model")


async def test_handoff_rejects_empty_target_model_before_encoding() -> None:
    backend = _Backend()
    handoff = ExternalEncoderHandoff(backend)

    with pytest.raises(ValueError, match="target_model"):
        await handoff.prepare_request(
            {"multi_modal_data": {"image_url": [{"Url": "first"}]}},
            target_model="",
        )

    assert backend.forward_calls == []
