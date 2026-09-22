# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import torch

from dynamo.common.external_encoder import (
    ExternalEncoderResult,
    decode_request_plane_tensor,
)
from dynamo.experimental.llm import LLMUnaryClient
from dynamo.llm.exceptions import InvalidArgument
from examples.custom_encoder.remote.encoder import InlineEncoder
from examples.custom_encoder.remote.orchestrator import ExternalEncoderOrchestrator

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _Encoder:
    def __init__(self, artifacts: list[Any]) -> None:
        self.artifacts = artifacts
        self.raws: list[str] | None = None
        self.closed = False

    async def encode(self, raws: list[str]) -> list[Any]:
        self.raws = raws
        return self.artifacts

    def shutdown(self) -> None:
        self.closed = True


class _Client:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.request: dict[str, Any] | None = None
        self.context: Any = None

    async def round_robin(
        self,
        request: Any,
        *,
        annotated: bool,
        context: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        assert annotated is False
        self.request = request
        self.context = context

        async def stream() -> AsyncIterator[dict[str, Any]]:
            for chunk in self.chunks:
                yield chunk

        return stream()


def _request() -> dict[str, Any]:
    return {
        "token_ids": [1, 99, 2, 99, 3],
        "multi_modal_data": {
            "image_url": [
                {"Url": "https://example.com/one.png"},
                {"Url": "https://example.com/two.png"},
            ]
        },
        "multi_modal_uuids": {"image": ["one", "two"]},
        "mm_processor_kwargs": {"max_pixels": 1024},
        "mm_routing_info": {"routing_token_ids": [1, 2, 3]},
        "media_io_kwargs": {"timeout": 1},
        "extra_args": {
            "mm_processor_kwargs": {"max_pixels": 1024},
            "mm_kwargs_shm": {"unused": True},
            "mm_kwargs_nixl": {"unused": True},
            "mm_hashes": ["unused"],
            "mm_hashes_by_modality": {"image": ["unused"]},
            "mm_placeholders": [{"offset": 1, "length": 2}],
            "mm_placeholders_by_modality": {"image": [{"offset": 1, "length": 2}]},
            "expanded_token_ids": [1, 99, 99, 2, 3],
            "keep": 1,
        },
    }


async def test_orchestrator_packages_inline_encoder_result() -> None:
    first = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    second = torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)
    raw_encoder = _Encoder([first, second])
    encoder = InlineEncoder(raw_encoder, image_token_id=99)
    raw_client = _Client(
        [
            {"token_ids": [7], "index": 0},
            {"token_ids": [8], "index": 0, "finish_reason": "stop"},
        ]
    )
    context = object()
    request = _request()
    orchestrator = ExternalEncoderOrchestrator(
        encoder,
        LLMUnaryClient(raw_client),
        "decoder-model",
    )

    completion = await orchestrator(request, context=context)

    assert completion["token_ids"] == [7, 8]
    assert completion["finish_reason"] == "stop"
    assert raw_encoder.raws == [
        "https://example.com/one.png",
        "https://example.com/two.png",
    ]
    assert raw_client.context is context
    assert raw_client.request is not None
    assert raw_client.request["model"] == "decoder-model"
    for field_name in (
        "multi_modal_data",
        "multi_modal_uuids",
        "mm_processor_kwargs",
        "mm_routing_info",
        "media_io_kwargs",
    ):
        assert field_name not in raw_client.request
    assert raw_client.request["extra_args"] == {"keep": 1}
    assert "encoder_result" not in request

    result = ExternalEncoderResult.from_dict(raw_client.request["encoder_result"])
    assert result.row_splits == (0, 2, 3)
    torch.testing.assert_close(
        decode_request_plane_tensor(result.features),
        torch.cat([first, second]),
    )


async def test_inline_encoder_rejects_mismatched_artifacts() -> None:
    encoder = InlineEncoder(
        _Encoder([torch.ones((1, 4)), torch.ones((1, 5))]),
        image_token_id=99,
    )

    with pytest.raises(InvalidArgument, match="one hidden size"):
        await encoder.encode(_request())


async def test_inline_encoder_requires_images() -> None:
    encoder = InlineEncoder(_Encoder([]), image_token_id=99)

    with pytest.raises(InvalidArgument, match="at least one image"):
        await encoder.encode({"multi_modal_data": {}})


def test_inline_encoder_closes_driver() -> None:
    raw_encoder = _Encoder([])
    encoder = InlineEncoder(raw_encoder, image_token_id=99)

    encoder.close()

    assert raw_encoder.closed
