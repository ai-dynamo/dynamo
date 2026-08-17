# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the workflow adapter over stock NIXL WRITE transfer."""

import asyncio

import pytest
import torch

from dynamo.common.multimodal.embedding_transfer import TransferRequest
from dynamo.vllm.workflow.components.embedding_transfer import NixlWriteTensorCarrier

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _Sender:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, bool]] = []
        self.completions: list[asyncio.Future[None]] = []
        self.closed = False

    async def send_embeddings(self, tensor, stage_embeddings=False):
        self.calls.append((tensor, stage_embeddings))
        completion = asyncio.get_running_loop().create_future()
        self.completions.append(completion)
        return (
            TransferRequest(
                embeddings_shape=list(tensor.shape),
                embedding_dtype_str=str(tensor.dtype).removeprefix("torch."),
                serialized_request=f"request-{len(self.calls)}",
            ),
            completion,
        )

    async def close(self) -> None:
        self.closed = True


async def test_write_carrier_exports_existing_transfer_requests_per_edge() -> None:
    sender = _Sender()
    carrier = NixlWriteTensorCarrier(sender=sender, torch_module=torch)
    tensor = torch.ones((3, 8), dtype=torch.bfloat16)

    references = await carrier.export_tensor_fanout(
        tensor, ("classifier.embedding", "generator.embedding")
    )

    assert references == {
        "classifier.embedding": {
            "embeddings_shape": [3, 8],
            "embedding_dtype_str": "bfloat16",
            "serialized_request": "request-1",
        },
        "generator.embedding": {
            "embeddings_shape": [3, 8],
            "embedding_dtype_str": "bfloat16",
            "serialized_request": "request-2",
        },
    }
    assert sender.calls == [(tensor, True), (tensor, True)]

    for completion in sender.completions:
        completion.set_result(None)
    await asyncio.sleep(0)
    await carrier.close()
    assert sender.closed
