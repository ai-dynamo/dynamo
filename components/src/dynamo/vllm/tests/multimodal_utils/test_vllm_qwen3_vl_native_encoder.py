# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the native Qwen3-VL custom encoder."""

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder import Qwen3VLImageEncoding
from examples.custom_encoder.qwen3_vl_native_encoder import (
    Qwen3VLImageInputs,
    Qwen3VLNativeEncoder,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeVisual(torch.nn.Module):
    dtype = torch.bfloat16
    spatial_merge_size = 2

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> SimpleNamespace:
        token_count = int((grid_thw.prod(dim=-1) // 4).sum().item())
        primary = torch.ones((token_count, 4), dtype=torch.bfloat16)
        return SimpleNamespace(
            pooler_output=primary,
            deepstack_features=[primary * 2, primary * 3],
        )


def _item(grid: tuple[int, int, int]) -> Qwen3VLImageInputs:
    return Qwen3VLImageInputs(
        pixel_values=torch.ones((grid[0] * grid[1] * grid[2], 6)),
        image_grid_thw=torch.tensor([grid]),
    )


def test_forward_batch_packs_primary_and_deepstack_features():
    encoder = Qwen3VLNativeEncoder()
    encoder._device = torch.device("cpu")
    encoder._visual = _FakeVisual()

    outputs = encoder.forward_batch([_item((1, 2, 2)), _item((1, 2, 4))])

    assert all(isinstance(output, Qwen3VLImageEncoding) for output in outputs)
    assert [output.grid_thw for output in outputs] == [(1, 2, 2), (1, 2, 4)]
    assert [tuple(output.embeddings.shape) for output in outputs] == [(1, 12), (2, 12)]
    assert outputs[0].embeddings[0].tolist() == [1] * 4 + [2] * 4 + [3] * 4
