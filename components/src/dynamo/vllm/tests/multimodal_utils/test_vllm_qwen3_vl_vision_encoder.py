# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the public Qwen3-VL CustomEncoder backend."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.custom_encoder.qwen3_vl_vision_encoder import (
    Qwen3VLImageInputs,
    Qwen3VLVisionEncoder,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeVisual(torch.nn.Module):
    spatial_merge_size = 2
    dtype = torch.bfloat16

    def __init__(self, return_object: bool = False) -> None:
        super().__init__()
        self.return_object = return_object
        self.calls: list[tuple[torch.Size, torch.Tensor]] = []

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | SimpleNamespace:
        self.calls.append((pixel_values.shape, grid_thw.cpu().clone()))
        output_tokens = int((grid_thw.prod(dim=-1) // 4).sum().item())
        embeds = torch.arange(output_tokens * 4, dtype=torch.bfloat16).reshape(
            output_tokens, 4
        )
        if self.return_object:
            return SimpleNamespace(pooler_output=embeds, deepstack_features=[])
        return embeds, []


def _item(grid: tuple[int, int, int]) -> Qwen3VLImageInputs:
    patch_count = grid[0] * grid[1] * grid[2]
    return Qwen3VLImageInputs(
        pixel_values=torch.ones((patch_count, 6)),
        image_grid_thw=torch.tensor([grid]),
    )


@pytest.mark.parametrize("return_object", [False, True])
def test_forward_batch_packs_and_splits_variable_resolution_inputs(
    return_object: bool,
) -> None:
    encoder = Qwen3VLVisionEncoder()
    encoder._device = torch.device("cpu")
    encoder._visual = _FakeVisual(return_object=return_object)

    outputs = encoder.forward_batch([_item((1, 4, 4)), _item((1, 8, 8))])

    assert [output.shape for output in outputs] == [(4, 4), (16, 4)]
    assert all(output.device.type == "cpu" for output in outputs)
    assert all(output.dtype == torch.bfloat16 for output in outputs)
    assert encoder._visual.calls[0][0] == (80, 6)
    assert encoder._visual.calls[0][1].tolist() == [[1, 4, 4], [1, 8, 8]]
    assert outputs[1][0, 0] == 16


@pytest.mark.parametrize(
    ("items", "target_bucket", "message"),
    [
        ([], None, "at least one"),
        ([_item((1, 4, 4))] * 9, None, "max_batch_cost"),
        ([_item((1, 4, 4))], 8, "target_bucket"),
    ],
)
def test_forward_batch_rejects_invalid_dispatches(
    items: list[Qwen3VLImageInputs], target_bucket: int | None, message: str
) -> None:
    encoder = Qwen3VLVisionEncoder()
    encoder._device = torch.device("cpu")
    encoder._visual = _FakeVisual()

    with pytest.raises(ValueError, match=message):
        encoder.forward_batch(items, target_bucket=target_bucket)


def test_forward_batch_requires_build() -> None:
    with pytest.raises(RuntimeError, match="before build"):
        Qwen3VLVisionEncoder().forward_batch([_item((1, 4, 4))])


def test_load_image_accepts_local_and_base64_sources(tmp_path) -> None:
    path = tmp_path / "source.png"
    Image.new("RGBA", (7, 5), color=(1, 2, 3, 255)).save(path)

    buffer = io.BytesIO()
    Image.new("RGB", (11, 9), color=(4, 5, 6)).save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    local = Qwen3VLVisionEncoder._load_image(str(path))
    inline = Qwen3VLVisionEncoder._load_image(data_url)

    assert (local.mode, local.size) == ("RGB", (7, 5))
    assert (inline.mode, inline.size) == ("RGB", (11, 9))


def test_load_image_rejects_malformed_data_url() -> None:
    with pytest.raises(ValueError, match="missing comma"):
        Qwen3VLVisionEncoder._load_image("data:image/png;base64")
