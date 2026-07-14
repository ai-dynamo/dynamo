# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the public Qwen2.5-VL CustomEncoder backend."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from dynamo.vllm.multimodal_utils.vision_encoder_backend import Preprocessed
from examples.custom_encoder.qwen2_vl_vision_encoder import (
    Qwen2VLImageInputs,
    Qwen2VLVisionEncoder,
    _parse_graph_buckets,
    _parse_graph_image_sizes,
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

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Size, torch.Tensor]] = []

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((pixel_values.shape, grid_thw.cpu().clone()))
        output_tokens = int((grid_thw.prod(dim=-1) // 4).sum().item())
        return torch.arange(output_tokens * 4, dtype=torch.bfloat16).reshape(
            output_tokens, 4
        )


def _item(grid: tuple[int, int, int]) -> Qwen2VLImageInputs:
    patch_count = grid[0] * grid[1] * grid[2]
    return Qwen2VLImageInputs(
        pixel_values=torch.ones((patch_count, 6)),
        image_grid_thw=torch.tensor([grid]),
    )


def test_forward_batch_packs_and_splits_variable_resolution_inputs() -> None:
    encoder = Qwen2VLVisionEncoder()
    encoder._device = torch.device("cpu")
    encoder._visual = _FakeVisual()

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
    items: list[Qwen2VLImageInputs], target_bucket: int | None, message: str
) -> None:
    encoder = Qwen2VLVisionEncoder()
    encoder._device = torch.device("cpu")
    encoder._visual = _FakeVisual()

    with pytest.raises(ValueError, match=message):
        encoder.forward_batch(items, target_bucket=target_bucket)


def test_forward_batch_requires_build() -> None:
    with pytest.raises(RuntimeError, match="before build"):
        Qwen2VLVisionEncoder().forward_batch([_item((1, 4, 4))])


def test_load_image_accepts_local_and_base64_sources(tmp_path) -> None:
    path = tmp_path / "source.png"
    Image.new("RGBA", (7, 5), color=(1, 2, 3, 255)).save(path)

    buffer = io.BytesIO()
    Image.new("RGB", (11, 9), color=(4, 5, 6)).save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    local = Qwen2VLVisionEncoder._load_image(str(path))
    inline = Qwen2VLVisionEncoder._load_image(data_url)

    assert (local.mode, local.size) == ("RGB", (7, 5))
    assert (inline.mode, inline.size) == ("RGB", (11, 9))


def test_load_image_rejects_malformed_data_url() -> None:
    with pytest.raises(ValueError, match="missing comma"):
        Qwen2VLVisionEncoder._load_image("data:image/png;base64")


def test_preprocess_cache_disabled_hit_and_lru_eviction() -> None:
    encoder = Qwen2VLVisionEncoder()
    calls: list[str] = []

    def preprocess(raw: str) -> Preprocessed[str]:
        calls.append(raw)
        return Preprocessed(item=raw)

    encoder._preprocess_uncached = preprocess  # type: ignore[method-assign]
    encoder._configure_preprocess_cache(0)
    assert encoder._cached_preprocess is None

    encoder._configure_preprocess_cache(2)
    cache = encoder._cached_preprocess
    assert cache is not None
    first = cache("image-a")
    assert cache("image-a") is first
    cache("image-b")
    cache("image-a")
    cache("image-c")
    cache("image-b")

    assert calls == ["image-a", "image-b", "image-c", "image-b"]
    assert cache.cache_info().hits == 2
    assert cache.cache_info().misses == 4
    assert cache.cache_info().currsize == 2


def test_preprocess_cache_rejects_negative_capacity() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        Qwen2VLVisionEncoder()._configure_preprocess_cache(-1)


def test_build_failure_clears_partial_state(monkeypatch) -> None:
    encoder = Qwen2VLVisionEncoder()

    def fail_after_setup(model_id: str) -> None:
        del model_id
        encoder._device = torch.device("cpu")
        encoder._processor = object()
        encoder._visual = _FakeVisual()
        encoder.tokenizer = object()
        raise RuntimeError("partial build failed")

    monkeypatch.setattr(encoder, "_build", fail_after_setup)

    with pytest.raises(RuntimeError, match="partial build failed"):
        encoder.build("model")

    assert encoder._processor is None
    assert encoder._visual is None
    assert encoder.tokenizer is None


def test_visual_architecture_validation() -> None:
    config = SimpleNamespace(
        depth=32,
        out_hidden_size=2048,
        patch_size=14,
        spatial_merge_size=2,
        window_size=112,
        fullatt_block_indexes=[7, 15, 23, 31],
    )
    Qwen2VLVisionEncoder._validate_visual_architecture(SimpleNamespace(config=config))

    config.depth = 31
    with pytest.raises(ValueError, match="window-attention architecture"):
        Qwen2VLVisionEncoder._validate_visual_architecture(
            SimpleNamespace(config=config)
        )


@pytest.mark.parametrize("value", ["", "0", "2,1", "1,1", "one"])
def test_graph_bucket_configuration_rejects_invalid_values(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS", value)
    with pytest.raises(ValueError, match="GRAPH_BATCH_BUCKETS"):
        _parse_graph_buckets()


def test_graph_bucket_configuration(monkeypatch) -> None:
    monkeypatch.setenv("DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS", "1,4,8")
    assert _parse_graph_buckets() == (1, 4, 8)


@pytest.mark.parametrize("value", ["", "500", "0x500", "axb"])
def test_graph_image_size_configuration_rejects_invalid_values(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("DYN_QWEN2_VL_GRAPH_IMAGE_SIZES", value)
    with pytest.raises(ValueError, match="GRAPH_IMAGE_SIZES"):
        _parse_graph_image_sizes()


def test_graph_image_size_configuration(monkeypatch) -> None:
    monkeypatch.setenv("DYN_QWEN2_VL_GRAPH_IMAGE_SIZES", "299x299,500x500")
    assert _parse_graph_image_sizes() == ((299, 299), (500, 500))
