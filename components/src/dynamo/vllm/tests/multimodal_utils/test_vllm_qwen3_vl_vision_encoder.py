# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the public Qwen3-VL CustomEncoder backend."""

from __future__ import annotations

import base64
import hashlib
import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.multimodal_utils.vision_encoder_backend import Preprocessed
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


def _item(
    grid: tuple[int, int, int], content_digest: bytes | None = None
) -> Qwen3VLImageInputs:
    patch_count = grid[0] * grid[1] * grid[2]
    return Qwen3VLImageInputs(
        pixel_values=torch.ones((patch_count, 6)),
        image_grid_thw=torch.tensor([grid]),
        content_digest=content_digest,
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

    local, local_digest = Qwen3VLVisionEncoder._load_image(str(path))
    inline, inline_digest = Qwen3VLVisionEncoder._load_image(data_url)

    assert (local.mode, local.size) == ("RGB", (7, 5))
    assert (inline.mode, inline.size) == ("RGB", (11, 9))
    assert local_digest == hashlib.sha256(path.read_bytes()).digest()
    assert inline_digest == hashlib.sha256(buffer.getvalue()).digest()


def test_load_image_rejects_malformed_data_url() -> None:
    with pytest.raises(ValueError, match="missing comma"):
        Qwen3VLVisionEncoder._load_image("data:image/png;base64")


def test_preprocess_cache_disabled_hit_and_lru_eviction():
    encoder = Qwen3VLVisionEncoder()
    calls: list[str] = []

    def preprocess(raw: str):
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
    cache("image-a")  # make image-b the least-recently-used entry
    cache("image-c")  # evicts least-recently-used image-b
    cache("image-b")

    assert calls == ["image-a", "image-b", "image-c", "image-b"]
    assert cache.cache_info().hits == 2
    assert cache.cache_info().misses == 4
    assert cache.cache_info().currsize == 2


def test_preprocess_cache_rejects_negative_capacity():
    encoder = Qwen3VLVisionEncoder()
    with pytest.raises(ValueError, match="must be >= 0"):
        encoder._configure_preprocess_cache(-1)


def test_embedding_cache_skips_forward_and_owns_output() -> None:
    encoder = Qwen3VLVisionEncoder()
    encoder._device = torch.device("cpu")
    encoder._visual = _FakeVisual()
    encoder._embedding_cache = MultimodalEmbeddingCacheManager(1024)
    item = _item((1, 4, 4), b"a" * 32)

    first = encoder.forward_batch([item])[0]
    second = encoder.forward_batch([item])[0]
    expected = second.clone()
    first.zero_()
    third = encoder.forward_batch([item])[0]

    assert len(encoder._visual.calls) == 1
    assert first is not second
    assert torch.equal(second, expected)
    assert torch.equal(third, expected)
    assert second.is_contiguous()
    assert second._base is None
    assert encoder._embedding_cache.stats["hits"] == 2
    assert encoder._embedding_cache.stats["misses"] == 1


def test_embedding_cache_deduplicates_misses_and_resizes_graph_bucket() -> None:
    encoder = Qwen3VLVisionEncoder()
    encoder._device = torch.device("cpu")
    encoder._visual = _FakeVisual()
    encoder._embedding_cache = MultimodalEmbeddingCacheManager(4096)
    calls: list[tuple[int, int | None]] = []

    def forward(items, target_bucket):
        calls.append((len(items), target_bucket))
        return [
            torch.full((4, 4), index, dtype=torch.bfloat16)
            for index in range(len(items))
        ]

    encoder._forward_uncached = forward  # type: ignore[method-assign]
    items = [
        _item((1, 4, 4), b"a" * 32),
        _item((1, 4, 4), b"a" * 32),
        _item((1, 4, 4), b"b" * 32),
        _item((1, 4, 4), b"c" * 32),
    ]

    first = encoder.forward_batch(items, target_bucket=8)
    second = encoder.forward_batch(items, target_bucket=8)
    third = encoder.forward_batch(
        [items[0], items[1], items[2], _item((1, 4, 4), b"d" * 32)],
        target_bucket=8,
    )

    assert calls == [(3, 4), (1, 1)]
    assert first[0] is not first[1]
    assert torch.equal(first[0], first[1])
    assert all(before is not after for before, after in zip(first, second))
    assert all(torch.equal(before, after) for before, after in zip(first, second))
    assert third[-1][0, 0] == 0


def test_embedding_cache_rejects_invalid_batch_transactionally() -> None:
    encoder = Qwen3VLVisionEncoder()
    encoder._device = torch.device("cpu")
    encoder._visual = _FakeVisual()
    encoder._embedding_cache = MultimodalEmbeddingCacheManager(4096)

    def invalid_forward(items, target_bucket):
        del target_bucket
        return [torch.ones((4, 4), dtype=torch.bfloat16), torch.ones((3, 4))]

    encoder._forward_uncached = invalid_forward  # type: ignore[method-assign]
    items = [
        _item((1, 4, 4), b"a" * 32),
        _item((1, 4, 4), b"b" * 32),
    ]

    with pytest.raises(RuntimeError, match="invalid embedding"):
        encoder.forward_batch(items, target_bucket=2)

    assert encoder._embedding_cache.stats["entries"] == 0


def test_disabled_embedding_cache_skips_content_digest(tmp_path) -> None:
    path = tmp_path / "source.png"
    Image.new("RGB", (5, 5), color="red").save(path)

    _, digest = Qwen3VLVisionEncoder._load_image(str(path), compute_digest=False)

    assert digest is None


def test_embedding_cache_capacity_configuration(monkeypatch) -> None:
    monkeypatch.delenv("DYN_QWEN3_VL_EMBEDDING_CACHE_BYTES", raising=False)
    assert Qwen3VLVisionEncoder._parse_embedding_cache_capacity() == 1024**3

    monkeypatch.setenv("DYN_QWEN3_VL_EMBEDDING_CACHE_BYTES", "0")
    assert Qwen3VLVisionEncoder._parse_embedding_cache_capacity() == 0

    monkeypatch.setenv("DYN_QWEN3_VL_EMBEDDING_CACHE_BYTES", "-1")
    with pytest.raises(ValueError, match="nonnegative integer"):
        Qwen3VLVisionEncoder._parse_embedding_cache_capacity()
