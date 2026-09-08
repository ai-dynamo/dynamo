# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for encode-worker multimodal helpers."""

import logging
from types import SimpleNamespace

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.multimodal_handlers import encode_worker_handler
from dynamo.vllm.multimodal_handlers.encode_worker_handler import (
    EmbeddingItem,
    EncodeWorkerHandler,
)
from dynamo.vllm.multimodal_utils.embedding_cache import generate_hash_key
from dynamo.vllm.multimodal_utils.protocol import MultiModalInput

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _handler(
    *, frontend_decoding: bool, capacity_bytes: int = 1 << 20
) -> EncodeWorkerHandler:
    """Build a handler without running __init__.

    The real __init__ loads an image processor and a vision tower from a
    checkpoint, which a unit test has no way to provide; every attribute the
    cache paths touch is set here instead.
    """
    handler = EncodeWorkerHandler.__new__(EncodeWorkerHandler)
    handler._enable_frontend_decoding = frontend_decoding
    handler._decoded_content_hash_warning_emitted = False
    handler.embedding_cache_manager = MultimodalEmbeddingCacheManager(capacity_bytes)
    return handler


def _embedding_item(values: torch.Tensor) -> EmbeddingItem:
    return EmbeddingItem(key=None, image_grid_thw=[], embeddings=values)


def test_prepare_embedding_transfers_coalesces_uneven_images():
    first = torch.arange(8, dtype=torch.float16).reshape(1, 2, 4)
    second = torch.arange(8, 20, dtype=torch.float16).reshape(1, 3, 4)
    items = [_embedding_item(first), _embedding_item(second)]

    transfers, indices = encode_worker_handler._prepare_embedding_transfers(
        items, coalesce=True
    )

    assert indices == [0, None]
    assert len(transfers) == 1
    assert torch.equal(transfers[0], torch.cat((first, second), dim=1))

    split_transfers, split_indices = encode_worker_handler._prepare_embedding_transfers(
        items, coalesce=False
    )
    assert split_transfers[0] is first
    assert split_transfers[1] is second
    assert split_indices == [0, 1]


def test_prepare_embedding_transfers_reuses_combined_encoder_output():
    combined = torch.randn(1, 5, 4)
    items = [
        _embedding_item(combined[:, :2]),
        _embedding_item(combined[:, 2:]),
    ]

    transfers, indices = encode_worker_handler._prepare_embedding_transfers(
        items,
        coalesce=True,
        combined_embedding=combined,
    )

    assert len(transfers) == 1
    assert transfers[0] is combined
    assert indices == [0, None]


def test_split_encode_controls_qwen_transfer_coalescing(monkeypatch):
    model = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

    monkeypatch.setattr(encode_worker_handler, "SPLIT_ENCODE", 0)
    assert encode_worker_handler._should_coalesce_embedding_transfers(model, 2)
    assert not encode_worker_handler._should_coalesce_embedding_transfers(model, 1)

    monkeypatch.setattr(encode_worker_handler, "SPLIT_ENCODE", 1)
    assert not encode_worker_handler._should_coalesce_embedding_transfers(model, 2)


def test_image_processor_receives_engine_mm_processor_kwargs(monkeypatch):
    expected = {"min_pixels": 65536, "max_pixels": 262144}
    sentinel = object()

    def mock_from_pretrained(model, **kwargs):
        assert model == "model"
        assert kwargs == {"trust_remote_code": True, **expected}
        return sentinel

    monkeypatch.setattr(
        encode_worker_handler.AutoImageProcessor,
        "from_pretrained",
        mock_from_pretrained,
    )
    engine_args = SimpleNamespace(
        model="model",
        trust_remote_code=True,
        mm_processor_kwargs=expected,
    )

    assert encode_worker_handler._load_image_processor(engine_args) is sentinel


def test_cache_key_for_url_image_is_unchanged():
    # Pinned literal, not a call to the helper under test: this digest is a
    # persisted cache key that must stay stable across releases.
    expected = "494a30704d4f32ac0b81739d18a66d3638d440cbc6f5669f6af66f840edee5ab"
    handler = _handler(frontend_decoding=False)
    group_input = MultiModalInput(image_url="https://example.com/a.png")

    assert handler._image_cache_key(group_input) == expected
    assert generate_hash_key("https://example.com/a.png") == expected


def test_cache_key_for_decoded_image_uses_content_hash():
    handler = _handler(frontend_decoding=True)
    group_input = MultiModalInput(
        image_decoded={"shape": [4, 4, 3], "content_hash": "0123456789abcdef"}
    )

    assert handler._image_cache_key(group_input) == "0123456789abcdef"


def test_decoded_image_without_hash_is_unkeyed_and_warns_once(caplog):
    handler = _handler(frontend_decoding=True)
    group_input = MultiModalInput(image_decoded={"shape": [4, 4, 3]})

    with caplog.at_level(logging.WARNING):
        assert handler._image_cache_key(group_input) is None
        assert handler._image_cache_key(group_input) is None

    assert caplog.text.count("missing or invalid canonical content_hash") == 1


def test_decoded_image_rejected_without_frontend_decoding():
    handler = _handler(frontend_decoding=False)
    group_input = MultiModalInput(
        image_decoded={"shape": [4, 4, 3], "content_hash": "0123456789abcdef"}
    )

    with pytest.raises(ValueError, match="not enabled on the encode worker"):
        handler._image_cache_key(group_input)


def test_empty_group_rejected():
    handler = _handler(frontend_decoding=True)

    with pytest.raises(ValueError, match="image_url or image_decoded"):
        handler._image_cache_key(MultiModalInput())
    with pytest.raises(ValueError, match="image_url or image_decoded"):
        handler._image_cache_key(None)


def test_group_with_url_and_decoded_image_rejected():
    handler = _handler(frontend_decoding=True)
    group_input = MultiModalInput(
        image_url="https://example.com/a.png",
        image_decoded={"content_hash": "0123456789abcdef"},
    )

    with pytest.raises(ValueError, match="Exactly one"):
        handler._image_cache_key(group_input)


def test_configured_capacity_sizes_the_cache(monkeypatch):
    monkeypatch.setattr(encode_worker_handler, "ENABLE_ENCODER_CACHE", 1)

    cache = encode_worker_handler._build_embedding_cache(0.25)

    assert isinstance(cache, MultimodalEmbeddingCacheManager)
    assert cache.stats["capacity_bytes"] == int(0.25 * 1024**3)


def test_unset_capacity_falls_back_to_a_bounded_default(monkeypatch):
    monkeypatch.setattr(encode_worker_handler, "ENABLE_ENCODER_CACHE", 1)

    # A 0 capacity flag means 'unset', not 'disabled': the cache is still built
    # and sized from DEFAULT_ENCODER_CACHE_CAPACITY_GB.
    cache = encode_worker_handler._build_embedding_cache(0)

    assert cache is not None
    assert cache.stats["capacity_bytes"] == int(
        encode_worker_handler.DEFAULT_ENCODER_CACHE_CAPACITY_GB * 1024**3
    )


def test_encoder_cache_switch_disables_the_cache(monkeypatch):
    monkeypatch.setattr(encode_worker_handler, "ENABLE_ENCODER_CACHE", 0)

    assert encode_worker_handler._build_embedding_cache(1.0) is None


def test_store_path_evicts_instead_of_growing_past_capacity():
    entry_bytes = 256 * 1024
    element_count = entry_bytes // torch.tensor([], dtype=torch.float32).element_size()
    handler = _handler(frontend_decoding=False, capacity_bytes=4 * entry_bytes)

    for index in range(5):
        handler._store_embedding_item(
            EmbeddingItem(
                key=f"key-{index}",
                image_grid_thw=[[1, 2, 2]],
                embeddings=torch.full((1, element_count), float(index)),
            )
        )

    stats = handler.embedding_cache_manager.stats
    assert stats["current_bytes"] <= stats["capacity_bytes"]
    assert stats["entries"] == 4
    assert stats["evictions"] == 1
    assert handler._lookup_embedding_item("key-0") is None
    assert handler._lookup_embedding_item("key-4") is not None


def test_store_path_does_not_pin_the_encoder_batch():
    # Embeddings reach the cache as split views over one encoder output, which
    # are already contiguous. Storing the view would charge the manager for one
    # image while keeping the whole batch's storage alive.
    handler = _handler(frontend_decoding=False)
    batch = torch.arange(8 * 1024, dtype=torch.float32).reshape(8, 1024)
    view = batch.split([1] * 8)[1].unsqueeze(0)
    assert view.is_contiguous()

    handler._store_embedding_item(
        EmbeddingItem(key="key", image_grid_thw=[[1, 1, 1]], embeddings=view)
    )

    cached = handler._lookup_embedding_item("key").embeddings
    assert torch.equal(cached, view)
    assert cached.untyped_storage().data_ptr() != batch.untyped_storage().data_ptr()
    # The entry owns exactly the bytes the manager charged for it.
    assert cached.untyped_storage().nbytes() == cached.element_size() * cached.numel()
    assert handler.embedding_cache_manager.stats["current_bytes"] == (
        cached.element_size() * cached.numel()
    )


def test_store_then_lookup_round_trips_tensor_and_grid():
    handler = _handler(frontend_decoding=False)
    embeddings = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    handler._store_embedding_item(
        EmbeddingItem(key="k", image_grid_thw=[[1, 4, 4]], embeddings=embeddings)
    )

    item = handler._lookup_embedding_item("k")

    assert item is not None
    assert item.key == "k"
    assert item.image_grid_thw == [[1, 4, 4]]
    assert torch.equal(item.embeddings, embeddings)
    assert handler.embedding_cache_manager.stats["hits"] == 1


def test_unkeyed_item_is_not_cached():
    handler = _handler(frontend_decoding=True)

    handler._store_embedding_item(
        EmbeddingItem(key=None, image_grid_thw=[], embeddings=torch.zeros(1, 4))
    )

    assert handler.embedding_cache_manager.stats["entries"] == 0
    assert handler._lookup_embedding_item(None) is None


def test_non_contiguous_embedding_is_stored():
    # The manager asserts contiguity when sizing an entry; the old dict cache
    # never did, so a transposed view must be made contiguous on the way in.
    handler = _handler(frontend_decoding=False)
    view = torch.arange(8, dtype=torch.float32).reshape(2, 4).t()
    assert not view.is_contiguous()

    handler._store_embedding_item(
        EmbeddingItem(key="k", image_grid_thw=[], embeddings=view)
    )

    item = handler._lookup_embedding_item("k")
    assert item is not None
    assert torch.equal(item.embeddings, view)
