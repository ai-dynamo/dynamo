# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for encode-worker multimodal helpers."""

import importlib.util
import logging
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.constants import EmbeddingTransferMode
from dynamo.vllm.multimodal_handlers import encode_worker_handler
from dynamo.vllm.multimodal_handlers.encode_worker_handler import (
    EmbeddingItem,
    EncodeWorkerHandler,
)
from dynamo.vllm.multimodal_utils.encode_utils import get_embedding_hash
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


def _image_loader_class_reading_current_env() -> type:
    """Execute a private copy of the module so the environment-derived ``ImageLoader``
    cache-size default is refreshed without mutating the shared module.
    """
    spec = importlib.util.find_spec("dynamo.common.multimodal.image_loader")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not locate dynamo.common.multimodal.image_loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ImageLoader


def _encode_handler_with_cache_env(monkeypatch, cache_size_env) -> EncodeWorkerHandler:
    """Run the real ``EncodeWorkerHandler.__init__`` with the heavy parts stubbed."""
    monkeypatch.setenv("DYN_MM_IMAGE_CACHE_SIZE", cache_size_env)

    monkeypatch.setattr(
        encode_worker_handler, "ImageLoader", _image_loader_class_reading_current_env()
    )
    monkeypatch.setattr(
        encode_worker_handler, "_load_image_processor", lambda engine_args: object()
    )
    monkeypatch.setattr(
        encode_worker_handler, "load_vision_model", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        encode_worker_handler,
        "get_encoder_components",
        lambda *args, **kwargs: (object(), object()),
    )

    engine_args = SimpleNamespace(
        model="model",
        trust_remote_code=False,
        enforce_eager=True,
    )
    return EncodeWorkerHandler(engine_args, EmbeddingTransferMode.LOCAL)


@pytest.mark.parametrize(
    "cache_size_env, expected_capacity",
    [("2", 2), ("0", 0)],
    ids=["env-set", "env-set-zero"],
)
async def test_encode_worker_image_cache_capacity_follows_env(
    monkeypatch, cache_size_env, expected_capacity
):
    handler = _encode_handler_with_cache_env(monkeypatch, cache_size_env)
    try:
        loader = handler.image_loader
        keys = [
            f"https://example.com/{index}.png" for index in range(expected_capacity + 1)
        ]
        for key in keys:
            loader._cache_put(key, Image.new("RGB", (4, 4), color="blue"))

        assert len(loader._image_cache) == expected_capacity
        assert keys[0] not in loader._image_cache
        if expected_capacity:
            assert keys[-1] in loader._image_cache
    finally:
        handler.cleanup()
        await handler.send_complete_checker_task


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
    assert get_embedding_hash("https://example.com/a.png") == expected


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


@pytest.mark.parametrize("capacity_gb", [0, -1.0])
def test_non_positive_capacity_disables_the_cache(monkeypatch, capacity_gb):
    # 0 is the flag's default and its documented 'disabled' value, so a stock
    # deployment runs without this cache rather than with an implicit one.
    monkeypatch.setattr(encode_worker_handler, "ENABLE_ENCODER_CACHE", 1)

    assert encode_worker_handler._build_embedding_cache(capacity_gb) is None


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


class _RecordingEmbedding:
    """Stand-in embedding that records cache state at the moment it is copied.

    Whether the cache had room before the copy was made is not recoverable
    from its final state, so the assertion has to be made between the
    admission decision and the copy. Only the three members the store path
    uses are implemented; everything else about a tensor is out of scope.
    """

    def __init__(
        self, tensor: torch.Tensor, manager: MultimodalEmbeddingCacheManager
    ) -> None:
        self._tensor = tensor
        self._manager = manager
        self.clone_calls = 0
        self.stats_at_clone: dict | None = None

    def element_size(self) -> int:
        return self._tensor.element_size()

    def numel(self) -> int:
        return self._tensor.numel()

    def clone(self, memory_format=None) -> torch.Tensor:
        self.clone_calls += 1
        self.stats_at_clone = self._manager.stats
        return self._tensor.clone(memory_format=memory_format)


def _float32_element_count(entry_bytes: int) -> int:
    return entry_bytes // torch.tensor([], dtype=torch.float32).element_size()


def _fill_cache(handler: EncodeWorkerHandler, count: int, element_count: int) -> None:
    for index in range(count):
        handler._store_embedding_item(
            EmbeddingItem(
                key=f"key-{index}",
                image_grid_thw=[[1, 2, 2]],
                embeddings=torch.full((1, element_count), float(index)),
            )
        )


def test_store_path_makes_room_before_copying_into_the_cache():
    entry_bytes = 256 * 1024
    element_count = _float32_element_count(entry_bytes)
    handler = _handler(frontend_decoding=False, capacity_bytes=4 * entry_bytes)
    _fill_cache(handler, 4, element_count)
    assert handler.embedding_cache_manager.stats["current_bytes"] == 4 * entry_bytes

    probe = _RecordingEmbedding(
        torch.full((1, element_count), 4.0), handler.embedding_cache_manager
    )
    handler._store_embedding_item(
        EmbeddingItem(key="key-4", image_grid_thw=[[1, 2, 2]], embeddings=probe)
    )

    assert probe.clone_calls == 1
    # The bytes this entry needs were already free when the copy was made, so a
    # full cache never holds its whole capacity and the new copy at once.
    assert (
        probe.stats_at_clone["current_bytes"] + entry_bytes
        <= probe.stats_at_clone["capacity_bytes"]
    )
    assert probe.stats_at_clone["evictions"] == 1
    # Evicting early neither counts the eviction twice nor changes what the
    # cache ends up holding.
    stats = handler.embedding_cache_manager.stats
    assert stats["evictions"] == 1
    assert stats["entries"] == 4
    assert stats["current_bytes"] == stats["capacity_bytes"]
    assert handler._lookup_embedding_item("key-0") is None
    assert handler._lookup_embedding_item("key-4") is not None


def test_store_path_rejects_an_oversize_entry_without_copying_it():
    entry_bytes = 256 * 1024
    element_count = _float32_element_count(entry_bytes)
    handler = _handler(frontend_decoding=False, capacity_bytes=entry_bytes // 2)
    probe = _RecordingEmbedding(
        torch.zeros(1, element_count), handler.embedding_cache_manager
    )

    handler._store_embedding_item(
        EmbeddingItem(key="key", image_grid_thw=[[1, 2, 2]], embeddings=probe)
    )

    # Rejected on its byte count alone, so the copy that the cache would have
    # refused to keep is never allocated.
    assert probe.clone_calls == 0
    stats = handler.embedding_cache_manager.stats
    assert stats["entries"] == 0
    assert stats["current_bytes"] == 0
    assert stats["evictions"] == 0


def test_restoring_a_cached_key_frees_the_old_entry_before_copying():
    # One request can carry the same uncached image twice: both misses are
    # queued for encoding and stored in turn, so the second store re-stores a
    # key the first just made resident. Deducting the old entry's bytes without
    # releasing them would keep its storage alive across the copy, and a full
    # cache would peak at capacity plus one entry after all.
    entry_bytes = 256 * 1024
    element_count = _float32_element_count(entry_bytes)
    handler = _handler(frontend_decoding=False, capacity_bytes=4 * entry_bytes)
    _fill_cache(handler, 4, element_count)
    assert handler.embedding_cache_manager.stats["current_bytes"] == 4 * entry_bytes

    probe = _RecordingEmbedding(
        torch.full((1, element_count), 9.0), handler.embedding_cache_manager
    )
    handler._store_embedding_item(
        EmbeddingItem(key="key-1", image_grid_thw=[[1, 2, 2]], embeddings=probe)
    )

    assert probe.clone_calls == 1
    # The replaced entry is gone before the copy is taken, not merely promised
    # back, so the bytes the copy needs are genuinely free.
    assert probe.stats_at_clone["entries"] == 3
    assert (
        probe.stats_at_clone["current_bytes"] + entry_bytes
        <= probe.stats_at_clone["capacity_bytes"]
    )
    # Dropping the replaced entry is not an eviction: nothing was displaced for
    # want of capacity, and no other key was touched.
    assert probe.stats_at_clone["evictions"] == 0
    stats = handler.embedding_cache_manager.stats
    assert stats["evictions"] == 0
    assert stats["entries"] == 4
    assert stats["current_bytes"] == 4 * entry_bytes
    assert handler._lookup_embedding_item("key-0") is not None


def test_restoring_a_cached_key_evicts_nothing():
    # Making room for a re-store frees the entry being replaced, so its bytes
    # cover the incoming entry and no other key has to go.
    entry_bytes = 256 * 1024
    element_count = _float32_element_count(entry_bytes)
    handler = _handler(frontend_decoding=False, capacity_bytes=4 * entry_bytes)
    _fill_cache(handler, 4, element_count)

    handler._store_embedding_item(
        EmbeddingItem(
            key="key-1",
            image_grid_thw=[[1, 2, 2]],
            embeddings=torch.full((1, element_count), 9.0),
        )
    )

    stats = handler.embedding_cache_manager.stats
    assert stats["evictions"] == 0
    assert stats["entries"] == 4
    assert stats["current_bytes"] == 4 * entry_bytes
    assert handler._lookup_embedding_item("key-0") is not None
    assert torch.equal(
        handler._lookup_embedding_item("key-1").embeddings,
        torch.full((1, element_count), 9.0),
    )


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
