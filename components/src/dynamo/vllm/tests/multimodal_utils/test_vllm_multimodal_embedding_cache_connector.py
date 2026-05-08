# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMultimodalEmbeddingCacheConnector."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from dynamo.vllm.multimodal_utils import multimodal_embedding_cache_connector as mod

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _make_vllm_config(capacity_gb: float = 1.0) -> MagicMock:
    config = MagicMock()
    config.ec_transfer_config.ec_connector_extra_config = {
        "multimodal_embedding_cache_capacity_gb": capacity_gb,
    }
    config.model_config.get_hidden_size.return_value = 4096
    config.model_config.dtype = torch.float16
    return config


class TestVersionCheck:
    def test_warns_old_vllm(self):
        with (
            patch.object(mod, "_vllm_version", "0.16.5"),
            patch.object(mod.ECConnectorBase, "__init__", return_value=None),
            patch.object(mod.logger, "warning") as mock_warn,
        ):
            connector = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(),
                role=MagicMock(),
            )
            assert connector is not None
            mock_warn.assert_called_once()
            assert mock_warn.call_args[0][1] == mod.MINIMUM_VLLM_VERSION
            assert mock_warn.call_args[0][2] == "0.16.5"


class TestSchedulerSideLRU:
    """Test the scheduler-side logical LRU cache and metadata generation."""

    def _make_connector(self, capacity_gb: float = 1.0):
        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            return mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(capacity_gb),
                role=MagicMock(),
            )

    def _make_request(self, hashes_and_embeds: list[tuple[str, int]]) -> MagicMock:
        request = MagicMock()
        features = []
        for h, _ in hashes_and_embeds:
            f = MagicMock()
            f.identifier = h
            features.append(f)
        request.mm_features = features

        def get_num_encoder_embeds(idx):
            return hashes_and_embeds[idx][1]

        request.get_num_encoder_embeds = get_num_encoder_embeds
        return request

    def test_has_cache_item_miss_then_hit(self):
        conn = self._make_connector()
        assert not conn.has_cache_item("hash_a")

        request = self._make_request([("hash_a", 100)])
        conn.update_state_after_alloc(request, 0)

        assert conn.has_cache_item("hash_a")

    def test_update_state_plans_save(self):
        conn = self._make_connector()
        request = self._make_request([("hash_a", 100)])
        conn.update_state_after_alloc(request, 0)

        scheduler_output = MagicMock()
        meta = conn.build_connector_meta(scheduler_output)
        assert isinstance(meta, mod.MultimodalEmbeddingCacheConnectorMetadata)
        assert "hash_a" in meta.saves
        assert meta.loads == []
        assert meta.evicts == []

    def test_update_state_plans_load_for_cached(self):
        conn = self._make_connector()
        request = self._make_request([("hash_a", 100)])

        conn.update_state_after_alloc(request, 0)
        conn.build_connector_meta(MagicMock())

        conn.update_state_after_alloc(request, 0)
        meta = conn.build_connector_meta(MagicMock())
        assert "hash_a" in meta.loads
        assert meta.saves == []

    def test_eviction_under_pressure(self):
        # 4096 hidden_size * 2 bytes (fp16) = 8192 bytes per embed
        conn = self._make_connector()
        bpe = conn._bytes_per_embed  # 8192
        # Set capacity to hold exactly 200 embeds worth of bytes
        conn._capacity_bytes = 200 * bpe

        req_a = self._make_request([("hash_a", 100)])
        conn.update_state_after_alloc(req_a, 0)
        conn.build_connector_meta(MagicMock())

        req_b = self._make_request([("hash_b", 100)])
        conn.update_state_after_alloc(req_b, 0)
        conn.build_connector_meta(MagicMock())

        assert conn._num_used_bytes == 200 * bpe

        # Adding hash_c (100 embeds) should evict hash_a (LRU)
        req_c = self._make_request([("hash_c", 100)])
        conn.update_state_after_alloc(req_c, 0)
        meta = conn.build_connector_meta(MagicMock())

        assert "hash_c" in meta.saves
        assert "hash_a" in meta.evicts
        assert "hash_a" not in conn._cache_order
        assert "hash_c" in conn._cache_order

    def test_skip_oversized_item(self):
        conn = self._make_connector()
        bpe = conn._bytes_per_embed
        conn._capacity_bytes = 50 * bpe

        request = self._make_request([("huge_hash", 100)])
        conn.update_state_after_alloc(request, 0)
        meta = conn.build_connector_meta(MagicMock())

        assert meta.saves == []
        assert meta.loads == []
        assert "huge_hash" not in conn._cache_order


# ---------------------------------------------------------------------------
# Worker-side fixtures for start_load_caches / save_caches /
# clear_connector_metadata tests. Avoid the real `_setup_worker` (cudaHostRegister
# + shared_memory.SharedMemory) by pre-populating the worker state.
# ---------------------------------------------------------------------------


def _make_worker_connector(
    *,
    tp_rank: int = 0,
    tp_world_size: int = 2,
    feature_dim: int = 4,
    arena_bytes: int = 4096,
    fence_active: bool = True,
) -> mod.DynamoMultimodalEmbeddingCacheConnector:
    """Build a connector with worker-side state ready for load/save calls,
    bypassing _setup_worker (which would touch shm + CUDA)."""
    with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
        conn = mod.DynamoMultimodalEmbeddingCacheConnector(
            vllm_config=_make_vllm_config(),
            role=MagicMock(),
        )
    conn._tp_rank = tp_rank
    conn._tp_world_size = tp_world_size
    conn._worker_initialized = True
    conn._nccl_fence_active = fence_active
    conn._model_dtype = torch.float16
    conn._feature_dim = feature_dim
    conn._save_stream = MagicMock()

    # Deterministic arena: byte at offset i is `i % 256`. Lets tests assert
    # that views map to the right arena slice.
    pattern = (torch.arange(arena_bytes, dtype=torch.long) % 256).to(torch.uint8)
    conn._arena_uint8 = pattern.contiguous()
    conn._connector_metadata = None
    return conn


class _LoadCallTracker:
    """Captures every Tensor.to / Tensor.record_stream call during a
    start_load_caches invocation. Class-method patching (not autospec) so
    Python's descriptor protocol binds `self` for us."""

    def __init__(self) -> None:
        self.to_calls: list[tuple[torch.Tensor, tuple, dict]] = []
        self.record_stream_calls: list[tuple[int, int]] = []  # (data_ptr, numel)


def _patch_cuda_for_load(monkeypatch) -> _LoadCallTracker:
    """Patch the three CUDA-only call sites used by start_load_caches so the
    method runs end-to-end on CPU. Returns the tracker for assertions."""
    tracker = _LoadCallTracker()
    fake_stream = MagicMock()
    fake_stream.device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **kw: fake_stream)

    def fake_to(self, *args, **kwargs):
        tracker.to_calls.append((self, args, kwargs))
        # No-op transfer: stay on CPU and return self so the view chain is
        # built off the arena buffer (lets test_view_content_* read bytes).
        return self

    def fake_record_stream(self, _stream):
        tracker.record_stream_calls.append((self.data_ptr(), self.numel()))

    monkeypatch.setattr(torch.Tensor, "to", fake_to, raising=False)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", fake_record_stream, raising=False
    )
    return tracker


def _start_load(monkeypatch, conn, metadata, encoder_cache) -> _LoadCallTracker:
    """Drive start_load_caches with metadata bound and CUDA mocks active."""
    tracker = _patch_cuda_for_load(monkeypatch)
    conn.bind_connector_metadata(metadata)
    conn.start_load_caches(encoder_cache=encoder_cache)
    return tracker


# 4 embeds * 4 features * 2 bytes (fp16) = 32 bytes per slot — keeps tests
# concise.
_NB = 32
_NE = 4


class TestStartLoadCachesCoalesce:
    """Worker-side load path: H2D coalescing, skip-cached, sort, view shape,
    record_stream, counter increments."""

    def test_three_contiguous_loads_coalesce_to_single_h2d(self, monkeypatch):
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="c", offset=2 * _NB, nbytes=_NB, num_embeds=_NE),
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        tracker = _start_load(monkeypatch, conn, metadata, cache)

        assert len(tracker.to_calls) == 1
        assert set(cache) == {"a", "b", "c"}
        assert conn._loads_issued == 3

    def test_non_contiguous_loads_remain_separate(self, monkeypatch):
        conn = _make_worker_connector()
        # Gap between 32 and 128 → two groups.
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=128, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="c", offset=128 + _NB, nbytes=_NB, num_embeds=_NE),
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        tracker = _start_load(monkeypatch, conn, metadata, cache)

        assert len(tracker.to_calls) == 2
        assert set(cache) == {"a", "b", "c"}
        assert conn._loads_issued == 3

    def test_unsorted_loads_sorted_before_grouping(self, monkeypatch):
        # Cmds emitted in non-offset order should still coalesce.
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="c", offset=2 * _NB, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE),
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        tracker = _start_load(monkeypatch, conn, metadata, cache)

        assert len(tracker.to_calls) == 1
        assert set(cache) == {"a", "b", "c"}

    def test_already_cached_loads_are_skipped(self, monkeypatch):
        conn = _make_worker_connector()
        # 'b' already in cache → only 'a' and 'c' fire H2Ds, and they're
        # non-contiguous in the *pending* set so two groups.
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="c", offset=2 * _NB, nbytes=_NB, num_embeds=_NE),
            ]
        )
        prior = torch.zeros((_NE, 4), dtype=torch.float16)
        cache: dict[str, torch.Tensor] = {"b": prior}
        tracker = _start_load(monkeypatch, conn, metadata, cache)

        assert len(tracker.to_calls) == 2
        assert cache["b"] is prior  # untouched
        assert conn._loads_issued == 2  # only 'a' and 'c'
        # record_stream fires once per pending view — 'b' was skipped.
        assert len(tracker.record_stream_calls) == 2

    def test_view_shape_dtype_match_metadata(self, monkeypatch):
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE),
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        _start_load(monkeypatch, conn, metadata, cache)

        for h in ("a", "b"):
            t = cache[h]
            assert t.dtype == torch.float16
            assert t.shape == (_NE, conn._feature_dim)

    def test_view_content_reads_correct_arena_bytes(self, monkeypatch):
        # With Tensor.to patched to be identity, the encoder_cache views
        # observe the raw arena bytes — so we can check that each hash
        # maps to the right slice of the deterministic arena pattern.
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE),
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        _start_load(monkeypatch, conn, metadata, cache)

        for h, off in (("a", 0), ("b", _NB)):
            view_bytes = cache[h].contiguous().view(torch.uint8).reshape(-1)
            expected = conn._arena_uint8.narrow(0, off, _NB)
            assert torch.equal(view_bytes, expected)

    def test_record_stream_called_for_every_view(self, monkeypatch):
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="c", offset=128, nbytes=_NB, num_embeds=_NE),
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        tracker = _start_load(monkeypatch, conn, metadata, cache)
        assert len(tracker.record_stream_calls) == 3

    def test_empty_metadata_short_circuits(self, monkeypatch):
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata()
        cache: dict[str, torch.Tensor] = {}
        tracker = _start_load(monkeypatch, conn, metadata, cache)
        assert len(tracker.to_calls) == 0
        assert cache == {}
        assert conn._loads_issued == 0

    def test_single_load(self, monkeypatch):
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="solo", offset=0, nbytes=_NB, num_embeds=_NE),
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        tracker = _start_load(monkeypatch, conn, metadata, cache)
        assert len(tracker.to_calls) == 1
        assert len(tracker.record_stream_calls) == 1
        assert cache["solo"].shape == (_NE, conn._feature_dim)

    def test_all_loads_already_cached_zero_h2ds(self, monkeypatch):
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE),
            ]
        )
        prior_a = torch.zeros((_NE, 4), dtype=torch.float16)
        prior_b = torch.ones((_NE, 4), dtype=torch.float16)
        cache: dict[str, torch.Tensor] = {"a": prior_a, "b": prior_b}
        tracker = _start_load(monkeypatch, conn, metadata, cache)
        assert len(tracker.to_calls) == 0
        assert cache["a"] is prior_a
        assert cache["b"] is prior_b
        assert conn._loads_issued == 0

    def test_mixed_contiguous_and_gap_two_groups(self, monkeypatch):
        # Two contiguous groups separated by a gap → 2 H2Ds, 5 views.
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(
                    mm_hash=f"g0_{i}", offset=i * _NB, nbytes=_NB, num_embeds=_NE
                )
                for i in range(3)
            ]
            + [
                mod._LoadCmd(
                    mm_hash=f"g1_{i}",
                    offset=512 + i * _NB,
                    nbytes=_NB,
                    num_embeds=_NE,
                )
                for i in range(2)
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        tracker = _start_load(monkeypatch, conn, metadata, cache)
        assert len(tracker.to_calls) == 2
        assert len(tracker.record_stream_calls) == 5
        assert conn._loads_issued == 5

    def test_skip_middle_breaks_contiguity(self, monkeypatch):
        # 4 originally-contiguous offsets; 'b' (the second) is already cached.
        # Pending = a, c, d. After sort: a [0..32], c [64..96], d [96..128].
        # That's two groups: {a} and {c, d}. → 2 H2Ds.
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="c", offset=2 * _NB, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="d", offset=3 * _NB, nbytes=_NB, num_embeds=_NE),
            ]
        )
        prior = torch.zeros((_NE, 4), dtype=torch.float16)
        cache: dict[str, torch.Tensor] = {"b": prior}
        tracker = _start_load(monkeypatch, conn, metadata, cache)
        assert len(tracker.to_calls) == 2
        assert conn._loads_issued == 3

    def test_views_in_same_group_share_storage(self, monkeypatch):
        # All views from one coalesced H2D must share an underlying Storage —
        # this is the lifetime contract that keeps the packed buffer alive
        # while any view is still consumed by compute.
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="c", offset=2 * _NB, nbytes=_NB, num_embeds=_NE),
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        _start_load(monkeypatch, conn, metadata, cache)
        ptrs = {h: cache[h].untyped_storage().data_ptr() for h in ("a", "b", "c")}
        assert ptrs["a"] == ptrs["b"] == ptrs["c"]

    def test_views_in_different_groups_have_distinct_storage(self, monkeypatch):
        # Two groups separated by a gap → two distinct underlying Storages
        # for the GPU side. (With Tensor.to patched to identity, the views
        # actually share the arena's storage — so we instead verify the
        # storage offset differs by exactly the gap, which is the same
        # invariant in disguise.)
        conn = _make_worker_connector()
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            loads=[
                mod._LoadCmd(mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE),
                mod._LoadCmd(mm_hash="b", offset=128, nbytes=_NB, num_embeds=_NE),
            ]
        )
        cache: dict[str, torch.Tensor] = {}
        _start_load(monkeypatch, conn, metadata, cache)
        # Each view's element address is its arena offset under identity .to().
        offset_a = cache["a"].data_ptr() - conn._arena_uint8.data_ptr()
        offset_b = cache["b"].data_ptr() - conn._arena_uint8.data_ptr()
        assert offset_a == 0
        assert offset_b == 128


def _patch_save_path(monkeypatch) -> dict:
    """Stub the CUDA-only side of save_caches so it can run on CPU. Returns a
    dict with the captured stream / event mocks for assertions."""
    captured: dict = {}
    fake_save_event = MagicMock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(torch.cuda, "stream", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(torch.cuda, "Event", lambda: fake_save_event)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda self, _s: None, raising=False
    )
    captured["event"] = fake_save_event
    return captured


class TestSaveCachesEndToEnd:
    """End-to-end save_caches behavior: writer-rank D2H, counter, fence flag,
    error paths."""

    def test_writer_rank_save_increments_counter_and_flag(self, monkeypatch):
        conn = _make_worker_connector(tp_rank=0, fence_active=True)
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=0
                ),
            ]
        )
        conn.bind_connector_metadata(metadata)
        encoder_cache = {
            "a": torch.zeros((_NE, conn._feature_dim), dtype=torch.float16)
        }
        _patch_save_path(monkeypatch)

        conn.save_caches(encoder_cache=encoder_cache, mm_hash="a")

        assert conn._saves_issued == 1
        assert conn._save_pending_this_step is True

    def test_wrong_rank_save_is_no_op(self, monkeypatch):
        # Owner is rank 1; we are rank 0. Even with the dict unfiltered,
        # save_caches must early-return without touching state.
        conn = _make_worker_connector(tp_rank=0)
        conn._worker_initialized = False  # forces unfiltered dict
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=1
                ),
            ]
        )
        conn.bind_connector_metadata(metadata)
        # Now flip _worker_initialized back so save_caches doesn't trigger
        # _setup_worker (which would touch shm).
        conn._worker_initialized = True
        encoder_cache = {
            "a": torch.zeros((_NE, conn._feature_dim), dtype=torch.float16)
        }
        _patch_save_path(monkeypatch)

        conn.save_caches(encoder_cache=encoder_cache, mm_hash="a")

        assert conn._saves_issued == 0
        assert conn._save_pending_this_step is False

    def test_size_mismatch_raises(self, monkeypatch):
        conn = _make_worker_connector(tp_rank=0)
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=0
                ),
            ]
        )
        conn.bind_connector_metadata(metadata)
        # Wrong shape — totals to 64 bytes (8 elements * 8 bytes for fp32),
        # but cmd.nbytes is 32.
        encoder_cache = {
            "a": torch.zeros((_NE, conn._feature_dim), dtype=torch.float32),
        }
        _patch_save_path(monkeypatch)

        with pytest.raises(RuntimeError, match="size mismatch"):
            conn.save_caches(encoder_cache=encoder_cache, mm_hash="a")

    def test_missing_from_encoder_cache_warns_and_returns(self, monkeypatch):
        conn = _make_worker_connector(tp_rank=0)
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=0
                ),
            ]
        )
        conn.bind_connector_metadata(metadata)
        _patch_save_path(monkeypatch)

        with patch.object(mod.logger, "warning") as warn:
            conn.save_caches(encoder_cache={}, mm_hash="a")

        warn.assert_called_once()
        assert conn._saves_issued == 0
        assert conn._save_pending_this_step is False

    def test_three_saves_one_event(self, monkeypatch):
        # Plan B fence: even if save_caches fires N times, clear emits ONE
        # event. _save_pending_this_step is the only flag mutated in
        # save_caches; the event lives in clear.
        conn = _make_worker_connector(tp_rank=0, fence_active=True)
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash=h, offset=i * _NB, nbytes=_NB, num_embeds=_NE, writer_rank=0
                )
                for i, h in enumerate(("a", "b", "c"))
            ]
        )
        conn.bind_connector_metadata(metadata)
        encoder_cache = {
            h: torch.zeros((_NE, conn._feature_dim), dtype=torch.float16)
            for h in ("a", "b", "c")
        }
        captured = _patch_save_path(monkeypatch)

        for h in ("a", "b", "c"):
            conn.save_caches(encoder_cache=encoder_cache, mm_hash=h)

        # No events created during save_caches itself.
        captured["event"].record.assert_not_called()
        assert conn._save_pending_this_step is True

        # The single event is created in clear_connector_metadata.
        with patch.object(mod.ECConnectorBase, "clear_connector_metadata"):
            conn.clear_connector_metadata()
        # Exactly one record on save_stream at clear time.
        assert captured["event"].record.call_count == 1

    def test_bind_clear_bind_cycle_resets_dict(self, monkeypatch):
        conn = _make_worker_connector(tp_rank=0)
        meta1 = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=0
                ),
            ]
        )
        meta2 = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="x", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=0
                ),
            ]
        )
        conn.bind_connector_metadata(meta1)
        assert set(conn._save_cmd_by_hash) == {"a"}

        with patch.object(mod.ECConnectorBase, "clear_connector_metadata"):
            conn.clear_connector_metadata()
        assert conn._save_cmd_by_hash == {}

        conn.bind_connector_metadata(meta2)
        assert set(conn._save_cmd_by_hash) == {"x"}


class TestSaveCachesO1Lookup:
    """save_caches uses the dict built in bind_connector_metadata."""

    def test_dict_filtered_by_writer_rank_after_setup(self):
        conn = _make_worker_connector(tp_rank=0)
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=0
                ),
                mod._SaveCmd(
                    mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE, writer_rank=1
                ),
            ]
        )
        conn.bind_connector_metadata(metadata)
        assert set(conn._save_cmd_by_hash) == {"a"}

    def test_dict_unfiltered_before_setup(self):
        conn = _make_worker_connector(tp_rank=0)
        conn._worker_initialized = False  # simulate first-step path
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=0
                ),
                mod._SaveCmd(
                    mm_hash="b", offset=_NB, nbytes=_NB, num_embeds=_NE, writer_rank=1
                ),
            ]
        )
        conn.bind_connector_metadata(metadata)
        # Both present; writer_rank gate in save_caches still enforces correctness.
        assert set(conn._save_cmd_by_hash) == {"a", "b"}

    def test_non_our_metadata_type_resets_dict(self):
        conn = _make_worker_connector()
        conn._save_cmd_by_hash = {"stale": MagicMock()}  # type: ignore[dict-item]
        conn.bind_connector_metadata(MagicMock())
        assert conn._save_cmd_by_hash == {}

    def test_save_caches_returns_silently_on_unknown_hash(self):
        # Non-writer ranks see an empty dict; save_caches must early-return
        # without touching state.
        conn = _make_worker_connector(tp_rank=1)
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=0
                ),
            ]
        )
        conn.bind_connector_metadata(metadata)
        assert conn._save_cmd_by_hash == {}  # filtered to rank 1, none match

        # Calling save_caches for "a" should be a no-op — counters unchanged,
        # no event flag set.
        conn._saves_issued = 0
        conn._save_pending_this_step = False
        conn.save_caches(encoder_cache={}, mm_hash="a")
        assert conn._saves_issued == 0
        assert conn._save_pending_this_step is False


class TestPlanBSingleEvent:
    """Plan B fence: one event per step, recorded in clear_connector_metadata."""

    def test_clear_records_event_when_save_pending(self):
        conn = _make_worker_connector(fence_active=True)
        conn._save_pending_this_step = True

        fake_stream = MagicMock()
        fake_event = MagicMock()
        with (
            patch.object(torch.cuda, "Event", return_value=fake_event),
            patch.object(torch.cuda, "current_stream", return_value=fake_stream),
            patch.object(mod.ECConnectorBase, "clear_connector_metadata"),
        ):
            conn.clear_connector_metadata()

        fake_event.record.assert_called_once_with(conn._save_stream)
        fake_stream.wait_event.assert_called_once_with(fake_event)
        assert conn._save_pending_this_step is False
        assert conn._save_cmd_by_hash == {}

    def test_clear_no_op_when_no_save_pending(self):
        conn = _make_worker_connector(fence_active=True)
        conn._save_pending_this_step = False

        fake_stream = MagicMock()
        with (
            patch.object(torch.cuda, "Event") as event_cls,
            patch.object(torch.cuda, "current_stream", return_value=fake_stream),
            patch.object(mod.ECConnectorBase, "clear_connector_metadata"),
        ):
            conn.clear_connector_metadata()

        event_cls.assert_not_called()
        fake_stream.wait_event.assert_not_called()

    def test_plan_a_never_records_event(self, monkeypatch):
        # With the fence inactive, save_caches must NOT set the pending flag,
        # so clear_connector_metadata stays a no-op even if a save fired.
        conn = _make_worker_connector(fence_active=False, tp_rank=0)
        metadata = mod.MultimodalEmbeddingCacheConnectorMetadata(
            saves=[
                mod._SaveCmd(
                    mm_hash="a", offset=0, nbytes=_NB, num_embeds=_NE, writer_rank=0
                ),
            ]
        )
        conn.bind_connector_metadata(metadata)
        src = torch.zeros((_NE, conn._feature_dim), dtype=torch.float16)
        encoder_cache = {"a": src}
        # Bypass real CUDA: stub current_stream + stream + record_stream so
        # the save path runs to completion on CPU.
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(torch.cuda, "stream", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(
            torch.Tensor, "record_stream", lambda self, _s: None, raising=False
        )

        conn.save_caches(encoder_cache=encoder_cache, mm_hash="a")

        assert conn._save_pending_this_step is False
        # Plan A still must increment the per-rank save counter.
        assert conn._saves_issued == 1
