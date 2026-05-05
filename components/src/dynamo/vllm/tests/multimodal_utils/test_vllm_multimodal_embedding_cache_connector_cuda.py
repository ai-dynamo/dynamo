# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-side unit tests for DynamoMultimodalEmbeddingCacheConnector.

These tests exercise the worker-side async-save lifecycle and the
pre-allocated pinned arena: DtoH on a dedicated stream into arena
slices, HtoD on compute with record_stream protection, deferred-free
queue gated on save+load events, and the fragmentation→compaction
fallback.

Markers: gpu_1 (single CUDA device required). The CPU-only scheduler-side
tests live in test_vllm_multimodal_embedding_cache_connector.py.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from dynamo.vllm.multimodal_utils import multimodal_embedding_cache_connector as mod

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.multimodal,
]


HIDDEN = 64
DTYPE = torch.float16
DTYPE_BYTES = 2
BYTES_PER_EMBED = HIDDEN * DTYPE_BYTES  # 128


def _capacity_gb_for_chunks(num_chunks: int) -> float:
    return num_chunks * BYTES_PER_EMBED / (1024**3)


def _make_vllm_config(capacity_gb: float) -> MagicMock:
    config = MagicMock()
    config.ec_transfer_config.ec_connector_extra_config = {
        "multimodal_embedding_cache_capacity_gb": capacity_gb,
    }
    config.model_config.get_hidden_size.return_value = HIDDEN
    config.model_config.dtype = DTYPE
    return config


def _make_connector(
    capacity_gb: float = 0.001,
) -> mod.DynamoMultimodalEmbeddingCacheConnector:
    with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
        return mod.DynamoMultimodalEmbeddingCacheConnector(
            vllm_config=_make_vllm_config(capacity_gb=capacity_gb),
            role=MagicMock(),
        )


def _set_metadata(
    conn: mod.DynamoMultimodalEmbeddingCacheConnector,
    *,
    loads: list[str] | None = None,
    saves: list[str] | None = None,
    evicts: list[str] | None = None,
) -> mod.MultimodalEmbeddingCacheConnectorMetadata:
    md = mod.MultimodalEmbeddingCacheConnectorMetadata(
        loads=loads or [],
        saves=saves or [],
        evicts=evicts or [],
    )
    conn._get_connector_metadata = lambda: md  # type: ignore[method-assign]
    return md


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda", torch.cuda.current_device())


class TestSaveLoadRoundTrip:
    """Save → load round-trips data correctly through the arena."""

    def test_round_trip(self, device):
        conn = _make_connector()
        src = torch.randn(8, HIDDEN, dtype=DTYPE, device=device)

        encoder_cache: dict[str, torch.Tensor] = {"h0": src}
        _set_metadata(conn, saves=["h0"])
        conn.save_caches(encoder_cache, "h0")

        encoder_cache.pop("h0")
        _set_metadata(conn, loads=["h0"])
        conn.start_load_caches(encoder_cache)
        torch.cuda.synchronize(device)

        loaded = encoder_cache["h0"]
        assert loaded.device == device
        assert torch.equal(loaded, src)

        entry = conn._cpu_store["h0"]
        assert entry.cpu_view.is_pinned()
        assert entry.offset == 0
        assert entry.nbytes == 8 * BYTES_PER_EMBED
        assert conn._used_bytes == 8 * BYTES_PER_EMBED


class TestArenaAllocator:
    """Direct unit tests against the byte-level allocator."""

    def test_alloc_first_fit_and_split(self):
        conn = _make_connector(_capacity_gb_for_chunks(8))
        conn._ensure_arena()
        # Free regions = [(0, 8*128)] = [(0, 1024)]
        off1 = conn._alloc(2 * BYTES_PER_EMBED)
        off2 = conn._alloc(3 * BYTES_PER_EMBED)
        assert off1 == 0
        assert off2 == 2 * BYTES_PER_EMBED
        assert conn._free_regions == [(5 * BYTES_PER_EMBED, 3 * BYTES_PER_EMBED)]
        assert conn._used_bytes == 5 * BYTES_PER_EMBED

    def test_alloc_returns_none_on_no_fit(self):
        conn = _make_connector(_capacity_gb_for_chunks(4))
        conn._ensure_arena()
        conn._alloc(3 * BYTES_PER_EMBED)
        # 1 chunk free; cannot fit 2 chunks contiguous.
        assert conn._alloc(2 * BYTES_PER_EMBED) is None

    def test_free_coalesces_right_neighbor(self):
        conn = _make_connector(_capacity_gb_for_chunks(8))
        conn._ensure_arena()
        a = conn._alloc(2 * BYTES_PER_EMBED)
        b = conn._alloc(2 * BYTES_PER_EMBED)
        # Free in reverse order: free B first, then A. A coalesces with B.
        conn._free(b, 2 * BYTES_PER_EMBED)
        conn._free(a, 2 * BYTES_PER_EMBED)
        # All 8 chunks should be one big free region again.
        assert conn._free_regions == [(0, 8 * BYTES_PER_EMBED)]

    def test_free_coalesces_left_neighbor(self):
        conn = _make_connector(_capacity_gb_for_chunks(8))
        conn._ensure_arena()
        a = conn._alloc(2 * BYTES_PER_EMBED)
        b = conn._alloc(2 * BYTES_PER_EMBED)
        # Free A first, then B. B coalesces with A.
        conn._free(a, 2 * BYTES_PER_EMBED)
        conn._free(b, 2 * BYTES_PER_EMBED)
        assert conn._free_regions == [(0, 8 * BYTES_PER_EMBED)]

    def test_free_coalesces_both_sides(self):
        conn = _make_connector(_capacity_gb_for_chunks(8))
        conn._ensure_arena()
        a = conn._alloc(2 * BYTES_PER_EMBED)
        b = conn._alloc(2 * BYTES_PER_EMBED)
        c = conn._alloc(2 * BYTES_PER_EMBED)
        # Free A and C first, leaving B in the middle; freeing B then
        # coalesces with both neighbors.
        conn._free(a, 2 * BYTES_PER_EMBED)
        conn._free(c, 2 * BYTES_PER_EMBED)
        # Live B occupies [256, 512); free regions surround it.
        # _free(c) coalesces with the trailing tail region.
        assert conn._free_regions == [
            (0, 2 * BYTES_PER_EMBED),
            (4 * BYTES_PER_EMBED, 4 * BYTES_PER_EMBED),
        ]
        conn._free(b, 2 * BYTES_PER_EMBED)
        # All three collapse into one.
        assert conn._free_regions == [(0, 8 * BYTES_PER_EMBED)]


class TestEvictReuseSlot:
    """After evict + reap, the freed slice is reusable by a new save."""

    def test_evict_then_save_reuses(self, device):
        conn = _make_connector()
        src_a = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)
        src_b = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)

        _set_metadata(conn, saves=["a"])
        conn.save_caches({"a": src_a}, "a")
        torch.cuda.synchronize(device)
        a_offset = conn._cpu_store["a"].offset

        _set_metadata(conn, evicts=["a"])
        conn.start_load_caches({})
        # Reap inside start_load_caches frees the slice.
        assert "a" not in conn._cpu_store
        assert conn._used_bytes == 0
        assert conn._pending_free == []

        _set_metadata(conn, saves=["b"])
        conn.save_caches({"b": src_b}, "b")
        # New save lands at the same offset (first-fit on a single free region).
        assert conn._cpu_store["b"].offset == a_offset


class TestRetiredReaping:
    """Eviction never blocks; deferred-free entries are released only when
    save+loads have drained. Mock event.query() so the test is deterministic."""

    @staticmethod
    def _stub_event(query_result: bool) -> MagicMock:
        event = MagicMock(spec=torch.cuda.Event)
        event.query.return_value = query_result
        return event

    def test_evict_does_not_free_pending_save(self, device):
        conn = _make_connector()
        src = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)

        _set_metadata(conn, saves=["h0"])
        conn.save_caches({"h0": src}, "h0")

        entry = conn._cpu_store["h0"]
        # Force save_done unresolved before eviction.
        entry.save_done = self._stub_event(False)
        nbytes = entry.nbytes

        _set_metadata(conn, evicts=["h0"])
        conn.start_load_caches({})

        # Entry moved from _cpu_store to _pending_free. Bytes are still
        # accounted because the slice has not been freed yet.
        assert "h0" not in conn._cpu_store
        assert len(conn._pending_free) == 1
        assert conn._used_bytes == nbytes

        # Resolve the event; reap drains the entry and clears bytes.
        conn._pending_free[0].save_done = self._stub_event(True)
        conn._reap_pending_free()
        assert conn._pending_free == []
        assert conn._used_bytes == 0

    def test_evict_does_not_free_pending_load(self, device):
        conn = _make_connector()
        src = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)

        _set_metadata(conn, saves=["h0"])
        conn.save_caches({"h0": src}, "h0")
        torch.cuda.synchronize(device)

        # Inject an unfinished pending load on the live entry.
        unresolved = self._stub_event(False)
        conn._cpu_store["h0"].pending_loads.append(unresolved)

        _set_metadata(conn, evicts=["h0"])
        conn.start_load_caches({})
        assert len(conn._pending_free) == 1

        # Reap is a no-op while the load is unresolved.
        conn._reap_pending_free()
        assert len(conn._pending_free) == 1

        unresolved.query.return_value = True
        conn._reap_pending_free()
        assert conn._pending_free == []
        assert conn._used_bytes == 0


class TestDuplicateSave:
    """Per ECConnector contract a hash is saved at most once per cache
    lifetime. The connector treats repeats as no-op."""

    def test_duplicate_save_is_noop(self, device):
        conn = _make_connector()
        src1 = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)
        src2 = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)

        _set_metadata(conn, saves=["h0"])
        conn.save_caches({"h0": src1}, "h0")
        torch.cuda.synchronize(device)

        first_offset = conn._cpu_store["h0"].offset
        first_view_ptr = conn._cpu_store["h0"].cpu_view.data_ptr()

        # Second save with the same hash: should not allocate, not change view.
        _set_metadata(conn, saves=["h0"])
        conn.save_caches({"h0": src2}, "h0")

        assert conn._cpu_store["h0"].offset == first_offset
        assert conn._cpu_store["h0"].cpu_view.data_ptr() == first_view_ptr
        assert conn._used_bytes == 4 * BYTES_PER_EMBED
        # Round-trip still returns the FIRST tensor.
        encoder_cache: dict[str, torch.Tensor] = {}
        _set_metadata(conn, loads=["h0"])
        conn.start_load_caches(encoder_cache)
        torch.cuda.synchronize(device)
        assert torch.equal(encoder_cache["h0"], src1)


class TestRecordStream:
    """Verify the GPU source has record_stream invoked with the save stream."""

    def test_save_records_src_on_save_stream(self, device):
        conn = _make_connector()
        src = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)

        record_calls: list[torch.cuda.Stream] = []
        original = src.record_stream

        def capture(stream):
            record_calls.append(stream)
            return original(stream)

        src.record_stream = capture  # type: ignore[method-assign]

        _set_metadata(conn, saves=["h0"])
        conn.save_caches({"h0": src}, "h0")

        assert len(record_calls) == 1
        assert record_calls[0] is conn._save_stream


class TestRoundTripUnderConsumerPop:
    """Functional regression for record_stream protection.

    Source-side: while async DtoH is in flight, vLLM pops encoder_cache
    and drops its reference. src.record_stream(save_stream) keeps the
    GPU source alive until save_done resolves.

    Destination-side: while compute is mid-kernel, vLLM pops
    encoder_cache. gpu_tensor.record_stream(compute) keeps the
    destination storage alive.
    """

    def test_save_pop_then_sync_content_correct(self, device):
        conn = _make_connector()
        original = torch.randn(64, HIDDEN, dtype=DTYPE, device=device)
        src = original.clone()

        encoder_cache: dict[str, torch.Tensor] = {"h0": src}
        _set_metadata(conn, saves=["h0"])
        conn.save_caches(encoder_cache, "h0")

        del encoder_cache["h0"]
        del src

        for _ in range(8):
            _ = torch.randn(64, HIDDEN, dtype=DTYPE, device=device)

        torch.cuda.synchronize(device)
        cpu_view = conn._cpu_store["h0"].cpu_view
        assert torch.equal(cpu_view, original.cpu())

    def test_load_pop_then_compute_consumer_correct(self, device):
        conn = _make_connector()
        original = torch.randn(64, HIDDEN, dtype=DTYPE, device=device)

        encoder_cache: dict[str, torch.Tensor] = {"h0": original.clone()}
        _set_metadata(conn, saves=["h0"])
        conn.save_caches(encoder_cache, "h0")
        encoder_cache.pop("h0")
        torch.cuda.synchronize(device)

        _set_metadata(conn, loads=["h0"])
        conn.start_load_caches(encoder_cache)

        gpu_tensor = encoder_cache["h0"]
        result = gpu_tensor.float() @ gpu_tensor.float().t()

        del encoder_cache["h0"]
        del gpu_tensor

        for _ in range(8):
            _ = torch.randn(64, HIDDEN, dtype=DTYPE, device=device)

        torch.cuda.synchronize(device)
        expected = original.float() @ original.float().t()
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


class TestMixedHitsAndMisses:
    def test_mixed_step(self, device):
        conn = _make_connector()
        src_a = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)
        src_b = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)

        _set_metadata(conn, saves=["a"])
        conn.save_caches({"a": src_a}, "a")
        torch.cuda.synchronize(device)

        _set_metadata(conn, saves=["b"])
        conn.save_caches({"b": src_b}, "b")

        encoder_cache: dict[str, torch.Tensor] = {}
        _set_metadata(conn, loads=["a"])
        conn.start_load_caches(encoder_cache)
        torch.cuda.synchronize(device)

        assert torch.equal(encoder_cache["a"], src_a)
        assert "a" in conn._cpu_store
        assert "b" in conn._cpu_store


class TestFragmentationCompaction:
    """Force the LRU-promotion fragmentation pattern that motivated R1
    Issue 1, then confirm the compaction fallback recovers."""

    def test_compaction_with_left_shift_overlap(self, device):
        # 8-chunk arena (1024 bytes). Layout after the 3 saves:
        #   [A: 0..256] [B: 256..768 (4 chunks)] [C: 768..1024]
        # Evict A and C. Free regions = [(0, 256), (768, 256)].
        # Save D (4 chunks = 512 bytes). _alloc fails (max contiguous = 256).
        # Compaction shifts B from offset 256 to 0 — left-shift by 256 with
        # entry size 512 → 256 bytes of source/destination overlap, which
        # only ctypes.memmove handles correctly.
        conn = _make_connector(_capacity_gb_for_chunks(8))

        src_a = torch.randn(2, HIDDEN, dtype=DTYPE, device=device)
        src_b = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)
        src_c = torch.randn(2, HIDDEN, dtype=DTYPE, device=device)

        for h, src in [("a", src_a), ("b", src_b), ("c", src_c)]:
            _set_metadata(conn, saves=[h])
            conn.save_caches({h: src}, h)
        torch.cuda.synchronize(device)

        assert conn._cpu_store["a"].offset == 0
        assert conn._cpu_store["b"].offset == 2 * BYTES_PER_EMBED
        assert conn._cpu_store["c"].offset == 6 * BYTES_PER_EMBED
        assert conn._used_bytes == 8 * BYTES_PER_EMBED
        assert conn._free_regions == []

        # Evict A and C; reap (events resolved post-sync).
        _set_metadata(conn, evicts=["a", "c"])
        conn.start_load_caches({})
        assert "a" not in conn._cpu_store
        assert "c" not in conn._cpu_store
        # Two non-contiguous free regions; no single one fits 4 chunks.
        assert conn._free_regions == [
            (0, 2 * BYTES_PER_EMBED),
            (6 * BYTES_PER_EMBED, 2 * BYTES_PER_EMBED),
        ]

        src_d = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)
        _set_metadata(conn, saves=["d"])
        conn.save_caches({"d": src_d}, "d")
        torch.cuda.synchronize(device)

        # B was repacked to 0 (left-shift with overlap), D landed in the tail.
        assert conn._cpu_store["b"].offset == 0
        assert conn._cpu_store["d"].offset == 4 * BYTES_PER_EMBED

        # Round-trip both through the post-compaction arena.
        encoder_cache: dict[str, torch.Tensor] = {}
        _set_metadata(conn, loads=["b", "d"])
        conn.start_load_caches(encoder_cache)
        torch.cuda.synchronize(device)
        assert torch.equal(encoder_cache["b"], src_b)
        assert torch.equal(encoder_cache["d"], src_d)

    def test_compaction_when_dict_order_differs_from_offset_order(self, device):
        # Insert entries into _cpu_store in the OPPOSITE order from their
        # offsets, then force a compact() and confirm the offset-sorted
        # iteration order means data lands correctly.
        conn = _make_connector(_capacity_gb_for_chunks(8))
        conn._ensure_arena()
        # Pretend the save_stream exists too (compact() asserts it).
        conn._save_stream = MagicMock(spec=torch.cuda.Stream)

        # Synthesize 3 entries with offsets out of insertion order:
        #   _cpu_store insertion order: x@offset 4, y@offset 0, z@offset 6
        arena = conn._pinned_arena
        assert arena is not None

        # Write distinct payloads at each entry's current offset.
        def _payload(start: int, n_chunks: int) -> bytes:
            return bytes((start + i) % 256 for i in range(n_chunks * BYTES_PER_EMBED))

        # x: 2 chunks at offset 4*128=512
        x_payload = _payload(0x10, 2)
        arena[4 * BYTES_PER_EMBED : 6 * BYTES_PER_EMBED] = torch.frombuffer(
            bytearray(x_payload), dtype=torch.uint8
        )
        # y: 2 chunks at offset 0
        y_payload = _payload(0x20, 2)
        arena[0 : 2 * BYTES_PER_EMBED] = torch.frombuffer(
            bytearray(y_payload), dtype=torch.uint8
        )
        # z: 2 chunks at offset 6*128=768
        z_payload = _payload(0x30, 2)
        arena[6 * BYTES_PER_EMBED : 8 * BYTES_PER_EMBED] = torch.frombuffer(
            bytearray(z_payload), dtype=torch.uint8
        )

        def _make_entry(offset: int, n_chunks: int) -> mod._CpuEntry:
            nbytes = n_chunks * BYTES_PER_EMBED
            view = arena.narrow(0, offset, nbytes).view(DTYPE).view(n_chunks, HIDDEN)
            return mod._CpuEntry(
                cpu_view=view,
                offset=offset,
                nbytes=nbytes,
                save_done=MagicMock(spec=torch.cuda.Event),
            )

        conn._cpu_store["x"] = _make_entry(4 * BYTES_PER_EMBED, 2)
        conn._cpu_store["y"] = _make_entry(0, 2)
        conn._cpu_store["z"] = _make_entry(6 * BYTES_PER_EMBED, 2)
        conn._used_bytes = 6 * BYTES_PER_EMBED
        # Emulate a fragmented arena.
        conn._free_regions = [(2 * BYTES_PER_EMBED, 2 * BYTES_PER_EMBED)]

        conn._compact()

        # After compaction, entries occupy [0, 2c), [2c, 4c), [4c, 6c) sorted
        # by their original offsets — y (was 0) → 0, x (was 4) → 2*BPE,
        # z (was 6) → 4*BPE.
        assert conn._cpu_store["y"].offset == 0
        assert conn._cpu_store["x"].offset == 2 * BYTES_PER_EMBED
        assert conn._cpu_store["z"].offset == 4 * BYTES_PER_EMBED
        assert conn._free_regions == [(6 * BYTES_PER_EMBED, 2 * BYTES_PER_EMBED)]

        # Bytes preserved per entry.
        assert bytes(arena[0 : 2 * BYTES_PER_EMBED].tolist()) == y_payload
        assert (
            bytes(arena[2 * BYTES_PER_EMBED : 4 * BYTES_PER_EMBED].tolist())
            == x_payload
        )
        assert (
            bytes(arena[4 * BYTES_PER_EMBED : 6 * BYTES_PER_EMBED].tolist())
            == z_payload
        )


class TestArenaExhausts:
    """If allocation fails after every fallback (blocking reap + compact),
    raise rather than silently drop. Drift between worker and scheduler
    state is a programming bug, not a normal cache miss path."""

    def test_arena_exhausted_raises(self, device):
        conn = _make_connector(_capacity_gb_for_chunks(2))
        src1 = torch.randn(2, HIDDEN, dtype=DTYPE, device=device)

        _set_metadata(conn, saves=["a"])
        conn.save_caches({"a": src1}, "a")

        # Arena full; no evictions issued; new save cannot fit.
        src2 = torch.randn(2, HIDDEN, dtype=DTYPE, device=device)
        _set_metadata(conn, saves=["b"])
        with pytest.raises(RuntimeError, match="EC arena exhausted"):
            conn.save_caches({"b": src2}, "b")


class TestPendingLoadsPrune:
    """Live entry's pending_loads list drops resolved events on the next load."""

    def test_resolved_load_events_pruned_on_next_load(self, device):
        conn = _make_connector()
        src = torch.randn(4, HIDDEN, dtype=DTYPE, device=device)

        _set_metadata(conn, saves=["h0"])
        conn.save_caches({"h0": src}, "h0")
        torch.cuda.synchronize(device)

        # First load → adds one event.
        _set_metadata(conn, loads=["h0"])
        conn.start_load_caches({})
        torch.cuda.synchronize(device)
        assert len(conn._cpu_store["h0"].pending_loads) == 1

        # Second load: previous event is resolved post-sync, so it should be
        # pruned and only the new event remains.
        _set_metadata(conn, loads=["h0"])
        conn.start_load_caches({})
        assert len(conn._cpu_store["h0"].pending_loads) == 1


class TestSingleDeviceLock:
    def test_second_device_assertion(self, device):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires ≥2 CUDA devices")

        conn = _make_connector()
        src0 = torch.randn(4, HIDDEN, dtype=DTYPE, device=torch.device("cuda", 0))
        _set_metadata(conn, saves=["h0"])
        conn.save_caches({"h0": src0}, "h0")

        src1 = torch.randn(4, HIDDEN, dtype=DTYPE, device=torch.device("cuda", 1))
        _set_metadata(conn, saves=["h1"])
        with pytest.raises(AssertionError, match="bound to device"):
            conn.save_caches({"h1": src1}, "h1")
