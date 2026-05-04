# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-side unit tests for DynamoMultimodalEmbeddingCacheConnector.

These tests exercise the worker-side async-save / async-load lifecycle:
DtoH on a dedicated stream with pinned host buffers, HtoD on compute with
record_stream protection, retired-list reaping gated on save+load events,
and the pinned-budget cap fallback.

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


def _make_vllm_config(
    capacity_gb: float = 0.001,
    pin_memory: bool = True,
    pinned_overhead_pct: float = 25.0,
) -> MagicMock:
    config = MagicMock()
    config.ec_transfer_config.ec_connector_extra_config = {
        "multimodal_embedding_cache_capacity_gb": capacity_gb,
        "pin_memory": pin_memory,
        "pinned_overhead_pct": pinned_overhead_pct,
    }
    config.model_config.get_hidden_size.return_value = 64
    config.model_config.dtype = torch.float16
    return config


def _make_connector(
    capacity_gb: float = 0.001,
    pin_memory: bool = True,
    pinned_overhead_pct: float = 25.0,
) -> mod.DynamoMultimodalEmbeddingCacheConnector:
    with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
        return mod.DynamoMultimodalEmbeddingCacheConnector(
            vllm_config=_make_vllm_config(
                capacity_gb=capacity_gb,
                pin_memory=pin_memory,
                pinned_overhead_pct=pinned_overhead_pct,
            ),
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
    """Save → load → encoder_cache equality on the pinned and pageable paths."""

    def _save_then_load(
        self,
        conn: mod.DynamoMultimodalEmbeddingCacheConnector,
        device: torch.device,
        mm_hash: str,
        src: torch.Tensor,
    ) -> torch.Tensor:
        encoder_cache: dict[str, torch.Tensor] = {mm_hash: src}
        _set_metadata(conn, saves=[mm_hash])
        conn.save_caches(encoder_cache, mm_hash)

        encoder_cache.pop(mm_hash)
        _set_metadata(conn, loads=[mm_hash])
        conn.start_load_caches(encoder_cache)

        torch.cuda.synchronize(device)
        return encoder_cache[mm_hash]

    def test_pinned_path(self, device):
        conn = _make_connector(pin_memory=True)
        src = torch.randn(8, 64, dtype=torch.float16, device=device)
        loaded = self._save_then_load(conn, device, "h0", src)

        assert loaded.device == device
        assert torch.equal(loaded, src)
        assert conn._cpu_store["h0"].is_pinned is True
        assert conn._cpu_store["h0"].cpu_tensor.is_pinned()
        assert conn._pinned_bytes_active == src.numel() * src.element_size()

    def test_pageable_path(self, device):
        conn = _make_connector(pin_memory=False)
        src = torch.randn(8, 64, dtype=torch.float16, device=device)
        loaded = self._save_then_load(conn, device, "h0", src)

        assert torch.equal(loaded, src)
        assert conn._cpu_store["h0"].is_pinned is False
        assert not conn._cpu_store["h0"].cpu_tensor.is_pinned()
        # Pageable path is never accounted against the pinned budget.
        assert conn._pinned_bytes_active == 0


class TestRetiredReaping:
    """Eviction never blocks; retired entries are released only when save+loads
    have drained. We mock event.query so the test is deterministic regardless
    of how fast the device is."""

    def _stub_event(self, query_result: bool) -> MagicMock:
        event = MagicMock(spec=torch.cuda.Event)
        event.query.return_value = query_result
        return event

    def test_evict_does_not_free_pending_save(self, device):
        conn = _make_connector(pin_memory=True)
        src = torch.randn(4, 64, dtype=torch.float16, device=device)

        encoder_cache: dict[str, torch.Tensor] = {"h0": src}
        _set_metadata(conn, saves=["h0"])
        conn.save_caches(encoder_cache, "h0")

        # Force save_done to look unfinished, then evict.
        entry = conn._cpu_store["h0"]
        entry.save_done = self._stub_event(False)
        nbytes = entry.nbytes

        _set_metadata(conn, evicts=["h0"])
        conn.start_load_caches({})

        # Bytes moved active → retired but the entry is still on _retired
        # because its save event has not resolved.
        assert "h0" not in conn._cpu_store
        assert len(conn._retired) == 1
        assert conn._pinned_bytes_active == 0
        assert conn._pinned_bytes_retired == nbytes

        # Resolve the event; reap drains the entry and clears retired bytes.
        conn._retired[0].save_done = self._stub_event(True)
        conn._reap_retired()
        assert conn._retired == []
        assert conn._pinned_bytes_retired == 0

    def test_evict_does_not_free_pending_load(self, device):
        conn = _make_connector(pin_memory=True)
        src = torch.randn(4, 64, dtype=torch.float16, device=device)

        encoder_cache: dict[str, torch.Tensor] = {"h0": src}
        _set_metadata(conn, saves=["h0"])
        conn.save_caches(encoder_cache, "h0")

        # Inject an unfinished pending load; save_done resolved (real event).
        entry = conn._cpu_store["h0"]
        torch.cuda.synchronize(device)
        unresolved = self._stub_event(False)
        entry.pending_loads.append(unresolved)

        _set_metadata(conn, evicts=["h0"])
        conn.start_load_caches({})

        assert len(conn._retired) == 1
        assert conn._pinned_bytes_retired == entry.nbytes

        # Pending load remains; reap is a no-op.
        conn._reap_retired()
        assert len(conn._retired) == 1

        # Resolve the load; reap drains.
        unresolved.query.return_value = True
        conn._reap_retired()
        assert conn._retired == []
        assert conn._pinned_bytes_retired == 0


class TestRecordStream:
    """Verify that the GPU source (save) tensor has record_stream invoked with
    the save stream. The destination-side record_stream is exercised
    functionally by TestRoundTripUnderConsumerPop.

    Tensor instance attributes shadow methods in CPython attribute lookup, so
    we can spy by assigning a closure to src.record_stream directly.
    """

    def test_save_records_src_on_save_stream(self, device):
        conn = _make_connector(pin_memory=True)
        src = torch.randn(4, 64, dtype=torch.float16, device=device)

        record_calls: list[torch.cuda.Stream] = []
        original = src.record_stream

        def capture(stream):
            record_calls.append(stream)
            return original(stream)

        src.record_stream = capture  # type: ignore[method-assign]

        encoder_cache: dict[str, torch.Tensor] = {"h0": src}
        _set_metadata(conn, saves=["h0"])
        conn.save_caches(encoder_cache, "h0")

        assert len(record_calls) == 1
        assert record_calls[0] is conn._save_stream


class TestRoundTripUnderConsumerPop:
    """Functional regression for record_stream protection.

    Source-side: while the async DtoH is in flight, vLLM pops encoder_cache[h]
    and drops its reference. The caching allocator could reuse the GPU source
    storage immediately; src.record_stream(save_stream) is what holds it alive
    until save_done resolves. After sync, the CPU copy must still equal the
    original.

    Destination-side: while compute is mid-kernel using the loaded GPU
    tensor, vLLM pops encoder_cache[h]. gpu_tensor.record_stream(compute)
    keeps the destination storage alive until compute drains. The matmul
    result must still equal the reference.
    """

    def test_save_pop_then_sync_content_correct(self, device):
        conn = _make_connector(pin_memory=True)
        original = torch.randn(64, 64, dtype=torch.float16, device=device)
        src = original.clone()

        encoder_cache: dict[str, torch.Tensor] = {"h0": src}
        _set_metadata(conn, saves=["h0"])
        conn.save_caches(encoder_cache, "h0")

        del encoder_cache["h0"]
        del src

        # Allocator pressure on compute before we sync save_stream.
        for _ in range(8):
            _ = torch.randn(64, 64, dtype=torch.float16, device=device)

        torch.cuda.synchronize(device)
        cpu_buf = conn._cpu_store["h0"].cpu_tensor
        assert torch.equal(cpu_buf, original.cpu())

    def test_load_pop_then_compute_consumer_correct(self, device):
        conn = _make_connector(pin_memory=True)
        original = torch.randn(64, 64, dtype=torch.float16, device=device)

        encoder_cache: dict[str, torch.Tensor] = {"h0": original.clone()}
        _set_metadata(conn, saves=["h0"])
        conn.save_caches(encoder_cache, "h0")
        encoder_cache.pop("h0")
        torch.cuda.synchronize(device)

        _set_metadata(conn, loads=["h0"])
        conn.start_load_caches(encoder_cache)

        # Queue a compute consumer on the loaded tensor; this is enqueued on
        # the compute stream right after the HtoD. The result owns its own
        # storage but its computation still reads from gpu_tensor's storage.
        gpu_tensor = encoder_cache["h0"]
        result = gpu_tensor.float() @ gpu_tensor.float().t()

        # Drop both references to gpu_tensor's storage.
        del encoder_cache["h0"]
        del gpu_tensor

        # Allocator pressure before sync. Without record_stream(compute) on the
        # gpu_tensor, the allocator could reuse the storage while the matmul
        # is still reading from it.
        for _ in range(8):
            _ = torch.randn(64, 64, dtype=torch.float16, device=device)

        torch.cuda.synchronize(device)
        expected = original.float() @ original.float().t()
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


class TestMixedHitsAndMisses:
    def test_mixed_step(self, device):
        conn = _make_connector(pin_memory=True)
        # Pre-populate one entry to be loaded.
        src_a = torch.randn(4, 64, dtype=torch.float16, device=device)
        src_b = torch.randn(4, 64, dtype=torch.float16, device=device)

        _set_metadata(conn, saves=["a"])
        conn.save_caches({"a": src_a}, "a")
        torch.cuda.synchronize(device)

        # Now simulate one step: load "a" from CPU, save "b" newly computed.
        # Note: save and load happen via different metadata in real vLLM, so we
        # exercise them sequentially — first save (per-hash), then load.
        _set_metadata(conn, saves=["b"])
        conn.save_caches({"b": src_b}, "b")

        encoder_cache: dict[str, torch.Tensor] = {}
        _set_metadata(conn, loads=["a"])
        conn.start_load_caches(encoder_cache)
        torch.cuda.synchronize(device)

        assert torch.equal(encoder_cache["a"], src_a)
        assert "a" in conn._cpu_store
        assert "b" in conn._cpu_store


class TestPinnedCapFallback:
    """When the pinned budget would be exceeded the next save takes the
    pageable sync path. The breaching entry must be flagged is_pinned=False
    and must NOT be added to the pinned-bytes counter."""

    def test_over_cap_falls_back(self, device):
        # Use overhead=0 so the cap equals capacity_bytes exactly. Each save
        # is 4×64×2 = 512 bytes; cap is set to 512 bytes too. First save:
        # 0+0+512 > 512 is False → pinned. Second save: 512+0+512 > 512 is
        # True → pageable fallback.
        conn = _make_connector(capacity_gb=512 / (1024**3), pinned_overhead_pct=0.0)

        src1 = torch.randn(4, 64, dtype=torch.float16, device=device)
        src2 = torch.randn(4, 64, dtype=torch.float16, device=device)

        _set_metadata(conn, saves=["h1"])
        conn.save_caches({"h1": src1}, "h1")
        first_active = conn._pinned_bytes_active

        _set_metadata(conn, saves=["h2"])
        conn.save_caches({"h2": src2}, "h2")
        torch.cuda.synchronize(device)

        # First entry is pinned and counted; second is pageable and uncounted.
        assert conn._cpu_store["h1"].is_pinned is True
        assert conn._cpu_store["h2"].is_pinned is False
        assert conn._pinned_bytes_active == first_active
        assert torch.equal(conn._cpu_store["h2"].cpu_tensor, src2.cpu())


class TestSingleDeviceLock:
    def test_second_device_assertion(self, device):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires ≥2 CUDA devices")

        conn = _make_connector(pin_memory=True)
        src0 = torch.randn(4, 64, dtype=torch.float16, device=torch.device("cuda", 0))
        _set_metadata(conn, saves=["h0"])
        conn.save_caches({"h0": src0}, "h0")

        src1 = torch.randn(4, 64, dtype=torch.float16, device=torch.device("cuda", 1))
        _set_metadata(conn, saves=["h1"])
        with pytest.raises(AssertionError, match="bound to device"):
            conn.save_caches({"h1": src1}, "h1")
