# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV capacity derived by the GMS scratch-KV worker for a writer and an importer.

A GMS weight *writer* and a GMS weight *importer* have genuinely different
measured memory footprints: the writer's GMS pool is already inside
``torch.cuda.max_memory_allocated()``, while the importer's mapped bytes are
not and are added back through ``get_imported_weights_bytes()``. When the
operator sizes the KV cache explicitly, the two engines must still derive the
same capacity, otherwise they build different KV layouts and the standby cannot
adopt the layout the active engine persisted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

from gpu_memory_service.integrations.vllm import worker as gms_worker
from vllm.config import CUDAGraphMode

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]

# 64 GiB, the value both engines were given in the report.
EXPLICIT_KV_CACHE_MEMORY_BYTES = 68719476736

REQUESTED_MEMORY = 72 * (1 << 30)
IMPORTED_WEIGHTS_BYTES = 16 * (1 << 30)
WRITER_TORCH_PEAK = 20 * (1 << 30)
# Writer and importer footprints do not cancel exactly; this residue is the
# block-count gap the two measured capacities produced.
FOOTPRINT_RESIDUE = 13 * 32 * 1024
IMPORTER_TORCH_PEAK = WRITER_TORCH_PEAK - IMPORTED_WEIGHTS_BYTES - FOOTPRINT_RESIDUE


class _FakeTorchDevice:
    """The few ``torch.cuda`` calls the measured scratch-KV path makes."""

    def __init__(self, torch_peak: int):
        self._torch_peak = torch_peak

    def reset_peak_memory_stats(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def max_memory_allocated(self) -> int:
        return self._torch_peak


def _available_memory_for_role(
    monkeypatch, *, importer: bool, kv_cache_memory_bytes: int | None
) -> int:
    """Run the scratch-KV override as one role and return the capacity it derives.

    The worker is built without ``__init__`` — only the attributes the override
    and vLLM's own explicit-capacity branch actually read are supplied.
    ``model_config.multimodal_config`` and ``parallel_config`` are both needed
    because vLLM reserves multimodal front-end memory through a bound method on
    0.26 and a module-level function taking the parallel config on 0.28.
    ``init_snapshot.free_memory`` is read by that branch's log message.
    """
    worker = gms_worker.GMSWorker.__new__(gms_worker.GMSWorker)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=kv_cache_memory_bytes)
    worker.model_config = SimpleNamespace(multimodal_config=None)
    worker.parallel_config = SimpleNamespace(_api_process_count=1)
    worker.init_snapshot = SimpleNamespace(free_memory=REQUESTED_MEMORY)
    worker.model_runner = SimpleNamespace(profile_run=lambda: None)
    worker.requested_memory = REQUESTED_MEMORY
    # CUDAGraphMode.NONE keeps the cudagraph estimate at zero, so the measured
    # arithmetic below stays readable.
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    )

    torch_peak = IMPORTER_TORCH_PEAK if importer else WRITER_TORCH_PEAK
    monkeypatch.setattr(gms_worker, "is_scratch_kv_enabled", lambda: True)
    monkeypatch.setattr(gms_worker, "has_pending_gms_write", lambda: not importer)
    monkeypatch.setattr(
        gms_worker, "get_imported_weights_bytes", lambda: IMPORTED_WEIGHTS_BYTES
    )
    monkeypatch.setattr(
        gms_worker, "torch_device", lambda: _FakeTorchDevice(torch_peak)
    )

    return worker._determine_available_memory_before_gms_publish()


def test_explicit_kv_cache_memory_bytes_is_identical_for_writer_and_importer(
    monkeypatch,
):
    """An explicitly sized KV cache is a constant, not a measurement."""
    writer = _available_memory_for_role(
        monkeypatch,
        importer=False,
        kv_cache_memory_bytes=EXPLICIT_KV_CACHE_MEMORY_BYTES,
    )
    importer = _available_memory_for_role(
        monkeypatch, importer=True, kv_cache_memory_bytes=EXPLICIT_KV_CACHE_MEMORY_BYTES
    )

    assert writer == EXPLICIT_KV_CACHE_MEMORY_BYTES
    assert importer == EXPLICIT_KV_CACHE_MEMORY_BYTES
    # Equal capacity is what makes the two engines derive the same block count
    # and therefore the same persisted KV layout.
    assert writer == importer


def test_measured_path_is_unchanged_without_an_explicit_value(monkeypatch):
    """Without an explicit value the measured accounting must behave as before."""
    writer = _available_memory_for_role(
        monkeypatch, importer=False, kv_cache_memory_bytes=None
    )
    importer = _available_memory_for_role(
        monkeypatch, importer=True, kv_cache_memory_bytes=None
    )

    # The writer's GMS pool is already in the torch peak; the importer's mapped
    # weights are added on top of a peak that does not contain them.
    assert writer == REQUESTED_MEMORY - WRITER_TORCH_PEAK
    assert importer == REQUESTED_MEMORY - IMPORTER_TORCH_PEAK - IMPORTED_WEIGHTS_BYTES
    assert writer != importer
