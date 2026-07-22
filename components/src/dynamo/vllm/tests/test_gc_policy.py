# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib

import pytest

pytestmark = pytest.mark.pre_merge


def _fresh_module(monkeypatch, policy: str | None):
    if policy is None:
        monkeypatch.delenv("DYN_FPM_GC_POLICY", raising=False)
    else:
        monkeypatch.setenv("DYN_FPM_GC_POLICY", policy)
    import dynamo.vllm.gc_policy as gc_policy

    return importlib.reload(gc_policy)


def test_policy_off_by_default(monkeypatch):
    gc_policy = _fresh_module(monkeypatch, None)
    assert gc_policy.start_gc_policy() is False


def test_policy_starts_and_is_idempotent(monkeypatch):
    monkeypatch.setenv("DYN_FPM_GC_FREEZE_INTERVAL_S", "3600")
    thresholds = gc.get_threshold()
    gc_policy = _fresh_module(monkeypatch, "freeze")
    try:
        assert gc_policy.start_gc_policy() is True
        assert gc_policy.start_gc_policy() is True
        assert gc.get_threshold()[2] == 1 << 30, "auto gen2 must be disabled"
    finally:
        gc.set_threshold(*thresholds)
        gc.unfreeze()


def test_gc_maintain_freezes_objects(monkeypatch):
    gc_policy = _fresh_module(monkeypatch, None)
    gc.unfreeze()
    try:
        frozen = gc_policy.gc_maintain()
        assert frozen > 0
        assert frozen == gc.get_freeze_count()
    finally:
        gc.unfreeze()


def test_worker_extension_methods(monkeypatch):
    gc_policy = _fresh_module(monkeypatch, None)
    ext = gc_policy.FpmGcWorkerExtension()
    assert ext.fpm_gc_start() is False
    gc.unfreeze()
    try:
        assert ext.fpm_gc_maintain() > 0
    finally:
        gc.unfreeze()


def test_invalid_interval_falls_back(monkeypatch):
    monkeypatch.setenv("DYN_FPM_GC_FREEZE_INTERVAL_S", "not-a-number")
    gc_policy = _fresh_module(monkeypatch, None)
    assert gc_policy._interval_seconds() == 60.0


def test_eager_warmup_points_dedupe_and_flag():
    from types import SimpleNamespace

    from dynamo.vllm.instrumented_scheduler import (
        EAGER_WARMUP_REASON,
        BenchmarkPoint,
        InstrumentedScheduler,
    )

    grid = [
        BenchmarkPoint(point_type="decode", batch_size=513, total_kv_read_tokens=513,
                       expected_capture_size=None),
        BenchmarkPoint(point_type="decode", batch_size=513, total_kv_read_tokens=2048,
                       expected_capture_size=None),
        BenchmarkPoint(point_type="decode", batch_size=512, total_kv_read_tokens=512,
                       expected_capture_size=512),
        BenchmarkPoint(point_type="prefill", batch_size=1, total_prefill_tokens=1024,
                       total_kv_read_tokens=0, expected_capture_size=None),
        BenchmarkPoint(point_type="prefill", batch_size=2, total_prefill_tokens=1024,
                       total_kv_read_tokens=4096, expected_capture_size=None),
        BenchmarkPoint(point_type="prefill", batch_size=1, total_prefill_tokens=256,
                       total_kv_read_tokens=0, expected_capture_size=256),
    ]
    stub = SimpleNamespace(_bench_grid=grid)
    replicas = InstrumentedScheduler._bench_eager_warmup_points(stub)

    assert [(p.point_type, p.batch_size, p.total_prefill_tokens) for p in replicas] == [
        ("decode", 513, 0),
        ("prefill", 1, 1024),
    ]
    assert all(p.sample_reasons == [EAGER_WARMUP_REASON] for p in replicas)
    # originals untouched
    assert all(EAGER_WARMUP_REASON not in p.sample_reasons for p in grid)
