# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import weakref

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _fresh_module(monkeypatch, policy: str | None):
    # Reload with the policy disabled: the module-tail autostart firing
    # during import/reload would rebuild the module globals and leak the
    # previously started daemon thread plus the disabled gen2 threshold
    # when a freeze test runs in isolation. Tests start the policy
    # explicitly after the reload.
    monkeypatch.delenv("DYN_FPM_GC_POLICY", raising=False)
    import dynamo.vllm.gc_policy as gc_policy

    gc_policy.stop_gc_policy()
    gc_policy = importlib.reload(gc_policy)
    if policy is not None:
        monkeypatch.setenv("DYN_FPM_GC_POLICY", policy)
    return gc_policy


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
        gc_policy.stop_gc_policy()
    assert gc.get_threshold() == thresholds, "stop must restore the thresholds"


def test_stop_is_idempotent_and_allows_restart(monkeypatch):
    monkeypatch.setenv("DYN_FPM_GC_FREEZE_INTERVAL_S", "3600")
    thresholds = gc.get_threshold()
    gc_policy = _fresh_module(monkeypatch, "freeze")
    try:
        assert gc_policy.start_gc_policy() is True
        gc_policy.stop_gc_policy()
        assert gc.get_threshold() == thresholds
        gc_policy.stop_gc_policy()  # no-op when already stopped
        assert gc.get_threshold() == thresholds
        assert gc_policy.start_gc_policy() is True, "restart after stop"
    finally:
        # An assertion failure above must not leak the daemon thread or the
        # disabled gen2 threshold into subsequent tests.
        gc_policy.stop_gc_policy()
    assert gc.get_threshold() == thresholds


def test_gc_maintain_freezes_objects(monkeypatch):
    gc_policy = _fresh_module(monkeypatch, None)
    gc.unfreeze()
    try:
        frozen = gc_policy.gc_maintain()
        assert frozen > 0
        assert frozen == gc.get_freeze_count()
    finally:
        gc.unfreeze()


def test_gc_maintain_reclaims_cycles_frozen_by_earlier_ticks(monkeypatch):
    gc_policy = _fresh_module(monkeypatch, None)

    class Node:
        pass

    gc.disable()
    try:
        node = Node()
        node.self_ref = node
        ref = weakref.ref(node)
        del node
        # Simulate a periodic tick freezing the still-uncollected cycle
        # into the permanent generation.
        gc.freeze()
        gc_policy.gc_maintain()
        assert ref() is None, "cycles frozen by a tick must still be reclaimed"
    finally:
        gc.enable()
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


def test_non_finite_interval_falls_back(monkeypatch):
    gc_policy = _fresh_module(monkeypatch, None)
    for raw in ("inf", "-inf", "nan"):
        monkeypatch.setenv("DYN_FPM_GC_FREEZE_INTERVAL_S", raw)
        assert gc_policy._interval_seconds() == 60.0
