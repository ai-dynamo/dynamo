# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import json
import os
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


def _assert_gc_equivalent_to_never_benchmarked(gc_policy, thresholds):
    """The bar for both completion paths: a worker that ran the policy must
    be indistinguishable from one that never did."""
    assert gc.get_threshold() == thresholds
    assert gc.isenabled()
    assert gc_policy._started is False
    thread = gc_policy._freeze_thread
    assert thread is None or not thread.is_alive()

    class Node:
        pass

    node = Node()
    node.self_ref = node
    ref = weakref.ref(node)
    del node
    gc.collect()
    assert ref() is None, "cyclic garbage must be reclaimable again"


def test_worker_lifecycle_normal_completion_restores_gc(monkeypatch):
    """Normal completion path: the launcher's collective_rpc reaches
    ``FpmGcWorkerExtension.fpm_gc_stop`` in each worker; afterwards the
    worker must be GC-equivalent to one that never benchmarked, including
    reclamation of cycles frozen while the policy was active."""
    monkeypatch.setenv("DYN_FPM_GC_FREEZE_INTERVAL_S", "3600")
    thresholds = gc.get_threshold()
    gc_policy = _fresh_module(monkeypatch, "freeze")
    ext = gc_policy.FpmGcWorkerExtension()
    try:
        assert ext.fpm_gc_start() is True
        assert gc.get_threshold()[2] == 1 << 30

        class Node:
            pass

        node = Node()
        node.self_ref = node
        ref = weakref.ref(node)
        del node
        # A periodic tick freezes the uncollected cycle: with auto gen2
        # disabled it is now unreachable garbage that only stop() reclaims.
        with gc_policy._lock:
            gc.freeze()
        gc.collect()
        assert ref() is not None, "policy must pin the frozen cycle"

        ext.fpm_gc_stop()
        assert ref() is None, "stop must reclaim cycles frozen by ticks"
    finally:
        gc_policy.stop_gc_policy()
    _assert_gc_equivalent_to_never_benchmarked(gc_policy, thresholds)


def test_invalid_interval_falls_back(monkeypatch):
    monkeypatch.setenv("DYN_FPM_GC_FREEZE_INTERVAL_S", "not-a-number")
    gc_policy = _fresh_module(monkeypatch, None)
    assert gc_policy._interval_seconds() == 60.0


def test_non_finite_interval_falls_back(monkeypatch):
    gc_policy = _fresh_module(monkeypatch, None)
    for raw in ("inf", "-inf", "nan"):
        monkeypatch.setenv("DYN_FPM_GC_FREEZE_INTERVAL_S", raw)
        assert gc_policy._interval_seconds() == 60.0


def test_stop_reestablishes_vllm_serving_freeze(monkeypatch):
    """vLLM freezes the serving heap after startup/warmup (freeze_gc_heap)
    so gen2 never scans model, KV-cache, or CUDA-graph objects. Stopping
    the policy must reclaim benchmark garbage and then put that baseline
    freeze back, not leave the whole heap unfrozen."""
    monkeypatch.setenv("DYN_FPM_GC_FREEZE_INTERVAL_S", "3600")
    thresholds = gc.get_threshold()
    gc_policy = _fresh_module(monkeypatch, "freeze")
    ext = gc_policy.FpmGcWorkerExtension()
    try:
        # Worker ordering: the policy auto-starts on extension import, and
        # vLLM's post-warmup freeze lands afterwards.
        assert ext.fpm_gc_start() is True

        # A GC-tracked stand-in for a static serving object. Plain object()
        # instances and atomic-only dicts are untracked in CPython and would
        # make the visibility assertion below vacuous.
        class Static:
            pass

        sentinel = Static()
        assert gc.is_tracked(sentinel)
        gc.collect()
        gc.freeze()
        assert gc.get_freeze_count() > 0

        class Node:
            pass

        node = Node()
        node.self_ref = node
        ref = weakref.ref(node)
        del node
        with gc_policy._lock:
            gc.freeze()  # a tick pins the benchmark-era cycle

        ext.fpm_gc_stop()

        assert ref() is None, "benchmark garbage must be reclaimed"
        # gc.get_objects() walks generations 0-2 only, never the permanent
        # generation: a live baseline object that is absent from it has been
        # re-frozen. The pre-fix stop (unfreeze without re-freeze) leaves the
        # sentinel visible here.
        assert not any(
            obj is sentinel for obj in gc.get_objects()
        ), "serving freeze was not re-established after stop"
    finally:
        gc_policy.stop_gc_policy()
        gc.unfreeze()  # do not leave the pytest heap frozen for later tests
    _assert_gc_equivalent_to_never_benchmarked(gc_policy, thresholds)


_LIFECYCLE_CHILD = r"""
import gc, json, os, sys, weakref
os.environ["DYN_FPM_GC_POLICY"] = "freeze"
os.environ["DYN_FPM_GC_FREEZE_INTERVAL_S"] = "3600"
baseline_thresholds = gc.get_threshold()

# Worker semantics: importing the extension module auto-starts the policy.
import dynamo.vllm.gc_policy as gc_policy
ext = gc_policy.FpmGcWorkerExtension()
thread = gc_policy._freeze_thread
assert thread is not None and thread.is_alive()

# vLLM's post-warmup serving freeze happens after the policy is already on.
class Static:
    pass

sentinel = Static()  # GC-tracked stand-in for model/KV/CUDA-graph objects
assert gc.is_tracked(sentinel)
gc.collect()
gc.freeze()
assert gc.get_freeze_count() > 0

class Node:
    pass

node = Node()
node.self_ref = node
bench_cycle = weakref.ref(node)
del node
with gc_policy._lock:
    gc.freeze()  # periodic tick pins the benchmark-era cycle

ext.fpm_gc_stop()  # what the launcher's collective_rpc("fpm_gc_stop") runs

probe = Node()
probe.self_ref = probe
new_cycle = weakref.ref(probe)
del probe
gc.collect()

print(json.dumps({
    "thresholds_restored": gc.get_threshold() == baseline_thresholds,
    "enabled": gc.isenabled(),
    "thread_dead": not thread.is_alive(),
    # permanent-generation membership: absent from gc.get_objects() == frozen
    "freeze_reestablished": not any(o is sentinel for o in gc.get_objects()),
    "benchmark_cycle_reclaimed": bench_cycle() is None,
    "new_cycle_collectible": new_cycle() is None,
    "policy_stopped": gc_policy._started is False,
}))
"""


def test_worker_process_gc_equivalent_after_stop():
    """Process-level equivalence check in a fresh interpreter mirroring a
    model worker (policy auto-started on import, vLLM serving freeze applied
    afterwards): after ``fpm_gc_stop`` the worker's GC state matches a process
    that never benchmarked (thresholds, enabled flag, dead freeze thread,
    serving freeze at least as large as the baseline) and both benchmark-era
    garbage and new cycles are collectible. The normal-vs-abort distinction
    lives in the launcher and is covered in test_vllm_worker_factory.py with
    a real ``status=failed`` artifact driving the failure path."""
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.startswith("DYN_FPM_GC")}
    result = subprocess.run(
        [sys.executable, "-c", _LIFECYCLE_CHILD],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    failed = {k: v for k, v in report.items() if v is not True}
    assert not failed, failed
