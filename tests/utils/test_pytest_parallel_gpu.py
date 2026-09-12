# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit + makespan-simulation tests for the GPU-parallel scheduler.

Covers the pure scheduling core (``_select_launches`` / ``_priority_key``) and a
discrete-event makespan simulation on the real ``job-log.txt`` workload, showing
the VRAM-aware ordering beats the legacy timeout-sorted first-fit. No GPU or
``pynvml`` required -- the scheduler core is pure arithmetic.
"""

from __future__ import annotations

import io
import itertools
import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.utils import pytest_parallel_gpu, vram_utils
from tests.utils.pytest_parallel_gpu import (
    _GpuState,
    _priority_key,
    _release_gpus,
    _reserve_gpus,
    _select_launches,
    _status_lines,
    _TestEntry,
    _unschedulable_reason,
)
from tests.utils.vram_utils import (
    _TEST_META_FILENAME,
    DEFAULT_GPU_COUNT,
    VRAM_MULTI_PROC_MARGIN,
    gpu_count_from_marker_names,
    write_test_meta,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _gpu(
    index: int,
    total_gib: float,
    *,
    budget_used: float = 0.0,
    running_count: int = 0,
) -> _GpuState:
    return _GpuState(
        index=index,
        total_gib=total_gib,
        budget_multi=total_gib * (1.0 - VRAM_MULTI_PROC_MARGIN),
        budget_used=budget_used,
        running_count=running_count,
    )


def _t(name: str, profiled: float, timeout: float = 600.0, gpus: int = 1) -> _TestEntry:
    return _TestEntry(
        id=name,
        name=name,
        profiled_gib=profiled,
        timeout=timeout,
        gpu_count=gpus,
    )


def _select(pending, gpus, *, num_slots, running_count=0, actual_free=None):
    if actual_free is None:
        actual_free = {gi: gs.total_gib - gs.budget_used for gi, gs in gpus.items()}
    return _select_launches(
        pending=pending,
        gpu_states=gpus,
        actual_free=actual_free,
        num_slots=num_slots,
        running_count=running_count,
    )


# --------------------------------------------------------------------------- #
# _priority_key
# --------------------------------------------------------------------------- #
def test_priority_orders_vram_first_then_duration_then_size():
    filler = _t("filler", 0.0, timeout=600)  # est 200s but zero VRAM
    short_big = _t("short_big", 13.0, timeout=300)  # est 100s
    long_small = _t("long_small", 3.8, timeout=1800)  # est 600s
    long_big = _t("long_big", 7.6, timeout=1800)  # est 600s, ties long_small

    ordered = sorted(
        [filler, short_big, long_small, long_big], key=_priority_key, reverse=True
    )
    names = [t.name for t in ordered]

    # VRAM tests all precede the filler.
    assert names[-1] == "filler"
    # Among VRAM tests: longest est_duration first; ties broken by larger VRAM.
    assert names[:3] == ["long_big", "long_small", "short_big"]


# --------------------------------------------------------------------------- #
# _select_launches: pairing / packing
# --------------------------------------------------------------------------- #
def test_pairs_large_with_small_on_one_gpu():
    # 22 GiB card, 19 GiB multi-proc budget. A 13 GiB test should anchor the GPU
    # (full-card cap) and a 3.8 GiB test should pack alongside it; a second 3.8
    # no longer fits (19 - 16.8 = 2.2).
    gpus = {0: _gpu(0, 22.0)}
    pending = [_t("big", 13.0), _t("small_a", 3.8), _t("small_b", 3.8)]

    launches = _select(pending, gpus, num_slots=8)

    assert launches == [(0, (0,)), (1, (0,))]


def test_first_test_uses_full_card_then_multi_proc_margin():
    # A 20 GiB test fits only because it is first (full 24 GiB cap). Once it is
    # placed the cap drops to budget_multi (20.4), so the 4 GiB test is rejected.
    gpus = {0: _gpu(0, 24.0)}  # budget_multi = 20.4
    pending = [_t("anchor", 20.0), _t("extra", 4.0)]

    launches = _select(pending, gpus, num_slots=8)

    assert launches == [(0, (0,))]


def test_multi_gpu_spreads_then_packs():
    # Two 22 GiB cards. 13 anchors GPU0; 12 cannot share it (19-13<12) so it
    # anchors GPU1; the 3.8 best-fits onto GPU1 (more free budget than GPU0).
    gpus = {0: _gpu(0, 22.0), 1: _gpu(1, 22.0)}
    pending = [_t("a", 13.0), _t("b", 12.0), _t("c", 3.8)]

    launches = _select(pending, gpus, num_slots=8)

    assert launches == [(0, (0,)), (1, (1,)), (2, (1,))]


# --------------------------------------------------------------------------- #
# _select_launches: fillers
# --------------------------------------------------------------------------- #
def test_zero_vram_fillers_bypass_budget():
    # GPU budget fully committed to a running VRAM test, yet 0-GiB fillers must
    # still take free slots (they allocate no memory).
    gpus = {0: _gpu(0, 22.0, budget_used=19.0, running_count=1)}
    pending = [_t("f0", 0.0), _t("f1", 0.0)]

    launches = _select(pending, gpus, num_slots=8, running_count=1)

    assert launches == [(0, (0,)), (1, (0,))]


def test_slot_cap_is_global():
    gpus = {0: _gpu(0, 80.0)}  # budget is huge; the cap here is the slot count
    pending = [_t("a", 3.8), _t("b", 3.8), _t("c", 3.8)]

    launches = _select(pending, gpus, num_slots=2, running_count=1)

    assert len(launches) == 1  # 1 running + 1 new == 2 slots


def test_actual_usage_gate_blocks_when_live_vram_exceeds_budget():
    # The reserved-budget gate alone would allow a 13 GiB test (the GPU is idle
    # by markers), but a live nvidia-smi reading of only 5 GiB free (an init
    # spike or residual allocation the markers don't reflect) must block it via
    # the independent actual-usage gate.
    gpus = {0: _gpu(0, 22.0)}  # budget_used=0 -> reserved-budget gate allows 13
    pending = [_t("big", 13.0)]

    # Budget gate alone (actual_free defaults to total - budget = 22): launches.
    assert _select(pending, gpus, num_slots=8) == [(0, (0,))]

    # Only 5 GiB actually free => 17 GiB live-used; 17 + 13 = 30 > 22 cap -> the
    # actual-usage gate blocks the launch the budget gate would have allowed.
    assert _select(pending, gpus, num_slots=8, actual_free={0: 5.0}) == []


# --------------------------------------------------------------------------- #
# _select_launches: anti-starvation reservation
# --------------------------------------------------------------------------- #
def test_reservation_keeps_room_for_blocked_high_priority_test():
    # An 8 GiB test is running (budget_used=8). The highest-priority pending test
    # needs 12 GiB and cannot fit yet (19-8=11<12). Without a reservation, two
    # 3.8 GiB backfills would fit now (8+3.8+3.8=15.6<=19) -- but then when the
    # 8 GiB test frees, only 19-7.6=11.4 GiB is free and the 12 GiB test is still
    # blocked. The reservation caps backfill at cap-required (19-12=7), so only
    # one 3.8 launches, guaranteeing the 12 fits once the 8 frees.
    gpus = {0: _gpu(0, 22.0, budget_used=8.0, running_count=1)}
    pending = [_t("blocked12", 12.0), _t("fill_a", 3.8), _t("fill_b", 3.8)]

    launches = _select(pending, gpus, num_slots=8, running_count=1)

    assert launches == [(1, (0,))]  # only one 3.8 backfill; room held for the 12


def test_blocked_test_launches_once_occupant_frees():
    # Same setup, but now the 8 GiB occupant has freed (budget_used back to the
    # single 3.8 backfill). The 12 GiB test is now first and must launch.
    gpus = {0: _gpu(0, 22.0, budget_used=3.8, running_count=1)}
    pending = [_t("blocked12", 12.0), _t("fill_b", 3.8)]

    launches = _select(pending, gpus, num_slots=8, running_count=1)

    # 12 fits (19-3.8=15.2) and is highest priority; the extra 3.8 no longer fits
    # (19-15.8=3.2<3.8).
    assert launches == [(0, (0,))]


# --------------------------------------------------------------------------- #
# makespan simulation: new ordering vs legacy timeout-sorted first-fit
# --------------------------------------------------------------------------- #
# (name, profiled_gib, real_runtime_s, timeout_s) for the 23 VRAM tests observed
# in job-log.txt, plus 213 zero-VRAM "needs vllm container, no GPU memory" unit
# tests that each pay ~27 s of interpreter/import startup in their own subprocess.
_GPU_TESTS = [
    ("mm_shm", 7.6, 233, 1800),
    ("mm_nixl", 7.6, 227, 1800),
    ("mm_disabled", 7.6, 184, 1800),
    ("engine", 3.8, 80, 900),
    ("serve_mm_agg_video", 8.2, 131, 600),
    ("self_benchmark", 3.8, 73, 600),
    ("serve_aggregated", 3.8, 168, 480),
    ("serve_mm_agg_router_qwen3", 13.0, 98, 400),
    ("router_kv_basic", 6.9, 76, 360),
    ("router_kv_without_block", 6.9, 75, 360),
    ("router_decisions", 6.9, 79, 360),
    ("router_indexers_sync", 6.9, 136, 360),
    ("serve_aggregated_lmcache", 3.8, 182, 360),
    ("serve_lmcache_multiproc", 3.8, 177, 360),
    ("serve_lmcache_mp", 3.8, 187, 360),
    ("serve_agg_request_plane", 3.8, 165, 360),
    ("serve_embedding_agg", 5.0, 92, 360),
    ("serve_mm_agg_gemma4", 12.0, 192, 300),
    ("serve_guided_decoding", 3.8, 67, 180),
    ("kvbm_offload", 3.8, 104, 170),
    ("kvbm_eviction", 3.8, 105, 170),
    ("kvbm_onboarding", 3.8, 77, 160),
    ("kvbm_chunked", 3.8, 100, 140),
]
_N_FILLERS = 213
_FILLER_RUNTIME = 27
_FILLER_TIMEOUT = 600  # no @pytest.mark.timeout -> scheduler default


class _SimTest(_TestEntry):
    """_TestEntry carrying a real runtime for the simulator."""

    def __init__(self, name: str, profiled: float, runtime: float, timeout: float):
        super().__init__(id=name, name=name, profiled_gib=profiled, timeout=timeout)
        self.runtime = runtime


def _build_workload() -> list[_SimTest]:
    tests = [_SimTest(n, p, r, to) for (n, p, r, to) in _GPU_TESTS]
    tests += [
        _SimTest(f"filler_{i}", 0.0, _FILLER_RUNTIME, _FILLER_TIMEOUT)
        for i in range(_N_FILLERS)
    ]
    return tests


def _legacy_select(pending, gpu_states, actual_free, num_slots, running_count):
    """The pre-change algorithm: first-fit (most-available-budget) over pending
    in its given order, no filler bypass, no reservation.

    Returns the same ``(index, device tuple)`` shape as ``_select_launches`` so
    the simulator can drive either one; it only ever assigns a single device,
    which is all the pre-change algorithm could do."""
    tent = {
        gi: {
            "budget": gs.budget_used,
            "free": actual_free[gi],
            "count": gs.running_count,
        }
        for gi, gs in gpu_states.items()
    }
    to_launch: list[tuple[int, tuple[int, ...]]] = []
    for i, test in enumerate(pending):
        if running_count + len(to_launch) >= num_slots:
            break
        best_gi, best_avail = None, -1.0
        for gi, gs in gpu_states.items():
            ts = tent[gi]
            cap = gs.budget_multi if ts["count"] >= 1 else gs.total_gib
            avail = cap - ts["budget"]
            if avail < test.profiled_gib:
                continue
            if (gs.total_gib - ts["free"]) + test.profiled_gib > cap:
                continue
            if avail > best_avail:
                best_gi, best_avail = gi, avail
        if best_gi is not None:
            to_launch.append((i, (best_gi,)))
            tent[best_gi]["budget"] += test.profiled_gib
            tent[best_gi]["free"] -= test.profiled_gib
            tent[best_gi]["count"] += 1
    return to_launch


def _simulate_makespan(tests, *, num_slots, gpus_total, order_key, select) -> float:
    """Discrete-event makespan model of run_parallel's loop.

    Polls/staggers are omitted (they affect both schedulers equally); GPU actual
    usage is modeled as the sum of running tests' profiled VRAM. Returns the wall
    time in seconds.
    """
    gpu_states = {i: _gpu(i, tot) for i, tot in enumerate(gpus_total)}
    pending = sorted(tests, key=order_key, reverse=True)
    running: list[dict] = []  # {finish, profiled, gpu}
    now = 0.0

    while pending or running:
        actual_free = {
            gi: gs.total_gib - gs.budget_used for gi, gs in gpu_states.items()
        }
        sel = select(
            pending=pending,
            gpu_states=gpu_states,
            actual_free=actual_free,
            num_slots=num_slots,
            running_count=len(running),
        )
        if sel:
            for idx, gpus in sorted(sel, key=lambda x: x[0], reverse=True):
                t = pending.pop(idx)
                for gi in gpus:
                    gpu_states[gi].budget_used += t.profiled_gib
                    gpu_states[gi].running_count += 1
                running.append(
                    {
                        "finish": now + t.runtime,
                        "profiled": t.profiled_gib,
                        "gpus": gpus,
                    }
                )
            continue
        assert running, "deadlock: pending tests but nothing running"
        now = min(r["finish"] for r in running)
        still = []
        for r in running:
            if r["finish"] <= now:
                for gi in r["gpus"]:
                    gpu_states[gi].budget_used -= r["profiled"]
                    gpu_states[gi].running_count -= 1
            else:
                still.append(r)
        running = still
    return now


def _retune_3x(tests):
    """Set timeout = 3x runtime (repo convention), floored so import-heavy
    fillers keep headroom. Mutates and returns the freshly-built tests."""
    for t in tests:
        floor = 90.0 if t.profiled_gib <= 0 else 30.0
        t.timeout = max(3.0 * t.runtime, floor)
    return tests


def test_new_algorithm_beats_legacy_at_equal_timeouts():
    # Isolate the *scheduling logic* change: identical (observed) timeouts and
    # runtimes for both, only the ordering + selection differ. Single 22 GiB
    # card, 8 slots -- the configuration from job-log.txt.
    kwargs = dict(num_slots=8, gpus_total=[22.0])

    legacy = _simulate_makespan(
        _build_workload(),
        order_key=lambda t: t.timeout,
        select=_legacy_select,
        **kwargs,
    )
    new = _simulate_makespan(
        _build_workload(), order_key=_priority_key, select=_select_launches, **kwargs
    )

    # The legacy order runs the 0-GiB fillers (default timeout 600) ahead of the
    # real GPU tests (timeout <= 480), leaving the GPU memory-idle then
    # serializing the VRAM tests on the tail. The new order front-loads + pairs
    # them regardless of how (in)accurate the timeouts are.
    print(
        f"\nalgo-only: legacy={legacy:.0f}s  new={new:.0f}s  ratio={new / legacy:.2f}"
    )
    assert new <= 0.90 * legacy  # >= 10% makespan reduction from the algorithm


def test_full_change_beats_status_quo():
    # What the PR actually ships: the new algorithm AND timeouts retuned to 3x
    # runtime, vs today's status quo (legacy ordering + observed timeouts, where
    # the fillers have no timeout marker at all).
    kwargs = dict(num_slots=8, gpus_total=[22.0])

    status_quo = _simulate_makespan(
        _build_workload(),
        order_key=lambda t: t.timeout,
        select=_legacy_select,
        **kwargs,
    )
    shipped = _simulate_makespan(
        _retune_3x(_build_workload()),
        order_key=_priority_key,
        select=_select_launches,
        **kwargs,
    )

    print(
        f"\nstatus-quo={status_quo:.0f}s  shipped={shipped:.0f}s  "
        f"ratio={shipped / status_quo:.2f}"
    )
    assert shipped <= 0.85 * status_quo  # >= 15% end-to-end makespan reduction


def test_effective_cpu_budget_caps_num_cpus_at_detected_quota(monkeypatch):
    monkeypatch.setattr(vram_utils, "_cgroup_cpu_budget", lambda: 1)
    monkeypatch.setenv("NUM_CPUS", "2")
    assert vram_utils.effective_cpu_budget() == 1

    monkeypatch.setattr(vram_utils, "_cgroup_cpu_budget", lambda: 96)
    assert vram_utils.effective_cpu_budget() == 2


@pytest.mark.parametrize("invalid_budget", ["bogus", "inf", "1e309", "0", "-1"])
def test_effective_cpu_budget_warns_for_invalid_num_cpus(
    monkeypatch, caplog, invalid_budget
):
    monkeypatch.setattr(vram_utils, "_cgroup_cpu_budget", lambda: 96)
    monkeypatch.setenv("NUM_CPUS", invalid_budget)

    assert vram_utils.effective_cpu_budget() == 96
    assert f"Ignoring invalid NUM_CPUS='{invalid_budget}'" in caplog.text


@pytest.mark.parametrize(
    ("cpu_max", "expected"),
    [
        ("400000 100000", 4),
        ("50000 100000", 1),
        ("max 100000", None),
        ("invalid", None),
    ],
)
def test_cgroup_v2_cpu_budget(tmp_path, cpu_max, expected):
    (tmp_path / "cpu.max").write_text(cpu_max)

    assert vram_utils._cgroup_cpu_budget(tmp_path) == expected


@pytest.mark.parametrize(
    ("quota", "period", "expected"),
    [
        ("400000", "100000", 4),
        ("50000", "100000", 1),
        ("-1", "100000", None),
        ("400000", "0", None),
    ],
)
def test_cgroup_v1_cpu_budget(tmp_path, quota, period, expected):
    cpu_root = tmp_path / "cpu"
    cpu_root.mkdir()
    (cpu_root / "cpu.cfs_quota_us").write_text(quota)
    (cpu_root / "cpu.cfs_period_us").write_text(period)

    assert vram_utils._cgroup_cpu_budget(tmp_path) == expected


def test_simulation_conserves_work_and_respects_budget():
    # Sanity guard for the simulator itself: every test runs, and makespan is at
    # least the longest single test and at least total-work / slots.
    tests = _build_workload()
    total_work = sum(t.runtime for t in tests)
    longest = max(t.runtime for t in tests)

    makespan = _simulate_makespan(
        tests,
        num_slots=8,
        gpus_total=[22.0],
        order_key=_priority_key,
        select=_select_launches,
    )

    assert makespan >= longest
    assert makespan >= total_work / 8 - 1e-6


# --------------------------------------------------------------------------- #
# multi-GPU (gang) scheduling
#
# A gpu_2 test needs two distinct devices at once. `profiled_vram_gib` is the
# maximum per-device peak, so the same figure is reserved on every member --
# see `_reserve_gpus`. These tests pin the semantics the scheduler must keep.
# --------------------------------------------------------------------------- #
def _snapshot(pending, gpu_states, actual_free):
    """Deep value-copy of everything `_select_launches` is handed."""
    return (
        [
            (t.id, t.profiled_gib, t.timeout, t.gpu_count, t.assigned_gpus)
            for t in pending
        ],
        {
            gi: (
                gs.index,
                gs.total_gib,
                gs.budget_multi,
                gs.budget_used,
                gs.running_count,
            )
            for gi, gs in gpu_states.items()
        },
        dict(actual_free),
    )


def _assert_selection_invariants(launches, pending, gpu_states, actual_free, before):
    """The scheduler contract, checked after any selection.

    Cardinality, distinctness, stable ordering, capacity, non-negative
    accounting, and that the selector did not mutate its inputs.
    """
    seen: set[int] = set()
    added = {gi: 0.0 for gi in gpu_states}
    counts = {gi: gs.running_count for gi, gs in gpu_states.items()}

    for idx, gpus in launches:
        assert idx not in seen, f"pending index {idx} launched twice"
        seen.add(idx)
        test = pending[idx]
        # I1: a job owns exactly gpu_count distinct GPUs -- never a partial gang.
        assert len(gpus) == test.gpu_count, f"{test.id}: gang size != gpu_count"
        assert len(set(gpus)) == len(gpus), f"{test.id}: duplicate GPU in gang"
        # I6: deterministic, stable device order.
        assert list(gpus) == sorted(gpus), f"{test.id}: gang not ascending"
        for gi in gpus:
            assert gi in gpu_states, f"{test.id}: unknown GPU {gi}"
            added[gi] += test.profiled_gib
            counts[gi] += 1

    for gi, gs in gpu_states.items():
        committed = gs.budget_used + added[gi]
        # I2: a reservation never exceeds the physical card.
        assert committed <= gs.total_gib + 1e-9, f"GPU{gi} over capacity"
        # I5: no resource count goes negative.
        assert committed >= -1e-9, f"GPU{gi} negative budget"
        assert counts[gi] >= 0, f"GPU{gi} negative process count"

    # Selector purity: pure means pure.
    assert _snapshot(pending, gpu_states, actual_free) == before


def _select_checked(pending, gpus, *, num_slots, running_count=0, actual_free=None):
    """`_select` plus the invariant sweep, for use by every gang test."""
    if actual_free is None:
        actual_free = {gi: gs.total_gib - gs.budget_used for gi, gs in gpus.items()}
    before = _snapshot(pending, gpus, actual_free)
    launches = _select_launches(
        pending=pending,
        gpu_states=gpus,
        actual_free=actual_free,
        num_slots=num_slots,
        running_count=running_count,
    )
    _assert_selection_invariants(launches, pending, gpus, actual_free, before)
    return launches


def test_gpu2_takes_two_distinct_devices():
    # The basic case: one gpu_2 test on an idle 2-GPU node takes both cards,
    # in ascending order, as a single launch costing a single slot.
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    pending = [_t("gang", 5.0, gpus=2)]

    assert _select_checked(pending, gpus, num_slots=8) == [(0, (0, 1))]


def test_gpu2_needs_two_devices_and_waits_on_a_single_gpu_node():
    # One card cannot host a gang however much VRAM is free. It must wait
    # rather than be silently downgraded to a single device.
    gpus = {0: _gpu(0, 80.0)}
    pending = [_t("gang", 5.0, gpus=2)]

    assert _select_checked(pending, gpus, num_slots=8) == []


def test_gang_is_atomic_when_one_member_fails_the_budget_gate():
    # GPU0 is idle; GPU1 has only 2 GiB of budget left. A 12 GiB gang must not
    # take GPU0 alone -- a partial gang is never a valid allocation.
    gpus = {
        0: _gpu(0, 22.0),
        1: _gpu(1, 22.0, budget_used=17.0, running_count=1),
    }
    pending = [_t("gang12", 12.0, gpus=2)]

    assert _select_checked(pending, gpus, num_slots=8) == []


def test_gang_is_atomic_when_one_member_fails_the_actual_free_gate():
    # Both GPUs look idle by scheduler budget, so the budget gate alone would
    # place the gang. GPU1 is actually holding 18 of its 22 GiB (an init spike
    # or a process outside this run), so the independent actual-usage gate must
    # veto the whole gang, not just that member.
    gpus = {0: _gpu(0, 22.0), 1: _gpu(1, 22.0)}
    pending = [_t("gang12", 12.0, gpus=2)]

    assert _select_checked(pending, gpus, num_slots=8) == [(0, (0, 1))]
    assert (
        _select_checked(pending, gpus, num_slots=8, actual_free={0: 22.0, 1: 4.0}) == []
    )


def test_gang_picks_the_two_emptiest_of_three_gpus():
    # Best-fit generalizes: the gang takes the two GPUs with the most free
    # budget (1 and 2), leaving the loaded GPU0 alone.
    gpus = {
        0: _gpu(0, 22.0, budget_used=10.0, running_count=1),
        1: _gpu(1, 22.0, budget_used=2.0, running_count=1),
        2: _gpu(2, 22.0),
    }
    pending = [_t("gang", 5.0, gpus=2)]

    assert _select_checked(pending, gpus, num_slots=8) == [(0, (1, 2))]


def test_gang_respects_asymmetric_physical_capacities():
    # A mixed node: only the two 80 GiB cards can hold a 30 GiB-per-device
    # gang; the 22 GiB card is too small and must be skipped.
    gpus = {0: _gpu(0, 22.0), 1: _gpu(1, 80.0), 2: _gpu(2, 80.0)}
    pending = [_t("gang30", 30.0, gpus=2)]

    assert _select_checked(pending, gpus, num_slots=8) == [(0, (1, 2))]


def test_gang_ordering_is_deterministic_under_ties():
    # All three GPUs are identical and idle, so every pair is equally good.
    # The tie-break is the caller's gpu_states ordering, and the emitted set is
    # ascending -- the same answer on every run, so CUDA_VISIBLE_DEVICES is
    # reproducible.
    for _ in range(20):
        gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0), 2: _gpu(2, 80.0)}
        pending = [_t("gang", 5.0, gpus=2)]
        assert _select_checked(pending, gpus, num_slots=8) == [(0, (0, 1))]

    # A non-ascending gpu_states order keeps the *caller's* preference (2 then
    # 0 are the first two offered) while still emitting an ascending set.
    gpus = {2: _gpu(2, 80.0), 0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    pending = [_t("gang", 5.0, gpus=2)]
    assert _select_checked(pending, gpus, num_slots=8) == [(0, (0, 2))]


def test_zero_profile_gang_still_takes_its_full_device_count():
    # A gpu_2 test with profiled_vram_gib(0) allocates no VRAM but still needs
    # two devices *visible*. The count is a visibility requirement, not a
    # memory one, so the budget bypass must not collapse it to one device.
    gpus = {0: _gpu(0, 22.0, budget_used=19.0, running_count=1), 1: _gpu(1, 22.0)}
    pending = [_t("mock_gang", 0.0, gpus=2)]

    assert _select_checked(pending, gpus, num_slots=8, running_count=1) == [(0, (0, 1))]


def test_zero_profile_gang_blocked_when_devices_are_scarce():
    # ...and it is still bounded by the device count, not by memory.
    gpus = {0: _gpu(0, 22.0)}
    pending = [_t("mock_gang", 0.0, gpus=2)]

    assert _select_checked(pending, gpus, num_slots=8) == []


def test_mixed_gpu1_and_gpu2_share_a_node_without_overlapping():
    # The gang goes first (longest est_duration), taking both cards. The two
    # gpu_1 tests then pack into the budget left on each card. No device is
    # over-committed and the gang's per-device reservation is respected.
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    pending = [
        _t("gang", 20.0, timeout=1800, gpus=2),
        _t("single_a", 20.0, timeout=600),
        _t("single_b", 20.0, timeout=600),
    ]

    launches = _select_checked(pending, gpus, num_slots=8)

    assert launches[0] == (0, (0, 1))
    placed = {idx: gpus_ for idx, gpus_ in launches}
    assert set(placed) == {0, 1, 2}
    # The two singles land on different cards -- best-fit spreads them.
    assert placed[1] != placed[2]


def test_gang_reservation_is_all_or_none():
    # Two GPUs, both busy; a blocked 12 GiB gang needs headroom on BOTH, so the
    # reservation applies to each card rather than just the better one.
    #
    # A gang's backfill limit is the ABSOLUTE committed budget against
    # budget_multi: a card may hold at most `budget_multi - required`
    # = 18.7 - 12.0 = 6.7 GiB while the gang waits on it. Both cards already
    # hold 8.0, which is past that line, so no filler may be added to either --
    # the gang's headroom has to be reached by attrition, not rented out again.
    # Admitting "just one" 3.8 here is what lets committed budget creep upward
    # one pass at a time until the gang can never assemble; see
    # test_blocked_gang_is_not_starved_by_endless_lower_priority_backfill.
    gpus = {
        0: _gpu(0, 22.0, budget_used=8.0, running_count=1),
        1: _gpu(1, 22.0, budget_used=8.0, running_count=1),
    }
    pending = [
        _t("gang12", 12.0, timeout=1800, gpus=2),
        _t("fill_a", 3.8),
        _t("fill_b", 3.8),
        _t("fill_c", 3.8),
    ]

    launches = _select_checked(pending, gpus, num_slots=8, running_count=2)

    # The gang itself cannot run yet (18.7 - 8.0 = 10.7 < 12).
    assert all(idx != 0 for idx, _ in launches)
    assert launches == []


def test_gang_reservation_admits_backfill_that_still_leaves_its_headroom():
    # The reservation throttles backfill; it does not ban it. A gang-held card
    # may still take work while it stays inside `budget_multi - required`
    # = 18.7 - 12.0 = 6.7 GiB.
    #
    # Note the two states are mutually exclusive on a single card: a card that
    # is *blocking* the gang is by definition already past that line, so it can
    # never also accept a filler. Backfill under a gang reservation is only
    # reachable on a member that is NOT the blocker -- here GPU0 has room and
    # GPU1 is the blocker, so the gang waits while GPU0 keeps working.
    gpus = {
        0: _gpu(0, 22.0, budget_used=1.0, running_count=1),
        1: _gpu(1, 22.0, budget_used=8.0, running_count=1),
    }
    pending = [
        _t("gang12", 12.0, timeout=1800, gpus=2),
        _t("fill_a", 3.8),
        _t("fill_b", 3.8),
    ]

    launches = _select_checked(pending, gpus, num_slots=8, running_count=2)

    # Gang blocked by GPU1 (18.7 - 8.0 = 10.7 < 12). One filler fits on GPU0
    # (1.0 + 3.8 = 4.8 <= 6.7); the second does not (8.6 > 6.7), and neither
    # may touch GPU1.
    assert launches == [(1, (0,))]


# --------------------------------------------------------------------------- #
# cross-pass progress -- the property a single-pass assertion cannot see
# --------------------------------------------------------------------------- #
def _drive_passes(
    gpus,
    seeded,
    *,
    backfill,
    passes,
    num_slots=8,
    queue_depth=4,
    external_hold=None,
):
    """Drive repeated `_select_launches` passes against live `gpu_states`.

    Mirrors ``run_parallel``'s loop -- retire what finished, select, commit --
    on an integer clock, with ``backfill(i)`` keeping the queue topped up so the
    supply of lower-priority work never runs dry.

    Every other selection test in this file asserts a property of ONE pass on a
    hand-authored state. Starvation is not visible there by construction: it is
    what happens on the pass after next, once ``backfill_added`` has been rebuilt
    and the committed budget it was meant to bound has moved. This driver is the
    only place the suite can see that.

    ``run_time`` is in passes. Returns ``{name: pass index it launched on}``.

    ``external_hold`` maps a GPU index to GiB held by a process OUTSIDE this run
    (another container on the node, a leaked engine). It is subtracted from that
    card's live free reading and is never released, so it models the one thing
    the scheduler's own accounting cannot see. Default ``None`` keeps the
    historical behaviour exactly.
    """
    external_hold = external_hold or {}
    pending = sorted(seeded, key=_priority_key, reverse=True)
    running: dict[str, tuple[_TestEntry, int]] = {}
    started: dict[str, int] = {}
    made = 0

    for now in range(passes):
        for name in [n for n, (_, end) in running.items() if end <= now]:
            _release_gpus(running.pop(name)[0], gpus)

        while len(pending) < queue_depth:
            pending.append(backfill(made))
            made += 1
        pending.sort(key=_priority_key, reverse=True)

        resident = {
            gi: sum(
                t.profiled_gib for t, _ in running.values() if gi in t.assigned_gpus
            )
            for gi in gpus
        }
        held = external_hold(now) if callable(external_hold) else external_hold
        actual_free = {
            gi: gs.total_gib - resident[gi] - held.get(gi, 0.0)
            for gi, gs in gpus.items()
        }
        if len(running) >= num_slots:
            continue
        launches = _select_launches(
            pending=pending,
            gpu_states=gpus,
            actual_free=actual_free,
            num_slots=num_slots,
            running_count=len(running),
        )
        batch = [(pending.pop(idx), got) for idx, got in reversed(launches)]
        for test, got in reversed(batch):
            _reserve_gpus(test, got, gpus)
            running[test.name] = (test, now + max(1, int(test.est_duration)))
            started.setdefault(test.name, now)

    return started


def test_blocked_gang_is_not_starved_by_endless_lower_priority_backfill():
    """A feasible gpu_2 test must not be postponed forever by gpu_1 backfill.

    Two 10 GiB cards (budget_multi 8.5). `hog` occupies one of them for 40
    passes; `gang` needs 6.0 GiB on BOTH, so it is blocked from pass 0 -- the
    idle card fits it but the hog's card does not (8.5 - 3.0 = 5.5 < 6.0).
    Behind them is an endless queue of strictly lower-priority 2.0 GiB gpu_1
    tests (shorter est_duration, smaller VRAM, so `_priority_key` ranks them
    last and `gang` is always scanned first).

    `gang` is feasible: `_unschedulable_reason` clears it, and with no backfill
    at all it launches the moment the hog retires. The requirement is that the
    backfill cannot take that away from it.
    """
    gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0)}
    hog = _t("hog", 3.0, timeout=120)  # est 40 passes
    gang = _t("gang", 6.0, timeout=90, gpus=2)  # est 30 passes, lower priority
    assert _unschedulable_reason(gang, gpus) is None
    assert _priority_key(hog) > _priority_key(gang) > _priority_key(_t("f", 2.0, 15.0))

    started = _drive_passes(
        gpus,
        [hog, gang],
        backfill=lambda i: _t(f"fill{i}", 2.0, timeout=15.0),
        passes=400,
    )

    assert "gang" in started, (
        "gpu_2 test never launched in 400 passes while lower-priority gpu_1 "
        "tests kept launching -- starved"
    )
    # It must not merely launch eventually: it must launch as soon as the hog
    # that was blocking it retires, not after some further backfill has been
    # let in ahead of it.
    assert started["gang"] <= 41, started["gang"]


def test_gpu1_reservation_still_makes_progress_under_the_same_backfill():
    """Control for the test above: the identical workload with gpu_count=1.

    Confirms the gpu_2 assertion is about gang scheduling and not about the
    driver, and pins that the unchanged single-GPU reservation path still lets
    a blocked gpu_1 test through.
    """
    gpus = {0: _gpu(0, 10.0)}
    hog = _t("hog", 3.0, timeout=120)
    blocked = _t("blocked", 6.0, timeout=90)

    started = _drive_passes(
        gpus,
        [hog, blocked],
        backfill=lambda i: _t(f"fill{i}", 2.0, timeout=15.0),
        passes=400,
    )

    assert "blocked" in started
    assert started["blocked"] <= 41, started["blocked"]


def test_nothing_is_admitted_past_a_gang_hold_across_passes():
    """The invariant the progress guarantee rests on.

    While a gang is blocked and holding card `g`, any test admitted to `g` must
    leave `budget_used[g] <= budget_multi[g] - required`. That is what makes the
    headroom monotone, and monotone headroom per card is what makes `gpu_count`
    cards eventually satisfy the gate at the same instant -- without it each
    card oscillates and the k of them need never coincide.

    Checked as a statement about admissions under an existing hold: an increase
    on a pass whose predecessor already had the gang blocked. The pass that
    first creates the hold is excluded -- the blocker that causes the block is
    legitimately placed before any hold exists.
    """
    gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0)}
    hog = _t("hog", 3.0, timeout=120)
    gang = _t("gang", 6.0, timeout=90, gpus=2)
    line = {gi: gs.budget_multi - gang.profiled_gib for gi, gs in gpus.items()}
    prev = {gi: 0.0 for gi in gpus}
    st = {"held_last_pass": False, "ever_launched": False}
    violations: list[tuple[int, float, float]] = []

    def _watch(i):
        # Flag only increases observed while the hold was ALREADY in place on
        # the previous pass: the pass that creates the hold legitimately places
        # the blocker itself. `ever_launched` latches permanently, so the free
        # backfill that follows the gang's own run is not mistaken for a breach.
        if gang.assigned_gpus:
            st["ever_launched"] = True
        holding = st["held_last_pass"] and not st["ever_launched"]
        for gi, gs in gpus.items():
            grew = gs.budget_used > prev[gi] + 1e-9
            if holding and grew and gs.budget_used > line[gi] + 1e-9:
                violations.append((gi, prev[gi], gs.budget_used))
            prev[gi] = gs.budget_used
        st["held_last_pass"] = bool(hog.assigned_gpus) and not st["ever_launched"]
        return _t(f"fill{i}", 2.0, timeout=15.0)

    started = _drive_passes(gpus, [hog, gang], backfill=_watch, passes=120)

    assert "gang" in started
    assert (
        not violations
    ), f"admitted past the gang hold (gi, before, after): {violations}"


def test_gang_hold_admits_backfill_exactly_at_the_line():
    """The boundary of the gang gate: at the line is admitted, over it is not.

    Sized off `budget_multi` itself rather than a literal, so the equality is
    exact whatever `total_gib * (1 - MARGIN)` rounds to. Without this the
    comparison could be `>=` instead of `>` and every other test would still
    pass -- the two differ only on this one state.
    """
    gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0, budget_used=3.0, running_count=1)}
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    exact = gpus[0].budget_multi - gang.profiled_gib  # lands exactly on the line
    over = exact * 1.02

    at_line = _select_checked(
        [gang, _t("fill_exact", exact)],
        gpus,
        num_slots=8,
        running_count=1,
        actual_free={0: 10.0, 1: 7.0},
    )
    assert at_line == [(1, (0,))], "backfill landing exactly on the line must fit"

    past_line = _select_checked(
        [gang, _t("fill_over", over)],
        gpus,
        num_slots=8,
        running_count=1,
        actual_free={0: 10.0, 1: 7.0},
    )
    assert past_line == [], "backfill past the line must be refused"


def test_idle_reserved_card_is_capped_at_budget_multi_not_the_whole_card():
    # An IDLE card reserved for a blocked gang reports the whole-card cap,
    # because that is the cap for whoever lands there first. But the instant a
    # filler lands, the gang's own cap on that card becomes budget_multi. Gating
    # the reservation on the whole-card cap therefore over-grants by exactly
    # VRAM_MULTI_PROC_MARGIN * total_gib (0.15 * 22 = 3.3) and admits a filler
    # that immediately re-blocks the gang the reservation exists to protect.
    #
    # GPU0 idle, GPU1 holds the gang's blocker. An 8.0 filler is inside
    # 22.0 - 12.0 = 10.0 but outside 18.7 - 12.0 = 6.7, so it must be refused.
    gpus = {
        0: _gpu(0, 22.0),
        1: _gpu(1, 22.0, budget_used=8.0, running_count=1),
    }
    pending = [_t("gang12", 12.0, timeout=1800, gpus=2), _t("fill8", 8.0)]

    launches = _select_checked(
        pending, gpus, num_slots=8, running_count=1, actual_free={0: 22.0, 1: 14.0}
    )

    assert launches == []


def test_gang_does_not_reserve_when_it_cannot_hold_every_device():
    # Only one unreserved GPU is left, but the gang needs two. Holding headroom
    # on that single card could never assemble into a launch, so it reserves
    # nothing and leaves the card free for backfill.
    gpus = {
        0: _gpu(0, 22.0, budget_used=8.0, running_count=1),
        1: _gpu(1, 22.0, budget_used=8.0, running_count=1),
    }
    pending = [
        _t("blocked_single", 15.0, timeout=1800),  # reserves the better card
        _t("blocked_gang", 15.0, timeout=1700, gpus=2),  # cannot reserve 2
        _t("fill", 3.8),
    ]

    launches = _select_checked(pending, gpus, num_slots=8, running_count=2)

    # The filler still backfills; the gang held nothing hostage.
    assert launches == [(2, (1,))]


def test_gang_launches_once_its_blockers_leave():
    # Same node as the reservation test, but the 8 GiB occupants have finished.
    # With both cards free the gang is highest priority and must launch on both.
    gpus = {0: _gpu(0, 22.0), 1: _gpu(1, 22.0)}
    pending = [_t("gang12", 12.0, timeout=1800, gpus=2), _t("fill", 3.8)]

    launches = _select_checked(pending, gpus, num_slots=8)

    assert launches[0] == (0, (0, 1))


def test_gang_costs_one_slot_not_one_per_device():
    # A gang is a single pytest subprocess. It must consume one concurrency
    # slot however many GPUs it holds, or -n would silently mean something
    # different for multi-GPU tests.
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    pending = [_t("gang", 5.0, gpus=2), _t("single", 5.0)]

    assert len(_select_checked(pending, gpus, num_slots=1)) == 1
    assert len(_select_checked(pending, gpus, num_slots=2)) == 2


# --------------------------------------------------------------------------- #
# unschedulable-job detection
#
# Without this the run loop waits for memory that is never coming: nothing is
# running, so nothing will ever free, and `while pending or running` spins to
# the CI timeout.
# --------------------------------------------------------------------------- #
def test_unschedulable_more_gpus_than_the_node_has():
    gpus = {0: _gpu(0, 80.0)}
    reason = _unschedulable_reason(_t("gang", 5.0, gpus=2), gpus)
    assert reason is not None and "needs 2 GPUs" in reason


def test_unschedulable_profiled_larger_than_every_card():
    gpus = {0: _gpu(0, 22.0), 1: _gpu(1, 22.0)}
    reason = _unschedulable_reason(_t("huge", 30.0), gpus)
    assert reason is not None and "30.0 GiB" in reason


def test_unschedulable_not_enough_large_cards_for_the_gang():
    # One 80 GiB card is big enough, but a 2-GPU gang needs two of them.
    gpus = {0: _gpu(0, 22.0), 1: _gpu(1, 80.0)}
    reason = _unschedulable_reason(_t("gang30", 30.0, gpus=2), gpus)
    assert reason is not None and "only 1 of 2 that large" in reason


def test_schedulable_jobs_report_no_reason():
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    assert _unschedulable_reason(_t("single", 5.0), gpus) is None
    assert _unschedulable_reason(_t("gang", 5.0, gpus=2), gpus) is None
    # A zero-VRAM gang is bounded by device count only.
    assert _unschedulable_reason(_t("mock", 0.0, gpus=2), gpus) is None
    # A busy GPU does not make a job impossible -- it will free.
    busy = {0: _gpu(0, 80.0, budget_used=79.0, running_count=3), 1: _gpu(1, 80.0)}
    assert _unschedulable_reason(_t("gang", 5.0, gpus=2), busy) is None


def test_impossible_job_is_detected_rather_than_deadlocking():
    # The state the detector exists for: nothing running, nothing launchable,
    # and no future event that could change either.
    gpus = {0: _gpu(0, 22.0)}
    pending = [_t("gang", 5.0, gpus=2)]

    assert _select_checked(pending, gpus, num_slots=8, running_count=0) == []
    assert _unschedulable_reason(pending[0], gpus) is not None


# --------------------------------------------------------------------------- #
# reservation lifecycle: conservation on every terminal path
# --------------------------------------------------------------------------- #
def _totals(gpu_states):
    return {
        gi: (round(gs.budget_used, 9), gs.running_count)
        for gi, gs in gpu_states.items()
    }


def test_reserve_then_release_conserves_every_member():
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0), 2: _gpu(2, 80.0)}
    idle = _totals(gpus)
    test = _t("gang", 5.0, gpus=2)

    _reserve_gpus(test, (1, 0), gpus)

    assert test.assigned_gpus == (0, 1)  # stored ascending regardless of input
    assert _totals(gpus)[0] == (5.0, 1)
    assert _totals(gpus)[1] == (5.0, 1)
    assert _totals(gpus)[2] == (0.0, 0)  # untouched

    _release_gpus(test, gpus)

    assert test.assigned_gpus == ()
    assert _totals(gpus) == idle


def test_release_is_idempotent_so_a_double_terminal_path_cannot_leak():
    # Completion, failure, runtime-skip and retry all funnel through
    # `_release_gpus`. Releasing twice must not drive a count negative.
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    idle = _totals(gpus)
    test = _t("gang", 5.0, gpus=2)

    _reserve_gpus(test, (0, 1), gpus)
    _release_gpus(test, gpus)
    _release_gpus(test, gpus)

    assert _totals(gpus) == idle
    for gs in gpus.values():
        assert gs.budget_used >= 0.0
        assert gs.running_count >= 0


def test_retry_cycle_conserves_across_many_rounds():
    # A retried test is released and re-reserved, possibly onto a different
    # gang. After the final release the node must be exactly as it started.
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0), 2: _gpu(2, 80.0)}
    idle = _totals(gpus)
    test = _t("flaky_gang", 7.5, gpus=2)

    for gang in [(0, 1), (1, 2), (0, 2), (0, 1)]:
        _reserve_gpus(test, gang, gpus)
        assert sum(gs.running_count for gs in gpus.values()) == 2
        _release_gpus(test, gpus)
        assert _totals(gpus) == idle

    assert _totals(gpus) == idle


def test_mixed_workload_release_order_does_not_matter():
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    idle = _totals(gpus)
    gang = _t("gang", 5.0, gpus=2)
    single = _t("single", 3.8)

    _reserve_gpus(gang, (0, 1), gpus)
    _reserve_gpus(single, (1,), gpus)
    assert _totals(gpus)[1] == (8.8, 2)

    _release_gpus(gang, gpus)  # release the gang first
    assert _totals(gpus)[1] == (3.8, 1)
    _release_gpus(single, gpus)
    assert _totals(gpus) == idle


# --------------------------------------------------------------------------- #
# gpu_count metadata
# --------------------------------------------------------------------------- #
def test_gpu_count_resolves_from_the_hardware_marker():
    assert gpu_count_from_marker_names(["gpu_1", "vllm", "e2e"]) == 1
    assert gpu_count_from_marker_names(["e2e", "gpu_2"]) == 2
    assert gpu_count_from_marker_names(["gpu_4"]) == 4
    assert gpu_count_from_marker_names(["gpu_8"]) == 8


def test_gpu_count_is_absent_without_a_hardware_marker():
    # No gpu_N marker -> None, so the caller falls back to DEFAULT_GPU_COUNT.
    # gpu_0 declares "no GPU needed", not a device count, so it does not
    # participate: such tests keep being scheduled as single-device fillers.
    assert gpu_count_from_marker_names([]) is None
    assert gpu_count_from_marker_names(["unit", "pre_merge"]) is None
    assert gpu_count_from_marker_names(["gpu_0"]) is None
    assert DEFAULT_GPU_COUNT == 1


# --------------------------------------------------------------------------- #
# write_test_meta -- the PRODUCER half of the gpu_count path
#
# gpu_count_from_marker_names is well covered below, but the line that actually
# puts its answer into the metadata file is the single point where every gpu_2
# test can be silently downgraded to gpu_1. Deleting it, renaming its key, or
# hardcoding it to 1 leaves the whole scheduler suite green: nothing downstream
# can tell "no gpu_N marker" (legitimately defaults to 1) from "the marker was
# dropped on the floor". These tests pin the producer against REAL marker
# objects taken off real `@pytest.mark.*` decorators.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_2
@pytest.mark.profiled_vram_gib(5.0)
@pytest.mark.requested_vllm_kv_cache_bytes(559_693_824)
@pytest.mark.timeout(420)
def _marker_sample_gpu2():
    """Stand-in carrying the exact marker set of the two real gpu_2 tests."""


@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(4.0)
def _marker_sample_gpu1():
    """A single-GPU test."""


@pytest.mark.gpu_4
@pytest.mark.profiled_vram_gib(9.0)
def _marker_sample_gpu4():
    """A four-GPU test."""


@pytest.mark.gpu_0
@pytest.mark.profiled_vram_gib(0)
def _marker_sample_gpu0():
    """gpu_0 declares 'no GPU required', not a device count."""


@pytest.mark.profiled_vram_gib(3.0)
def _marker_sample_unmarked():
    """No gpu_N marker at all -- the historical default."""


class _MarkedItem:
    """Minimal pytest item over the REAL marks of a decorated function.

    `pytestmark` holds genuine `pytest.Mark` instances, so this drives
    `write_test_meta` with the same objects a collected item would carry rather
    than hand-written name strings. `iter_markers` yields closest-first, which
    is pytest's own ordering and the order `get_closest_marker` depends on.
    """

    def __init__(self, nodeid: str, func):
        self.nodeid = nodeid
        self._marks = list(reversed(getattr(func, "pytestmark", [])))

    def iter_markers(self, name: str | None = None):
        for mark in self._marks:
            if name is None or mark.name == name:
                yield mark

    def get_closest_marker(self, name: str):
        return next(self.iter_markers(name), None)


def _write_and_read(items, tmp_path) -> dict:
    write_test_meta(items, dest_dir=str(tmp_path))
    written = tmp_path / _TEST_META_FILENAME
    assert written.exists(), "write_test_meta produced no metadata file"
    return json.loads(written.read_text())


def test_write_test_meta_serializes_gpu_count_from_a_real_gpu_2_marker(tmp_path):
    nodeid = "tests/serve/test_vllm.py::test_embedding_multi_worker_same_model"
    meta = _write_and_read([_MarkedItem(nodeid, _marker_sample_gpu2)], tmp_path)[nodeid]

    # The producer line itself.
    assert meta["gpu_count"] == 2
    # And the consumer, read exactly the way run_parallel reads it.
    assert meta.get("gpu_count", DEFAULT_GPU_COUNT) == 2
    # The rest of the marker set must survive alongside it.
    assert meta["profiled_vram_gib"] == 5.0
    assert meta["requested_vllm_kv_cache_bytes"] == 559_693_824
    assert meta["timeout"] == 420


def test_write_test_meta_gpu_count_round_trips_into_a_scheduled_gang(tmp_path):
    """Producer -> metadata -> _TestEntry -> a two-device selection.

    Closes the loop end to end: if the serialization is dropped the entry
    defaults to one GPU and the selection is a single device, not a gang.
    """
    nodeid = "tests/serve/test_vllm.py::test_gang"
    meta = _write_and_read([_MarkedItem(nodeid, _marker_sample_gpu2)], tmp_path)[nodeid]

    entry = _TestEntry(
        id=nodeid,
        name=nodeid,
        profiled_gib=meta["profiled_vram_gib"],
        timeout=meta["timeout"],
        requested_vllm_kv_cache_bytes=meta["requested_vllm_kv_cache_bytes"],
        gpu_count=meta.get("gpu_count", DEFAULT_GPU_COUNT),
    )
    assert entry.gpu_count == 2

    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    launches = _select_checked([entry], gpus, num_slots=8)
    assert launches == [(0, (0, 1))]


def test_write_test_meta_gpu_count_covers_every_marker_case(tmp_path):
    cases = {
        "one": (_marker_sample_gpu1, 1),
        "four": (_marker_sample_gpu4, 4),
        # gpu_0 means "no GPU required", not a device count, so no key is
        # written and the consumer default of 1 applies -- same as a test that
        # declares no gpu_N marker at all, and same as a metadata file written
        # before the field existed.
        "zero": (_marker_sample_gpu0, None),
        "unmarked": (_marker_sample_unmarked, None),
    }
    written = _write_and_read(
        [_MarkedItem(nid, func) for nid, (func, _) in cases.items()], tmp_path
    )

    for nodeid, (_, expected) in cases.items():
        meta = written[nodeid]
        if expected is None:
            assert "gpu_count" not in meta, nodeid
            assert meta.get("gpu_count", DEFAULT_GPU_COUNT) == DEFAULT_GPU_COUNT
        else:
            assert meta["gpu_count"] == expected, nodeid


def test_write_test_meta_propagates_a_marker_conflict_rather_than_guessing(tmp_path):
    """A conflict must abort collection, not silently pick one of the two."""

    @pytest.mark.gpu_1
    @pytest.mark.gpu_2
    @pytest.mark.profiled_vram_gib(5.0)
    def _conflicted():
        pass

    with pytest.raises(ValueError) as excinfo:
        write_test_meta(
            [_MarkedItem("conflicted", _conflicted)], dest_dir=str(tmp_path)
        )
    assert "['gpu_1', 'gpu_2']" in str(excinfo.value)
    assert not (tmp_path / _TEST_META_FILENAME).exists()


# --------------------------------------------------------------------------- #
# the gang assignment has to survive into the workers
#
# CUDA_VISIBLE_DEVICES is absolute, not relative: a child that sets it to "0"
# gets physical GPU 0 whatever the parent's value was -- the parent's
# restriction is replaced, not compounded. A launch script that hardcodes 0/1
# therefore ignores the gang the scheduler gave it, and on a node with more
# than two GPUs runs its workers on devices reserved for *other* tests, whose
# VRAM the scheduler still believes is disjoint. These tests execute the real
# lines out of the shipped script rather than asserting on its text.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GANG_LAUNCH_SCRIPT = (
    _REPO_ROOT / "examples/backends/vllm/launch/agg_embed_multiworker.sh"
)


def _derive_worker_gpus(cvd: str | None) -> tuple[int, str]:
    """Run the script's own device-derivation block under a given inherited set."""
    text = _GANG_LAUNCH_SCRIPT.read_text()
    start = text.index("IFS=',' read -r -a _VISIBLE_GPUS")
    end = text.index("\n", text.index('WORKER2_GPU="'))
    snippet = text[start:end] + '\necho "${WORKER1_GPU},${WORKER2_GPU}"\n'

    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    proc = subprocess.run(
        ["bash", "-c", snippet], capture_output=True, text=True, env=env
    )
    return proc.returncode, proc.stdout.strip()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_gang_launch_script_uses_the_devices_the_scheduler_assigned():
    if not _GANG_LAUNCH_SCRIPT.exists():
        pytest.skip("launch script not present in this checkout")

    # The case that matters: a gang other than (0,1). The scheduler hands the
    # whole gang down in CUDA_VISIBLE_DEVICES and the workers must land on it.
    assert _derive_worker_gpus("2,3") == (0, "2,3")
    assert _derive_worker_gpus("5,7") == (0, "5,7")
    # Order is the scheduler's, not re-sorted by the script.
    assert _derive_worker_gpus("1,0") == (0, "1,0")
    # The historical shape still works, and an unrestricted manual run still
    # defaults to the first two devices.
    assert _derive_worker_gpus("0,1") == (0, "0,1")
    assert _derive_worker_gpus(None) == (0, "0,1")
    # More devices than it needs: take the first two of the set.
    assert _derive_worker_gpus("0,1,2,3") == (0, "0,1")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_gang_launch_script_refuses_a_device_set_it_cannot_satisfy():
    if not _GANG_LAUNCH_SCRIPT.exists():
        pytest.skip("launch script not present in this checkout")

    # One device cannot host a two-worker gang.
    rc, _ = _derive_worker_gpus("3")
    assert rc != 0
    # Empty means "no devices at all" in CUDA -- it must NOT fall back to 0,1.
    rc, _ = _derive_worker_gpus("")
    assert rc != 0


def test_gang_launch_script_hardcodes_no_absolute_device():
    if not _GANG_LAUNCH_SCRIPT.exists():
        pytest.skip("launch script not present in this checkout")

    offenders = re.findall(
        r"^\s*CUDA_VISIBLE_DEVICES=\d.*$",
        _GANG_LAUNCH_SCRIPT.read_text(),
        re.MULTILINE,
    )
    assert not offenders, f"absolute device pinned in a gang launcher: {offenders}"


def test_conflicting_gpu_count_markers_fail_closed():
    # Guessing low would under-reserve devices and let another test be placed
    # on a GPU this one is already using. The declaration is rejected instead.
    with pytest.raises(ValueError) as excinfo:
        gpu_count_from_marker_names(["gpu_1", "gpu_2"])
    # The offenders must be named, not just the generic "one of gpu_1/..." hint.
    assert "['gpu_1', 'gpu_2']" in str(excinfo.value)

    # Repeating the same marker is not a conflict.
    assert gpu_count_from_marker_names(["gpu_2", "gpu_2"]) == 2


def test_conflicting_markers_are_named_even_from_a_generator():
    # write_test_meta passes a generator over the item's markers, so the error
    # path must not depend on being able to iterate the argument twice.
    names = (n for n in ["e2e", "gpu_2", "vllm", "gpu_4"])
    with pytest.raises(ValueError) as excinfo:
        gpu_count_from_marker_names(names)
    # Asserting on the rendered list, not a bare substring: the message also
    # ends with a generic "one of gpu_1/gpu_2/gpu_4/gpu_8" hint, which would
    # make a substring check pass even on an empty list.
    assert "['gpu_2', 'gpu_4']" in str(excinfo.value)


def test_metadata_without_gpu_count_defaults_to_one_gpu():
    # Backward compatibility: a metadata file written before gpu_count existed
    # has no such key, and must schedule exactly as it always did.
    assert _TestEntry(id="t", name="t", profiled_gib=3.8, timeout=600).gpu_count == 1


# --------------------------------------------------------------------------- #
# gpu_1 regression oracle
#
# `_reference_single_gpu_select` is the selection algorithm exactly as it stood
# before multi-GPU support, kept here as a frozen oracle. Generalizing to gangs
# must not perturb any single-GPU decision, so the two are compared over a
# bounded exhaustive state space and a seeded random sweep. If a future change
# means to alter gpu_1 scheduling, this test fails and the oracle has to be
# updated deliberately -- which is the point.
# --------------------------------------------------------------------------- #
def _reference_single_gpu_select(pending, gpu_states, actual_free, num_slots, running):
    """Frozen pre-gang `_select_launches`; assigns exactly one GPU per test."""
    tentative = {
        gi: {
            "budget": gs.budget_used,
            "free": actual_free[gi],
            "count": gs.running_count,
        }
        for gi, gs in gpu_states.items()
    }
    reserved_req: dict[int, float] = {}
    backfill_added: dict[int, float] = {}
    to_launch: list[tuple[int, int]] = []

    def cap(gi):
        gs = gpu_states[gi]
        return gs.total_gib if tentative[gi]["count"] < 1 else gs.budget_multi

    for idx, test in enumerate(pending):
        if running + len(to_launch) >= num_slots:
            break
        if test.profiled_gib <= 0:
            gi = min(gpu_states, key=lambda g: tentative[g]["count"])
            to_launch.append((idx, gi))
            tentative[gi]["count"] += 1
            continue
        best_gi, best_avail = None, -1.0
        for gi, gs in gpu_states.items():
            ts = tentative[gi]
            c = cap(gi)
            avail = c - ts["budget"]
            if avail < test.profiled_gib:
                continue
            if (gs.total_gib - ts["free"]) + test.profiled_gib > c:
                continue
            if gi in reserved_req and (
                backfill_added[gi] + test.profiled_gib > c - reserved_req[gi]
            ):
                continue
            if avail > best_avail:
                best_gi, best_avail = gi, avail
        if best_gi is not None:
            to_launch.append((idx, best_gi))
            tentative[best_gi]["budget"] += test.profiled_gib
            tentative[best_gi]["free"] -= test.profiled_gib
            tentative[best_gi]["count"] += 1
            if best_gi in reserved_req:
                backfill_added[best_gi] += test.profiled_gib
            continue
        cand, cand_avail = None, -1.0
        for gi in gpu_states:
            if gi in reserved_req:
                continue
            a = cap(gi) - tentative[gi]["budget"]
            if a > cand_avail:
                cand, cand_avail = gi, a
        if cand is not None:
            reserved_req[cand] = test.profiled_gib
            backfill_added[cand] = 0.0
    return to_launch


def _iter_bounded_states():
    """A small, fully enumerated corner of the scheduler's state space.

    Deliberately built from stdlib `itertools` -- the repo has no property-based
    testing dependency, and a fixed enumeration is reproducible without one.
    """
    sizes = (22.0, 80.0)
    useds = (0.0, 15.0)
    counts = (0, 1)
    catalog = (
        (0.0, 1),
        (0.0, 2),
        (3.8, 1),
        (12.0, 1),
        (12.0, 2),
        (20.0, 2),
    )
    for n_gpus in (1, 2, 3):
        for per_gpu in itertools.product(
            itertools.product(sizes, useds, counts), repeat=n_gpus
        ):
            gpus = {
                gi: _gpu(gi, size, budget_used=used, running_count=count)
                for gi, (size, used, count) in enumerate(per_gpu)
            }
            for job_a, job_b in itertools.product(catalog, repeat=2):
                pending = [
                    _t("a", job_a[0], timeout=1800, gpus=job_a[1]),
                    _t("b", job_b[0], timeout=600, gpus=job_b[1]),
                ]
                for tight in (False, True):
                    free = {
                        gi: (
                            gs.total_gib * 0.25
                            if tight
                            else gs.total_gib - gs.budget_used
                        )
                        for gi, gs in gpus.items()
                    }
                    yield pending, gpus, free


def test_bounded_exhaustive_states_hold_every_invariant():
    # Every reachable selection in the enumerated space satisfies gang
    # cardinality, distinctness, ascending order, capacity, non-negative
    # accounting, and selector purity.
    checked = 0
    for pending, gpus, free in _iter_bounded_states():
        for num_slots in (1, 3):
            _select_checked(
                pending,
                gpus,
                num_slots=num_slots,
                running_count=0,
                actual_free=free,
            )
            checked += 1
    assert checked > 2000, f"state space unexpectedly small ({checked})"


def test_bounded_exhaustive_single_gpu_matches_the_frozen_oracle():
    # Restricted to gpu_count == 1, the generalized selector must reproduce the
    # pre-change algorithm decision for decision, including its tie-breaks.
    compared = 0
    for pending, gpus, free in _iter_bounded_states():
        if any(t.gpu_count != 1 for t in pending):
            continue
        for num_slots in (1, 3):
            for running in (0, 1):
                new = _select_launches(
                    pending=pending,
                    gpu_states=gpus,
                    actual_free=free,
                    num_slots=num_slots,
                    running_count=running,
                )
                ref = _reference_single_gpu_select(
                    pending, gpus, free, num_slots, running
                )
                assert new == [
                    (i, (gi,)) for i, gi in ref
                ], f"gpu_1 regression: {new} != {ref}"
                compared += 1
    assert compared > 500, f"oracle comparison too narrow ({compared})"


def test_randomized_single_gpu_matches_the_frozen_oracle():
    # Wider and messier than the exhaustive sweep: shuffled gpu_states ordering
    # (so the tie-break is exercised), mixed card sizes, and live-free readings
    # unrelated to the reserved budget. Seeded, so a failure reproduces.
    rng = random.Random(13647)
    sizes = [16.0, 22.0, 24.0, 40.0, 80.0, 141.0]
    vrams = [0.0, 0.0, 3.8, 5.0, 6.9, 7.6, 12.0, 13.0, 20.0, 30.0, 45.0]

    for _ in range(3000):
        n_gpus = rng.randint(1, 4)
        gpus = {}
        for gi in rng.sample(range(8), n_gpus):  # non-contiguous, out of order
            total = rng.choice(sizes)
            gpus[gi] = _gpu(
                gi,
                total,
                budget_used=round(rng.uniform(0, total * 0.9), 2),
                running_count=rng.randint(0, 3),
            )
        pending = [
            _t(f"t{j}", rng.choice(vrams), timeout=rng.choice([140.0, 600.0, 1800.0]))
            for j in range(rng.randint(1, 6))
        ]
        free = {gi: round(rng.uniform(0, gs.total_gib), 2) for gi, gs in gpus.items()}
        num_slots = rng.randint(1, 8)
        running = rng.randint(0, num_slots)

        new = _select_launches(
            pending=pending,
            gpu_states=gpus,
            actual_free=free,
            num_slots=num_slots,
            running_count=running,
        )
        ref = _reference_single_gpu_select(pending, gpus, free, num_slots, running)
        assert new == [(i, (gi,)) for i, gi in ref]


def test_randomized_gang_selections_hold_every_invariant():
    # The same sweep with gangs mixed in, checked against the invariant set
    # rather than an oracle (there is no pre-change behaviour to compare to).
    rng = random.Random(846)
    sizes = [22.0, 40.0, 80.0]
    vrams = [0.0, 3.8, 12.0, 20.0, 45.0]

    for _ in range(3000):
        n_gpus = rng.randint(1, 4)
        gpus = {}
        for gi in rng.sample(range(8), n_gpus):
            total = rng.choice(sizes)
            gpus[gi] = _gpu(
                gi,
                total,
                budget_used=round(rng.uniform(0, total * 0.85), 2),
                running_count=rng.randint(0, 2),
            )
        pending = [
            _t(
                f"t{j}",
                rng.choice(vrams),
                timeout=rng.choice([140.0, 600.0, 1800.0]),
                gpus=rng.choice([1, 1, 2, 2, 4]),
            )
            for j in range(rng.randint(1, 5))
        ]
        free = {gi: round(rng.uniform(0, gs.total_gib), 2) for gi, gs in gpus.items()}
        _select_checked(
            pending,
            gpus,
            num_slots=rng.randint(1, 8),
            running_count=rng.randint(0, 3),
            actual_free=free,
        )


# --------------------------------------------------------------------------- #
# runtime lifecycle: conservation through run_parallel itself
#
# The pure allocator is only half the contract. These drive the real run loop
# with stand-in subprocesses so that every terminal path -- pass, fail, runtime
# skip, retry -- is shown to return exactly what it reserved, on every member of
# a gang. No GPU, no pynvml, no real pytest child.
# --------------------------------------------------------------------------- #
class _FakeProc:
    """Stand-in for the pytest child: finishes immediately with a fixed result."""

    def __init__(self, returncode: int, output: str, env: dict):
        self.returncode = returncode
        self.env = env
        self.stdout = io.StringIO(output)

    def poll(self):
        return self.returncode


class _RunHarness:
    """Wires run_parallel to fake GPUs and fake subprocesses, and records both
    every launch and the GPU accounting at each reserve/release."""

    def __init__(self, monkeypatch, tmp_path, gpu_totals_gib, outcomes):
        self.launches: list[dict] = []
        self.ledger: list[tuple[str, str, tuple[int, ...], dict]] = []
        self._outcomes = outcomes
        self._attempts: dict[str, int] = {}
        ppg = pytest_parallel_gpu

        monkeypatch.setattr(
            ppg,
            "detect_gpus",
            lambda: [
                {"index": i, "name": "fake", "total_mib": int(t * 1024)}
                for i, t in enumerate(gpu_totals_gib)
            ],
        )
        monkeypatch.setattr(ppg, "effective_cpu_budget", lambda: 64)
        monkeypatch.setattr(ppg, "_get_gpu_used_gib", lambda gi=0: 0.0)
        self.sleeps: list[float] = []
        monkeypatch.setattr(ppg.time, "sleep", self._sleep)
        monkeypatch.setattr(ppg, "_JUNIT_DIR", str(tmp_path / "junit"))
        monkeypatch.setattr(ppg, "_JUNIT_COMBINED", str(tmp_path / "junit" / "c.xml"))
        monkeypatch.setattr(
            ppg, "_parse_junit_skipped", lambda path: self._skip_reason(path)
        )
        monkeypatch.setattr(ppg.subprocess, "Popen", self._popen)

        real_reserve, real_release = ppg._reserve_gpus, ppg._release_gpus

        def reserve(test, gpus, gpu_states):
            real_reserve(test, gpus, gpu_states)
            self.ledger.append(
                ("reserve", test.id, test.assigned_gpus, self._state(gpu_states))
            )

        def release(test, gpu_states):
            held = test.assigned_gpus
            real_release(test, gpu_states)
            if held:
                self.ledger.append(("release", test.id, held, self._state(gpu_states)))

        monkeypatch.setattr(ppg, "_reserve_gpus", reserve)
        monkeypatch.setattr(ppg, "_release_gpus", release)

    def _sleep(self, seconds=0.0):
        # Never actually sleep -- just record what the scheduler asked for, so
        # the vLLM launch stagger is observable without slowing the suite.
        self.sleeps.append(seconds)

    @staticmethod
    def _state(gpu_states):
        return {
            gi: (round(gs.budget_used, 6), gs.running_count)
            for gi, gs in gpu_states.items()
        }

    def _test_id_from_cmd(self, cmd):
        # `_launch_test` puts the node id straight after `-m pytest`.
        return cmd[cmd.index("pytest") + 1]

    def _skip_reason(self, junit_path):
        for tid, spec in self._outcomes.items():
            safe = tid.replace("/", "_").replace("::", "__")
            if junit_path.endswith(f"{safe}.xml"):
                return spec.get("skip_reason")
        return None

    def _popen(self, cmd, env=None, **kwargs):
        tid = self._test_id_from_cmd(cmd)
        spec = self._outcomes[tid]
        attempt = self._attempts.get(tid, 0)
        self._attempts[tid] = attempt + 1
        self.launches.append(
            {
                "id": tid,
                "attempt": attempt,
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "env": env,
            }
        )
        rc = spec.get("returncodes", [spec.get("returncode", 0)])
        rc = rc[min(attempt, len(rc) - 1)]
        out = spec.get("outputs", [spec.get("output", "")])
        out = out[min(attempt, len(out) - 1)]
        return _FakeProc(rc, out, env)

    def assert_conserved(self):
        """Every reservation returned exactly once, nothing left held, no
        negative accounting anywhere along the way."""
        held: dict[str, tuple[int, ...]] = {}
        for action, tid, gpus, state in self.ledger:
            if action == "reserve":
                assert tid not in held, f"{tid} reserved twice without release"
                held[tid] = gpus
            else:
                assert tid in held, f"{tid} released without a reservation"
                assert held.pop(tid) == gpus, f"{tid} released a different gang"
            for gi, (budget, count) in state.items():
                assert budget >= -1e-9, f"GPU{gi} negative budget: {budget}"
                assert count >= 0, f"GPU{gi} negative process count: {count}"
        assert not held, f"reservations never returned: {held}"
        assert self.ledger, "harness recorded nothing"
        final = self.ledger[-1][3]
        assert all(
            budget == pytest.approx(0.0, abs=1e-9) and count == 0
            for budget, count in final.values()
        ), f"GPUs not back to idle: {final}"


def _meta(profiled, gpu_count=1, timeout=60):
    m = {"profiled_vram_gib": profiled, "timeout": timeout}
    if profiled > 0:
        m["requested_vllm_kv_cache_bytes"] = 1_000_000
    if gpu_count != 1:
        m["gpu_count"] = gpu_count
    return m


def _run(harness_outcomes, meta, monkeypatch, tmp_path, gpu_totals=(80.0, 80.0)):
    harness = _RunHarness(monkeypatch, tmp_path, gpu_totals, harness_outcomes)
    rc = pytest_parallel_gpu.run_parallel(
        test_ids=list(meta),
        meta=meta,
        max_vram_gib=80.0,
        num_slots=4,
    )
    return harness, rc


def test_gang_launch_sets_cuda_visible_devices_to_the_whole_gang(monkeypatch, tmp_path):
    meta = {"t.py::gang": _meta(5.0, gpu_count=2)}
    harness, rc = _run({"t.py::gang": {"returncode": 0}}, meta, monkeypatch, tmp_path)

    assert rc == 0
    assert len(harness.launches) == 1
    # The complete gang, ascending -- not a single index.
    assert harness.launches[0]["cuda_visible_devices"] == "0,1"
    harness.assert_conserved()


def test_single_gpu_launch_keeps_its_bare_index(monkeypatch, tmp_path):
    # gpu_1 regression: one device still renders as a bare index, not "0,".
    meta = {"t.py::single": _meta(5.0)}
    harness, rc = _run({"t.py::single": {"returncode": 0}}, meta, monkeypatch, tmp_path)

    assert rc == 0
    assert harness.launches[0]["cuda_visible_devices"] == "0"
    harness.assert_conserved()


def test_gang_passing_releases_every_member(monkeypatch, tmp_path):
    meta = {"t.py::gang": _meta(5.0, gpu_count=2), "t.py::single": _meta(5.0)}
    harness, rc = _run(
        {"t.py::gang": {"returncode": 0}, "t.py::single": {"returncode": 0}},
        meta,
        monkeypatch,
        tmp_path,
    )

    assert rc == 0
    harness.assert_conserved()


def test_gang_failing_releases_every_member(monkeypatch, tmp_path):
    meta = {"t.py::gang": _meta(5.0, gpu_count=2)}
    harness, rc = _run(
        {"t.py::gang": {"returncode": 1, "output": "E   assert False"}},
        meta,
        monkeypatch,
        tmp_path,
    )

    assert rc == 1  # the failure is reported...
    harness.assert_conserved()  # ...and both GPUs still came back


def test_gang_runtime_skip_releases_every_member(monkeypatch, tmp_path):
    # Exit code 0 with a skipped testcase in the JUnit XML: the run loop
    # reclassifies it as SKIPPED, and that path must release too.
    meta = {"t.py::gang": _meta(5.0, gpu_count=2)}
    harness, rc = _run(
        {"t.py::gang": {"returncode": 0, "skip_reason": "needs 2 H100s"}},
        meta,
        monkeypatch,
        tmp_path,
    )

    assert rc == 0
    harness.assert_conserved()


def test_gang_retry_releases_and_reacquires_every_member(monkeypatch, tmp_path):
    # First attempt trips a retryable init marker; the gang must be released in
    # full before requeueing, then reserved again for the retry.
    meta = {"t.py::gang": _meta(5.0, gpu_count=2)}
    harness, rc = _run(
        {
            "t.py::gang": {
                "returncodes": [1, 0],
                "outputs": ["Error in memory profiling: boom", ""],
            }
        },
        meta,
        monkeypatch,
        tmp_path,
    )

    assert rc == 0
    assert len(harness.launches) == 2, "test should have been retried once"
    assert [x["cuda_visible_devices"] for x in harness.launches] == ["0,1", "0,1"]
    # reserve, release (retry), reserve, release (pass) -- no leak in between.
    assert [entry[0] for entry in harness.ledger] == [
        "reserve",
        "release",
        "reserve",
        "release",
    ]
    harness.assert_conserved()


def test_mixed_gang_and_single_workload_conserves_and_does_not_overlap(
    monkeypatch, tmp_path
):
    meta = {
        "t.py::gang": _meta(20.0, gpu_count=2, timeout=1800),
        "t.py::s1": _meta(20.0, timeout=600),
        "t.py::s2": _meta(20.0, timeout=600),
        "t.py::filler": _meta(0.0, timeout=600),
    }
    outcomes = {tid: {"returncode": 0} for tid in meta}
    harness, rc = _run(outcomes, meta, monkeypatch, tmp_path)

    assert rc == 0
    assert len(harness.launches) == 4
    gang = next(x for x in harness.launches if x["id"] == "t.py::gang")
    assert gang["cuda_visible_devices"] == "0,1"
    harness.assert_conserved()


@pytest.mark.timeout(60)
def test_impossible_gang_is_rejected_before_anything_launches(monkeypatch, tmp_path):
    # One GPU, a gpu_2 test: the run must fail fast with a diagnosis instead of
    # waiting forever for a second device to appear.
    meta = {"t.py::gang": _meta(5.0, gpu_count=2)}
    harness, rc = _run(
        {"t.py::gang": {"returncode": 0}},
        meta,
        monkeypatch,
        tmp_path,
        gpu_totals=(80.0,),
    )

    assert rc == 1
    assert harness.launches == [], "nothing should have been launched"


@pytest.mark.timeout(60)
def test_oversized_single_gpu_test_is_rejected_before_anything_launches(
    monkeypatch, tmp_path
):
    # The same detector covers the pre-existing gpu_1 case: profiled larger
    # than any card used to spin the run loop until the CI timeout.
    meta = {"t.py::huge": _meta(200.0)}
    harness, rc = _run({"t.py::huge": {"returncode": 0}}, meta, monkeypatch, tmp_path)

    assert rc == 1
    assert harness.launches == []


# --------------------------------------------------------------------------- #
# status output
# --------------------------------------------------------------------------- #
def _running_entry(test, start_time=0.0):
    return pytest_parallel_gpu._RunningTest(
        proc=None, test=test, start_time=start_time  # type: ignore[arg-type]
    )


def test_status_shows_a_gang_on_every_gpu_it_occupies():
    # A gpu_2 worker occupies both cards, so it must be listed under both. If
    # it appeared under only one, the operator reading the log would think the
    # other card was idle and that the test was single-GPU.
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0), 2: _gpu(2, 80.0)}
    gang = _t("gang", 5.0, gpus=2)
    _reserve_gpus(gang, (0, 1), gpus)
    running = {7: _running_entry(gang)}

    lines = _status_lines(30.0, 0.0, gpus, running, lambda gi: 5.0)

    assert "[w7(30s)]" in lines[0], lines[0]  # GPU0
    assert "[w7(30s)]" in lines[1], lines[1]  # GPU1
    assert "[w" not in lines[2], lines[2]  # GPU2 genuinely idle


def test_status_lists_single_and_multi_gpu_workers_together():
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    gang = _t("gang", 5.0, gpus=2)
    single = _t("single", 5.0)
    _reserve_gpus(gang, (0, 1), gpus)
    _reserve_gpus(single, (1,), gpus)
    running = {0: _running_entry(gang), 1: _running_entry(single)}

    lines = _status_lines(10.0, 0.0, gpus, running, lambda gi: 10.0)

    assert "[w0(10s)]" in lines[0]  # GPU0: the gang only
    assert "w0(10s), w1(10s)" in lines[1]  # GPU1: gang + single, sorted by w_id


def test_status_omits_workers_from_idle_gpus():
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    single = _t("single", 5.0)
    _reserve_gpus(single, (0,), gpus)

    lines = _status_lines(5.0, 0.0, gpus, {3: _running_entry(single)}, lambda gi: 1.0)

    assert "[w3(5s)]" in lines[0]
    assert "[w" not in lines[1]


# --------------------------------------------------------------------------- #
# vLLM launch stagger across a gang
#
# vLLM's memory-profiling step snapshots free VRAM at startup, so two vLLM
# processes must not start on the same device within the stagger window
# (bug #10643). For a gang that means *every* device it touches, in both
# directions: it waits on the most recent launch on any member, and it stamps
# all of them when it starts.
# --------------------------------------------------------------------------- #
def _staggered(sleeps):
    """Sleeps that look like the stagger rather than the 1s poll."""
    return [d for d in sleeps if 1.5 < d <= pytest_parallel_gpu._VLLM_LAUNCH_STAGGER_S]


def test_gang_waits_on_a_recent_launch_on_any_member(monkeypatch, tmp_path):
    # GPU1 is the roomier card, so the single-GPU test best-fits onto it and
    # launches first. The gang then takes (0, 1) -- its *second* member is the
    # one that just saw a vLLM start, so it must still wait out the window.
    meta = {
        "t.py::single": _meta(5.0, timeout=1800),  # longest -> launches first
        "t.py::gang": _meta(5.0, gpu_count=2, timeout=600),
    }
    outcomes = {tid: {"returncode": 0} for tid in meta}
    harness, rc = _run(outcomes, meta, monkeypatch, tmp_path, gpu_totals=(40.0, 80.0))

    assert rc == 0
    single = next(x for x in harness.launches if x["id"] == "t.py::single")
    gang = next(x for x in harness.launches if x["id"] == "t.py::gang")
    assert single["cuda_visible_devices"] == "1"
    assert gang["cuda_visible_devices"] == "0,1"
    assert _staggered(
        harness.sleeps
    ), "gang launched without waiting out the stagger on GPU1"


def test_gang_stamps_the_stagger_clock_on_every_member(monkeypatch, tmp_path):
    # The gang launches first and takes both cards. The following single-GPU
    # vLLM test best-fits onto GPU1 -- again the gang's second member -- so it
    # must wait, which only happens if the gang stamped both devices.
    meta = {
        "t.py::gang": _meta(5.0, gpu_count=2, timeout=1800),  # launches first
        "t.py::single": _meta(5.0, timeout=600),
    }
    outcomes = {tid: {"returncode": 0} for tid in meta}
    harness, rc = _run(outcomes, meta, monkeypatch, tmp_path, gpu_totals=(40.0, 80.0))

    assert rc == 0
    assert harness.launches[0]["id"] == "t.py::gang"
    single = next(x for x in harness.launches if x["id"] == "t.py::single")
    assert single["cuda_visible_devices"] == "1"
    assert _staggered(
        harness.sleeps
    ), "the gang did not stamp GPU1, so the next vLLM launch there did not wait"


def test_non_vllm_tests_are_not_staggered(monkeypatch, tmp_path):
    # The stagger exists for vLLM's profiling race only. A gang with no vLLM KV
    # request must not pay it.
    meta = {
        "t.py::a": {"profiled_vram_gib": 0.0, "timeout": 600, "gpu_count": 2},
        "t.py::b": {"profiled_vram_gib": 0.0, "timeout": 600},
    }
    outcomes = {tid: {"returncode": 0} for tid in meta}
    harness, rc = _run(outcomes, meta, monkeypatch, tmp_path, gpu_totals=(40.0, 80.0))

    assert rc == 0
    assert _staggered(harness.sleeps) == []


# --------------------------------------------------------------------------- #
# reservation eligibility -- a gang may only reserve GPUs it could run on
#
# Found by clean-room review of the first gang implementation. The blocked-gang
# reservation ranked candidate devices by headroom magnitude but never applied
# the eligibility THRESHOLD the launch scan applies, so a device that can never
# host the gang -- too small, or holding memory from outside this run -- scored
# maximum headroom precisely BECAUSE it was idle, won a protector slot, and left
# a genuinely usable device unprotected to saturate. Reserving R outside the
# eligible set E is fictitious protection: it spends one of the `k` all-or-none
# slots and buys no path to launch.
# --------------------------------------------------------------------------- #
def test_gang_does_not_reserve_a_card_that_is_too_small_to_ever_host_it():
    """The tightest statement of the invariant, in a single pass.

    GPU0 (10 GiB, 6.0 committed to one process) can host the gang once it
    drains. GPU1 (10 GiB, idle) can host it now. GPU2 (5 GiB) can NEVER: 5.0 is
    below the gang's 6.0 per-device requirement.

    The gang needs two devices and only GPU1 qualifies, so it blocks and
    reserves. Ranking by headroom alone puts GPU2 (5.0 free) above GPU0 (2.5
    free), so the hold lands on {1, 2} and leaves GPU0 -- the only other card
    that could ever run the gang -- open for backfill.

    Observable consequence: the 2.0 GiB filler. If GPU0 is left unprotected the
    filler lands there and pushes the card further from hosting the gang. Under
    a correct reservation the hold is {0, 1}, the filler is refused on both, and
    it goes to GPU2, which is where work belongs -- GPU2 is useless to the gang
    but perfectly good for a filler.
    """
    gpus = {
        0: _gpu(0, 10.0, budget_used=6.0, running_count=1),
        1: _gpu(1, 10.0),
        2: _gpu(2, 5.0),
    }
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    filler = _t("filler", 3.0, timeout=15.0)
    assert _unschedulable_reason(gang, gpus) is None, "gang must be feasible"

    launches = _select_checked([gang, filler], gpus, num_slots=8, running_count=1)

    # Ranked by headroom, GPU2 (5.0 idle) outranks GPU0 (8.5 - 6.0 = 2.5), so an
    # unfiltered reservation takes {1, 2} and leaves GPU0 open. The filler is
    # 3.0: too big for what GPU0 has left, and refused on both held cards, so
    # under the unfiltered rule nothing launches at all and the gang's hold is
    # sitting on a card it can never use. Filtered, the hold is {0, 1} and GPU2
    # -- useless to the gang, fine for a filler -- takes the work.
    assert launches == [(1, (2,))], (
        "the gang must hold the two cards that can actually host it (0 and 1), "
        "leaving the 5 GiB card free for backfill; instead the hold landed on "
        "the card that can never host the gang"
    )


def test_gang_is_not_starved_by_a_card_too_small_to_ever_host_it():
    """Cross-pass form of the same defect: the gang must still launch.

    Identical to test_blocked_gang_is_not_starved_by_endless_lower_priority_backfill
    except for a third card of 5.9 GiB -- below the gang's 6.0 requirement, so it
    can never host it -- and a filler lifetime long enough that the unprotected
    card never drains between passes.
    """
    gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0), 2: _gpu(2, 5.9)}
    hog = _t("hog", 3.0, timeout=120)  # est 40 passes
    gang = _t("gang", 6.0, timeout=90, gpus=2)
    assert _unschedulable_reason(gang, gpus) is None

    started = _drive_passes(
        gpus,
        [hog, gang],
        backfill=lambda i: _t(f"fill{i}", 2.0, timeout=21.0),
        passes=400,
    )

    assert "gang" in started, (
        "feasible gpu_2 gang never launched in 400 passes on a node with two "
        "cards big enough for it -- starved by a reservation placed on the card "
        "that is too small to ever host it"
    )
    assert started["gang"] <= 41, started["gang"]


@pytest.mark.parametrize("third", [5.0, 5.5, 5.9, 6.0, 10.0])
def test_gang_progress_does_not_depend_on_an_unusable_third_card(third):
    """Sweep the third card across the feasibility boundary (requirement 6.0).

    Below 6.0 the card cannot host the gang; at or above it, it can. Either way
    two 10 GiB cards are present, so the gang is feasible throughout and must
    launch as soon as the hog retires. Progress must not depend on the size of a
    card the gang does not need.
    """
    gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0), 2: _gpu(2, third)}
    hog = _t("hog", 3.0, timeout=120)
    gang = _t("gang", 6.0, timeout=90, gpus=2)
    assert _unschedulable_reason(gang, gpus) is None

    started = _drive_passes(
        gpus,
        [hog, gang],
        backfill=lambda i: _t(f"fill{i}", 2.0, timeout=21.0),
        passes=400,
    )

    assert "gang" in started, f"starved with third card = {third} GiB"
    assert started["gang"] <= 41, (third, started["gang"])


def test_making_a_gpu_larger_never_delays_a_gang():
    """Monotonicity: growing a card must not make scheduling worse.

    A physically nonsensical result -- enlarging GPU2 from 5.0 to 5.9 turning a
    launching gang into a starving one -- is exactly what the unfiltered
    reservation ranking produced, because a larger idle card outranks a smaller
    one for a protector slot it can still never honour.
    """

    def start_pass(third):
        gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0), 2: _gpu(2, third)}
        started = _drive_passes(
            gpus,
            [_t("hog", 3.0, timeout=120), _t("gang", 6.0, timeout=90, gpus=2)],
            backfill=lambda i: _t(f"fill{i}", 2.0, timeout=21.0),
            passes=400,
        )
        return started.get("gang")

    sizes = [5.0, 5.5, 5.9, 6.0, 10.0]
    starts = [start_pass(s) for s in sizes]

    assert all(s is not None for s in starts), dict(zip(sizes, starts))
    for (small, a), (large, b) in zip(zip(sizes, starts), list(zip(sizes, starts))[1:]):
        assert b <= a, (
            f"growing GPU2 from {small} to {large} GiB delayed the gang "
            f"from pass {a} to pass {b}"
        )


def test_foreign_memory_on_a_card_the_gang_does_not_need_never_delays_it():
    """Progress must not depend on a card the gang has no need of.

    The physical-size filter was not the whole of the defect it closed. A hold is
    useless on any card the gang cannot occupy, and physical size is only one of
    the two reasons a card can be unoccupiable -- memory held by a process
    OUTSIDE this run is the other, and it produces the identical failure through
    the identical mechanism: an idle card scores its whole capacity for a
    protector slot precisely BECAUSE it is idle, wins one, and can never honour
    it.

    Three 10 GiB cards, a gpu_2 gang needing 6.0, and 4.5 GiB held forever on
    GPU2 by something outside this run. GPU0 and GPU1 are free, permanently, and
    two cards is exactly what the gang needs -- so it must launch as soon as the
    hog retires. Before this was fixed the hold landed on {1, 2}, GPU0 was left
    unprotected, the backfill took it, and the gang never launched in 400 passes.
    """
    gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0), 2: _gpu(2, 10.0)}
    gang = _t("gang", 6.0, timeout=90, gpus=2)
    assert _unschedulable_reason(gang, gpus) is None

    started = _drive_passes(
        gpus,
        [_t("hog", 3.0, timeout=120), gang],
        backfill=lambda i: _t(f"fill{i}", 2.0, timeout=21.0),
        passes=400,
        external_hold=lambda now: {2: 4.5},
    )

    assert "gang" in started, (
        "gang starved by a hold placed on the one card it can never occupy -- "
        "two permanently free cards big enough for it were available throughout"
    )
    assert started["gang"] <= 41, started["gang"]


@pytest.mark.parametrize("frees_at", [None, 60])
def test_gang_waits_only_while_foreign_memory_leaves_too_few_usable_cards(frees_at):
    """The honest remaining limit, pinned in both directions.

    Preferring usable cards for a hold cannot manufacture capacity. When foreign
    memory blocks so many cards that fewer than ``gpu_count`` remain, the gang
    genuinely cannot run and waiting is correct -- that is a property of the box,
    not of this scheduler. What it must NOT do is lose its place: the moment the
    neighbour releases, the gang launches on that very pass, which is only true
    if the holds were positioned on the cards that came back.

    Two of three cards blocked, so exactly one is usable against a need of two.
    """
    gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0), 2: _gpu(2, 10.0)}
    gang = _t("gang", 6.0, timeout=90, gpus=2)
    assert _unschedulable_reason(gang, gpus) is None

    started = _drive_passes(
        gpus,
        [_t("hog", 3.0, timeout=120), gang],
        backfill=lambda i: _t(f"fill{i}", 2.0, timeout=21.0),
        passes=400,
        external_hold=(
            lambda now: {1: 4.5, 2: 4.5} if (frees_at is None or now < frees_at) else {}
        ),
    )

    if frees_at is None:
        assert "gang" not in started, (
            "only one card was ever usable and the gang needs two -- launching "
            "would mean the memory accounting is wrong, not that it made progress"
        )
    else:
        assert started.get("gang") == frees_at, (
            f"gang should launch the pass the foreign memory is released "
            f"({frees_at}), got {started.get('gang')} -- it was not holding the "
            "cards that came back"
        )


def test_a_gang_hold_is_denominated_in_the_units_the_gang_is_gated_in():
    """A hold must bound backfill by what the gang has to survive, not by our share.

    The hold gate counts only budget WE committed (`ts.budget + profiled <=
    budget_multi - required`), but the gate the gang must itself pass counts
    OBSERVED usage (`total - free + required <= cap`), which includes memory
    held outside this run. The two are the same number only when that foreign
    hold is zero. For any F > 0 the hold therefore admits backfill sitting
    exactly in the band the gang cannot survive -- a band `required` wide.

    Two 10 GiB cards (budget_multi 8.5), a gpu_2 gang needing 6.0, 1.0 GiB held
    on each card from outside the run. GPU0 also carries 3.0 GiB of our own
    work, so the gang is blocked and holds both cards. A 2.0 GiB filler clears
    the old gate on GPU1 (0.0 + 2.0 <= 8.5 - 6.0) -- and the moment it lands the
    gang cannot collect GPU1 either, because 1.0 + 2.0 + 6.0 > 8.5. Observed,
    as everywhere else in this file, through what the hold refuses.
    """
    gpus = {
        0: _gpu(0, 10.0, budget_used=3.0, running_count=1),
        1: _gpu(1, 10.0),
    }
    # 1.0 GiB on each card belongs to a process outside this run.
    actual_free = {0: 10.0 - 3.0 - 1.0, 1: 10.0 - 1.0}
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    filler = _t("filler", 2.0, timeout=15.0)
    assert _unschedulable_reason(gang, gpus) is None
    # The card is usable in its own right: 1.0 GiB of foreign memory leaves
    # 9.0 free on a card that needs to fit 6.0, so condition 2 holds and the
    # gang is squarely inside the domain the progress property claims.
    assert 10.0 - 1.0 >= gang.profiled_gib

    launches = _select_checked(
        [gang, filler], gpus, num_slots=8, running_count=1, actual_free=actual_free
    )

    assert launches == [], (
        "the hold admitted a filler that leaves the reserved card unable to "
        "host a member: 1.0 foreign + 2.0 filler + 6.0 gang = 9.0 > 8.5. The "
        "hold is gated in our-budget units and the gang in observed-usage units"
    )


def test_a_free_reading_that_overshoots_cannot_loosen_a_gang_hold():
    """The negative clamp on the foreign-hold term is load bearing in the gate.

    `free + budget > total` is a real, ordinary state: a test is launched and
    charged to the budget before it has allocated anything, so the live reading
    still shows the card almost empty. `total - free - budget` then goes
    NEGATIVE, and without the clamp the hold's line reads
    `budget_multi - (-x) - required` -- wider than the unheld card, so the hold
    admits MORE the emptier the card looks.

    GPU0 carries 1.0 GiB of budget for a test that has not ramped yet (free
    still reads the whole 10.0), and GPU1 is too full for a member, so the gang
    is blocked and holds both. A 2.0 GiB filler must not land on GPU0:
    1.0 + 2.0 = 3.0 is already past `8.5 - 6.0`. Unclamped the term is -1.0,
    the line becomes 3.5, and the filler is let in.

    This clamp used to be an equivalent mutant -- reachable only behind
    `_can_ever_host`, which excludes the regime where it bites. Sharing the term
    with the admission gate, which sits behind no such filter, made it real.
    """
    gpus = {
        0: _gpu(0, 10.0, budget_used=1.0, running_count=1),
        1: _gpu(1, 10.0, budget_used=5.0, running_count=1),
    }
    # GPU0: charged 1.0 GiB, has allocated none of it yet -> free + budget > total.
    actual_free = {0: 10.0, 1: 5.0}
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    filler = _t("filler", 2.0, timeout=15.0)
    assert _unschedulable_reason(gang, gpus) is None
    assert actual_free[0] + gpus[0].budget_used > gpus[0].total_gib

    launches = _select_checked(
        [gang, filler], gpus, num_slots=8, running_count=2, actual_free=actual_free
    )

    assert launches == [], (
        "an un-ramped test made the card read emptier than empty and widened "
        "the hold: 1.0 committed + 2.0 filler = 3.0 is past the 2.5 line"
    )


def test_gang_progress_survives_foreign_memory_that_leaves_every_card_usable():
    """The multi-pass consequence of the gate above, and the reason it matters.

    Identical to `test_blocked_gang_is_not_starved_by_endless_lower_priority_backfill`
    except that 1.0 GiB is held on every card from outside the run and the
    backfill durations do not all divide the hog's lifetime -- a single fill
    duration makes every card fall clean on the same pass, which hides this.

    Every card reads usable (10.0 - 1.0 >= 6.0), so all five progress
    conditions hold. With the hold gated correctly the gang launches the pass
    the hog retires. With it gated in our-budget units the hold bounds nothing:
    each reserved card carries a filler the gang cannot fit beside, the gang
    needs all of them clean on the same pass, and it waits on a coincidence
    whose odds fall geometrically in `gpu_count` -- 13 of 16 seeds never
    launched in 20,000 passes at gpu_count=4.
    """
    gpus = {i: _gpu(i, 10.0) for i in range(3)}
    hogs = [_t(f"hog{i}", 3.0, timeout=127) for i in range(2)]  # est 42 passes
    gang = _t("gang", 6.0, timeout=90, gpus=3)  # needs every card
    assert _unschedulable_reason(gang, gpus) is None
    # Durations 11/13/17/19/23 passes: coprime with each other and with 42, so
    # the cards drift out of phase and no coincidence is handed to the gang.
    durations = itertools.cycle([33.0, 39.0, 51.0, 57.0, 69.0])

    started = _drive_passes(
        gpus,
        hogs + [gang],
        backfill=lambda i: _t(f"fill{i}", 2.0, timeout=next(durations)),
        passes=4000,
        external_hold={i: 1.0 for i in range(3)},
    )

    assert "gang" in started, (
        "gang never launched in 4000 passes while every card was usable and "
        "lower-priority backfill kept launching -- starved"
    )
    assert started["gang"] <= 43, (
        f"gang launched on pass {started['gang']}, not the pass the hogs "
        f"retired (42) -- backfill was let in past the hold ahead of it"
    )


def test_a_foreign_blocked_card_never_outranks_a_usable_one_for_a_hold():
    """The ranking, isolated from the multi-pass driver.

    Headroom alone is the wrong sort key for a gang hold, and an idle card is
    exactly where it misleads: ``_cap`` of an idle card is the whole card, so a
    card carrying 4.5 GiB of somebody else's memory still scores a perfect 10.0
    and takes a protector slot off a card that is genuinely free.

    GPU0 is busy with our own work -- which will finish -- and GPU1/GPU2 are
    idle, but GPU2 is holding foreign memory. The hold has to land on {0, 1}: the
    two cards a 6.0 GiB gang member could occupy once our own test retires.
    Observed, as everywhere else in this file, through what the hold refuses --
    the filler can only land on the one card not protected.
    """
    gpus = {
        0: _gpu(0, 10.0, budget_used=5.0, running_count=1),
        1: _gpu(1, 10.0),
        2: _gpu(2, 10.0),
    }
    # GPU2 reads 5.5 free while committing no budget of ours: 4.5 GiB is held by
    # a process outside this run, so it cannot take a 6.0 GiB member today.
    actual_free = {0: 5.0, 1: 10.0, 2: 5.5}
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    filler = _t("filler", 3.0, timeout=15.0)
    assert _unschedulable_reason(gang, gpus) is None

    launches = _select_checked(
        [gang, filler], gpus, num_slots=8, running_count=1, actual_free=actual_free
    )

    assert launches == [(1, (2,))], (
        "the hold went to a card holding another process's memory instead of a "
        "card the gang could actually occupy; the filler proves which card was "
        "left unprotected"
    )


def test_gpu1_control_is_unaffected_by_an_unusable_card():
    """Control: the same topology must not regress single-GPU scheduling.

    A gpu_1 test only ever needs one card to come free and any card will do, so
    the unfiltered reservation never cost it progress. This pins that the repair
    is about gangs and that gpu_1 behaviour is unchanged by it.
    """
    gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0), 2: _gpu(2, 5.9)}
    started = _drive_passes(
        gpus,
        [_t("hog", 3.0, timeout=120), _t("blocked", 6.0, timeout=90)],
        backfill=lambda i: _t(f"fill{i}", 2.0, timeout=21.0),
        passes=400,
    )

    assert "blocked" in started
    assert started["blocked"] <= 41, started["blocked"]


def test_gang_still_holds_its_cards_while_every_card_is_transiently_blocked():
    """The filter must not be able to REMOVE protection a gang already needed.

    Two 16 GiB cards (budget_multi 13.6) and a gang wanting 12.0 on each. Until
    pass 40 a neighbour holds 5.0 GiB on both, so neither card can host the gang
    yet -- 5.0 + 12.0 > 16.0 -- and no card passes an eligibility filter. The
    neighbours then leave and both cards become usable.

    Treating the filter as an admission test drops the reservation entirely on
    exactly the passes it is needed: the cards fill with 3.0 GiB backfill that
    outlives the neighbours, and the gang -- which the pre-repair code launched
    the moment they left -- waits for the backfill to drain instead. A hold is
    protection against the future; it must not be revoked because of one live
    reading of the present.
    """
    gpus = {0: _gpu(0, 16.0), 1: _gpu(1, 16.0)}
    gang = _t("gang", 12.0, timeout=1800, gpus=2)
    assert _unschedulable_reason(gang, gpus) is None

    started = _drive_passes(
        gpus,
        [gang],
        backfill=lambda i: _t(f"fill{i}", 3.0, timeout=600.0),
        passes=400,
        external_hold=lambda now: {0: 5.0, 1: 5.0} if now < 40 else {},
    )

    assert "gang" in started, "gang never launched after its blockers left"
    assert started["gang"] <= 60, (
        f"gang launched on pass {started['gang']}, long after the neighbours "
        "left on pass 40 -- its cards were given away while it was unprotected"
    )


def test_non_gang_work_still_launches_while_a_gang_holds_cards_it_cannot_use():
    """A hold must not suppress work that cannot affect the gang's admission.

    The suite has never asserted this. Every existing observer asks "did the
    watched gang launch", so a hold that takes the rest of the node to zero
    scores a clean pass -- which is how this reached a third audit.

    Two 80 GiB cards (``budget_multi`` 68.0) and a gpu_2 gang wanting 40.0 on
    each. Until pass 60 a process outside the run holds 45.0 GiB on both, so
    neither card can host the gang: ``_usable_now`` reads 80.0 - 45.0 = 35.0,
    short of 40.0. On such a card the hold line is
    ``budget_multi - foreign - required`` = 68.0 - 45.0 - 40.0 = **-17.0**,
    negative unconditionally, so every backfill test is refused for as long as
    the gang is queued. At ``n == gpu_count`` -- the only topology this ships to
    -- that is the whole node, and ``run_parallel``'s ``while pending or
    running:`` has no bail-out from it.

    The 10.0 GiB backfill here is *causally irrelevant* to the gang: once the
    neighbours leave, a running filler leaves 10.0 + 40.0 = 50.0 against a cap of
    68.0, so the gang still fits beside it. Refusing it buys the gang nothing --
    and the second assertion proves that directly, by pinning that the gang
    launches on the same pass either way.

    This is deliberately NOT the topology of
    ``test_gang_still_holds_its_cards_while_every_card_is_transiently_blocked``
    above, where the backfill outlives the neighbours and genuinely does cost the
    gang its launch. Both must hold at once: protect what the gang needs, refuse
    nothing else.
    """
    gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
    gang = _t("gang", 40.0, timeout=1800, gpus=2)
    assert _unschedulable_reason(gang, gpus) is None

    started = _drive_passes(
        gpus,
        [gang],
        backfill=lambda i: _t(f"fill{i}", 10.0, timeout=15.0),
        passes=300,
        external_hold=lambda now: {0: 45.0, 1: 45.0} if now < 60 else {},
    )

    held_window = [n for n, at in started.items() if n != "gang" and at < 60]
    assert held_window, (
        "no non-gang test launched in the 60 passes the gang held both cards it "
        "could not use -- the node made zero progress while waiting"
    )

    assert "gang" in started, "gang never launched after its blocker retired"
    assert started["gang"] <= 61, (
        f"gang launched on pass {started['gang']}; admitting backfill the gang "
        "cannot be harmed by must not delay it"
    )


def test_a_gang_hold_prefers_usable_cards_whatever_order_the_gpus_are_declared_in():
    """The hold must rank by usability, not by however ``gpu_states`` was built.

    Deleting ``unreserved.sort(key=_hold_key)`` -- the whole of the repair that
    exists because a hold landed on a card nobody in this run could use -- leaves
    all 100 other tests green. Every one of them happens to declare its GPUs in
    an order where the correct hold set is already the first ``need`` entries, so
    the sort is a no-op and its absence is invisible.

    Three 10 GiB cards (``budget_multi`` 8.5) **declared 2, 1, 0**. A process
    outside the run holds 4.5 GiB on GPU2 forever, so ``_usable_now`` reads
    10.0 - 4.5 = 5.5 against the gang's 6.0: GPU2 cannot host a member today and
    will not until that neighbour leaves. GPU0 carries a 5.0 GiB test of *ours*,
    which retires -- ``_foreign_held`` is 0 there, so GPU0 is usable. The gang is
    blocked meanwhile, needing two cards and having one.

    The hold must therefore protect {GPU0, GPU1}. Taking ``unreserved[:need]`` in
    declaration order protects {GPU2, GPU1} instead: GPU2 buys the gang nothing,
    and GPU0 -- unprotected -- fills with backfill that renews before the hog
    retires, so the gang never gets a second card. The assertion is on the
    launch, not on the ordering: this is what the missing sort *does*, not that
    it is called.
    """
    gpus = {gi: _gpu(gi, 10.0) for gi in (2, 1, 0)}
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    hog = _t("hog", 5.0, timeout=60)
    assert _unschedulable_reason(gang, gpus) is None

    # The hog is already running when the first pass is scheduled, so the gang
    # starts blocked and the hold's choice of cards is what decides its fate.
    _reserve_gpus(hog, (0,), gpus)
    running: dict[str, tuple[_TestEntry, int]] = {"hog": (hog, 60)}
    pending = [gang]
    started: dict[str, int] = {}
    made = 0

    for now in range(600):
        for name in [n for n, (_, end) in running.items() if end <= now]:
            _release_gpus(running.pop(name)[0], gpus)
        while len(pending) < 4:
            pending.append(_t(f"fill{made}", 3.0, timeout=21.0))
            made += 1
        pending.sort(key=_priority_key, reverse=True)
        resident = {
            gi: sum(
                t.profiled_gib for t, _ in running.values() if gi in t.assigned_gpus
            )
            for gi in gpus
        }
        actual_free = {
            gi: gs.total_gib - resident[gi] - (4.5 if gi == 2 else 0.0)
            for gi, gs in gpus.items()
        }
        for idx, got in reversed(
            _select_launches(
                pending=pending,
                gpu_states=gpus,
                actual_free=actual_free,
                num_slots=8,
                running_count=len(running),
            )
        ):
            test = pending.pop(idx)
            _reserve_gpus(test, got, gpus)
            running[test.name] = (test, now + max(1, int(test.est_duration)))
            started.setdefault(test.name, now)

    assert "gang" in started, (
        "gang never launched in 600 passes: its hold went to GPU2, which no "
        "member can use, leaving usable GPU0 to be taken by backfill"
    )
    assert started["gang"] <= 61, (
        f"gang launched on pass {started['gang']}, not on the pass its own hog "
        "retired -- a usable card was given away while the gang waited"
    )


def test_a_gang_hold_admits_the_same_backfill_however_far_into_the_pass_it_is():
    """``_foreign_held`` must not drift as a pass fills. Nothing tested this.

    The helper's whole reason for existing is that the hold's *ranking* and the
    hold's *admission gate* must read one number; they were denominated
    differently once and that was defect 3. Its docstring states the mechanism --
    ``total - free - budget``, where the two terms cancel exactly against this
    pass's tentative launches -- and asserts in prose that the answer therefore
    does not drift. Substituting the pass-invariant ``gs.budget_used`` for the
    tentative ``ts.budget`` leaves all 101 other tests green while making the
    estimate climb by exactly one profile per test already admitted.

    Three 80 GiB cards (``budget_multi`` 68.0). GPU1 and GPU2 each carry 50.0 GiB
    held outside the run, so neither can host a member of a 40.0 GiB/device gang
    (80.0 - 50.0 = 30.0). GPU0 is clean and usable, so with one usable card and a
    gang needing two, the gang is blocked and holds cards.

    On GPU0 -- held, and usable, so the foreign term applies -- the line is
    ``budget_multi - foreign - required`` = 68.0 - 0.0 - 40.0 = **28.0 GiB**, and
    5.0 GiB backfill fits five times (25.0 <= 28.0 < 30.0). A drifting estimate
    reads foreign as 5.0 after the first admission, 10.0 after the second, and
    stops at three. The assertion is on how much work the card takes, not on how
    the number is computed.
    """
    gpus = {gi: _gpu(gi, 80.0) for gi in range(3)}
    gang = _t("gang", 40.0, timeout=1800, gpus=2)
    pending = [gang] + [_t(f"fill{i}", 5.0, timeout=20.0) for i in range(24)]
    pending.sort(key=_priority_key, reverse=True)

    launched = _select_launches(
        pending=pending,
        gpu_states=gpus,
        actual_free={0: 80.0, 1: 30.0, 2: 30.0},
        num_slots=64,
        running_count=0,
    )

    names = [pending[idx].name for idx, _ in launched]
    assert "gang" not in names, "gang must be blocked for its hold to be under test"

    on_gpu0 = sum(
        1 for idx, got in launched if pending[idx].name != "gang" and 0 in got
    )
    assert on_gpu0 == 5, (
        f"GPU0 took {on_gpu0} backfill tests, not the 5 its hold line admits "
        "(68.0 - 0.0 - 40.0 = 28.0 GiB, five 5.0 GiB tests = 25.0): the foreign "
        "estimate drifted upward as the pass filled, so later tests in the same "
        "pass were gated against a larger number than earlier ones"
    )


@pytest.mark.parametrize("filler_gib", [1.5, 2.0, 3.0])
def test_a_hold_survives_foreign_memory_retiring_part_way_rather_than_all_at_once(
    filler_gib,
):
    """Backfill admitted on an unusable card must not outlast the card's usefulness.

    Every other test in this file that models foreign memory retiring retires it to
    exactly ``{}``. Zero is the one value at which admitting up to
    ``budget_multi - required`` on a card the gang cannot use today stays sufficient,
    because the gang then needs only ``committed + required <= budget_multi``. For any
    residual ``F' > 0`` it needs ``F' + committed + required <= budget_multi``, which
    that bound does not provide.

    The dominant term is not the committed GiB. Admitting *any* VRAM test drives
    ``running_count`` from 0 to 1, which drops ``_cap`` from ``total_gib`` to
    ``budget_multi``. The card's feasibility threshold therefore moves from
    ``F' <= total_gib - required`` to ``F' <= budget_multi - committed - required`` --
    it loses the multi-process margin *as well as* what was admitted.

    Three 10 GiB cards (``budget_multi`` 8.5) and a gpu_2 gang wanting 6.0 on each.
    GPU0 is clean; GPU1 and GPU2 carry foreign memory that retires in stages rather
    than vanishing: 4.5 -> 3.0 -> 1.0 -> 0.0. With one usable card and a gang needing
    two, the gang is blocked and holds cards.

    The gang first fits at pass 40, when foreign falls to 3.0: an untouched card reads
    ``3.0 + 6.0 = 9.0`` against the whole-card cap of 10.0. It must launch there. If any
    test was admitted onto that card while it was unusable, the card instead reads
    ``3.0 + committed + 6.0`` against 8.5 and the gang waits for it to drain -- which is
    the regression this pins.

    The filler size is swept because the bound is a specific quantity, not merely a
    non-zero one. Here ``budget_multi - required - (total_gib - budget_multi)`` is
    ``8.5 - 6.0 - 1.5 = 1.0``, so every size below refuses. Reserving only *half* the
    margin would raise the line to 1.75 and admit the 1.5 GiB case, which then delays the
    gang to pass 60 -- a single fixed size would not tell the two rules apart.
    """
    gpus = {gi: _gpu(gi, 10.0) for gi in range(3)}
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    assert _unschedulable_reason(gang, gpus) is None

    def staged(now: int) -> dict[int, float]:
        if now < 40:
            held = 4.5
        elif now < 60:
            held = 3.0
        elif now < 80:
            held = 1.0
        else:
            held = 0.0
        return {1: held, 2: held}

    started = _drive_passes(
        gpus,
        [gang],
        backfill=lambda i: _t(f"fill{i}", filler_gib, timeout=1800.0),
        passes=400,
        external_hold=staged,
    )

    assert (
        "gang" in started
    ), "gang never launched, though from pass 40 onward two cards could host a member"
    assert started["gang"] <= 41, (
        f"gang launched on pass {started['gang']}, not on pass 40 when foreign memory "
        "first fell far enough for two cards to take a member. Backfill admitted while "
        "those cards were unusable both consumed capacity and cost them the "
        "whole-card cap, so the gang waited for it to drain instead"
    )


def test_unusable_card_admits_exactly_the_margin_reserved_line_and_no_more():
    """Unusable-card admission line is B - r - M, not B - r.

    2 x 80 GiB (B=68, M=12), gang r=40, foreign 45 until pass 60.
    Cards are unusable (80-45=35 < 40). L_unusable = 68-40-12 = 16.

    Threshold lock:
      * b = 16.0  is admitted while the gang cannot use the cards.
      * b = 16.01 is refused.

    b > L_unusable is a reachable scheduler state, not an impossible one.
    Refusing that filler is expected conservative behavior. Maximal work
    conservation for b > L_unusable (admit every filler that would still
    leave room for the gang after F' = 0) is not claimed. When foreign
    clears, the gang still launches in both cases.
    """

    def run(fill):
        gpus = {0: _gpu(0, 80.0), 1: _gpu(1, 80.0)}
        gang = _t("gang", 40.0, timeout=1800, gpus=2)
        return _drive_passes(
            gpus,
            [gang],
            backfill=lambda i, f=fill: _t(f"fill{i}", f, timeout=15.0),
            passes=200,
            external_hold=lambda now: {0: 45.0, 1: 45.0} if now < 60 else {},
        )

    at_line = run(16.0)
    over = run(16.01)
    early_at = [n for n, at in at_line.items() if n != "gang" and at < 60]
    early_over = [n for n, at in over.items() if n != "gang" and at < 60]
    assert early_at, (
        "16.0 GiB sits on L_unusable=16 and must be admitted while the gang "
        "cannot use the cards"
    )
    assert not early_over, (
        f"{early_over} launched before pass 60 at b=16.01 > L_unusable=16 — "
        "expected conservative refusal of filler above the unusable-card line"
    )
    assert "gang" in at_line and at_line["gang"] <= 61
    assert "gang" in over and over["gang"] <= 61


def test_known_limitation_residual_above_margin_with_on_line_filler_delays_gang():
    """F' > M is reachable; progress is not claimed in that region.

    2 x 10 GiB (B=8.5, M=1.5), gang r=6, L_unusable = 8.5-6-1.5 = 1.0.
    Foreign 4.5 until pass 60, then residual 3.0 (> M). A 1.0 GiB filler is
    admitted on the line. At pass 60 the empty-card arithmetic T-F'+r = 10
    would fit the gang; the occupied card reads 3+1+6 against cap 8.5 and
    does not. The gang waits for the filler.

    This is a known limitation / excluded guarantee region, not a scheduler
    bug and not an unreachable state. The progress claim assumes residual
    foreign memory eventually satisfies F' <= M. Pinning the delay here
    keeps a later observer from mistaking it for an undiscovered regression.
    """
    gpus = {0: _gpu(0, 10.0), 1: _gpu(1, 10.0)}
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    started = _drive_passes(
        gpus,
        [gang],
        backfill=lambda i: _t(f"fill{i}", 1.0, timeout=900.0),
        passes=400,
        external_hold=lambda now: {0: 4.5, 1: 4.5} if now < 60 else {0: 3.0, 1: 3.0},
    )
    assert "gang" in started, "limitation is delay, not a permanent freeze"
    assert started["gang"] > 60, (
        f"gang launched on pass {started['gang']}; F'=3.0 > M=1.5 with a 1.0 "
        "on-line filler is a known limitation outside the progress guarantee"
    )


# --------------------------------------------------------------------------- #
# the assignment has to survive into the WORKERS, not just be derived
#
# The tests above execute the script's device-derivation block. They cannot see
# what the two worker launches actually do with it: pinning both workers to the
# same device, or dropping the pin entirely, leaves the derivation untouched and
# every one of them green. That is the whole bug this script was changed to fix,
# so it gets an end-to-end check -- the real script, unmodified, with a stand-in
# for the engine.
# --------------------------------------------------------------------------- #
def _bash_with_wait_n() -> str | None:
    """A bash new enough for ``wait -n``, which launch_utils.sh requires (4.3+).

    macOS ships bash 3.2, so this is a skip locally unless a newer bash is on
    PATH; CI containers run bash 5.
    """
    seen = []
    for cand in ("bash", "/opt/homebrew/bin/bash", "/usr/local/bin/bash"):
        path = shutil.which(cand) if "/" not in cand else cand
        if not path or path in seen or not Path(path).exists():
            continue
        seen.append(path)
        out = subprocess.run(
            [path, "-c", 'echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"'],
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            continue
        try:
            major, minor = (int(x) for x in out.stdout.split())
        except ValueError:
            continue
        if (major, minor) >= (4, 3):
            return path
    return None


_BASH43 = _bash_with_wait_n()

_STUB_GPU_UTILS = """\
build_vllm_gpu_mem_args() { echo ""; }
"""
_STUB_LAUNCH_UTILS = """\
print_launch_banner() { :; }
print_curl_footer() { cat > /dev/null; }
wait_any_exit() { wait; }
"""
_STUB_PYTHON3 = """\
#!/bin/sh
echo "CVD=${CUDA_VISIBLE_DEVICES-<unset>} ARGS=$*" >> "$DYN_TEST_LAUNCH_LOG"
exit 0
"""


def _run_launcher(tmp_path, visible_devices):
    """Run the REAL launch script with a stand-in engine; return its launches."""
    root = tmp_path / "tree"
    launch_dir = root / "examples/backends/vllm/launch"
    common_dir = root / "examples/common"
    bin_dir = tmp_path / "bin"
    for d in (launch_dir, common_dir, bin_dir):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(_GANG_LAUNCH_SCRIPT, launch_dir / _GANG_LAUNCH_SCRIPT.name)
    (common_dir / "gpu_utils.sh").write_text(_STUB_GPU_UTILS)
    (common_dir / "launch_utils.sh").write_text(_STUB_LAUNCH_UTILS)
    stub = bin_dir / "python3"
    stub.write_text(_STUB_PYTHON3)
    stub.chmod(0o755)

    log = tmp_path / "launches.log"
    log.write_text("")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["DYN_TEST_LAUNCH_LOG"] = str(log)
    if visible_devices is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices

    proc = subprocess.run(
        [_BASH43, str(launch_dir / _GANG_LAUNCH_SCRIPT.name), "model-a", "model-b"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        # `trap 'kill 0' EXIT` in the script would otherwise signal the whole
        # pytest process group.
        start_new_session=True,
    )
    workers = {}
    for line in log.read_text().splitlines():
        if "dynamo.vllm" not in line:
            continue
        cvd = line.split("CVD=", 1)[1].split(" ARGS=", 1)[0]
        if "embed-worker-1" in line:
            workers["w1"] = cvd
        elif "embed-worker-2" in line:
            workers["w2"] = cvd
    return proc, workers


@pytest.mark.skipif(
    _BASH43 is None, reason="launch_utils.sh needs bash >= 4.3 for `wait -n`"
)
def test_each_worker_is_pinned_to_its_own_device_of_the_assigned_gang(tmp_path):
    """Handed the gang (2,3), the two workers must land on 2 and 3 -- not both
    on 2, not both inheriting the whole set."""
    proc, workers = _run_launcher(tmp_path, "2,3")

    assert workers.get("w1") == "2", (proc.stderr, workers)
    assert workers.get("w2") == "3", (proc.stderr, workers)
    assert workers["w1"] != workers["w2"], (
        "both vLLM workers were pinned to the same device -- the scheduler "
        "reserved two GPUs and charged VRAM on both, so one of them is now "
        "double-booked and the other is idle"
    )


@pytest.mark.skipif(
    _BASH43 is None, reason="launch_utils.sh needs bash >= 4.3 for `wait -n`"
)
def test_workers_follow_a_reordered_assignment_rather_than_sorting_it(tmp_path):
    """The launcher must not re-sort what it inherits.

    The scheduler always emits an ascending gang, so it cannot itself produce
    (3,2); this pins the launcher's half of the contract for a hand-run
    invocation, and stops a future "tidy the list" change from silently
    reintroducing a fixed device order.
    """
    _, workers = _run_launcher(tmp_path, "3,2")
    assert (workers.get("w1"), workers.get("w2")) == ("3", "2"), workers


@pytest.mark.skipif(
    _BASH43 is None, reason="launch_utils.sh needs bash >= 4.3 for `wait -n`"
)
def test_launcher_starts_no_worker_when_it_is_handed_one_device(tmp_path):
    """Fail closed: a single visible device must not run both workers on it."""
    proc, workers = _run_launcher(tmp_path, "4")
    assert proc.returncode != 0, proc.stdout
    assert workers == {}, f"workers started despite an unusable device set: {workers}"


def test_no_card_is_committed_past_its_multi_process_budget(monkeypatch):
    """Once two VRAM-bearing tests share a card, the margin must hold.

    The cap in force is the whole card while at most one VRAM-bearing test lives
    there, and ``budget_multi`` once a second joins -- so a sole occupant above
    ``budget_multi`` is correct, and only co-residency is a violation. Zero-VRAM
    fillers raise ``running_count`` without consuming budget and are excluded by
    design.

    The regime matters. Every other sweep here derives ``actual_free`` from the
    scheduler's own accounting, which makes the live-usage gate shadow the
    budget gate exactly and hides any error in the intra-pass tentative charge.
    This one reports the whole card as free, so only the budget arithmetic is
    load bearing: dividing a gang's tentative charge across its members instead
    of charging each one in full over-commits a card here, and is invisible in
    the derived regime.
    """
    rnd = random.Random(20260823)
    for _ in range(400):
        n_gpus = rnd.choice([1, 2, 3, 4])
        gpus = {
            gi: _gpu(gi, rnd.choice([8.0, 10.0, 16.0, 24.0])) for gi in range(n_gpus)
        }
        running: list[list] = []
        for _pass in range(25):
            pending = sorted(
                (
                    _t(
                        f"t{i}",
                        rnd.choice([0.0, 1.0, 2.5, 5.0, 9.0]),
                        timeout=rnd.choice([30.0, 300.0]),
                        gpus=rnd.choice([1, 1, 2]),
                    )
                    for i in range(rnd.randint(1, 4))
                ),
                key=_priority_key,
                reverse=True,
            )
            launches = _select_launches(
                pending=pending,
                gpu_states=gpus,
                # the whole card reads free, so the live gate never substitutes
                # for the budget gate
                actual_free={gi: gs.total_gib for gi, gs in gpus.items()},
                num_slots=8,
                running_count=len(running),
            )
            for idx, devices in launches:
                _reserve_gpus(pending[idx], devices, gpus)
                running.append([pending[idx], rnd.randint(1, 3)])

            for gi, gs in gpus.items():
                bearing = sum(
                    1
                    for entry, _ in running
                    if gi in entry.assigned_gpus and entry.profiled_gib > 0
                )
                if bearing >= 2:
                    assert gs.budget_used <= gs.budget_multi + 1e-9, (
                        f"GPU{gi} holds {bearing} VRAM-bearing tests totalling "
                        f"{gs.budget_used:.2f} GiB, past its multi-process "
                        f"budget of {gs.budget_multi:.2f} GiB"
                    )

            survivors = []
            for record in running:
                record[1] -= 1
                if record[1] <= 0:
                    _release_gpus(record[0], gpus)
                else:
                    survivors.append(record)
            running = survivors


def test_a_card_too_small_is_rejected_even_when_its_foreign_memory_reads_negative():
    """The capacity half of the reservation filter, isolated.

    ``exogenous`` is ``(total - live_free) - committed_budget``, so a test that
    holds budget it has not finished allocating drives it NEGATIVE -- here GPU2
    has 2.0 GiB committed but only 0.3 GiB resident, giving -1.7. The
    foreign-memory term then reads ``-1.7 + 6.0 <= 5.9`` and would happily admit
    a 5.9 GiB card to a gang needing 6.0 on each device. Only the hard-capacity
    term rejects it.

    Without that term the hold lands on {1, 2}, GPU0 -- the one other card that
    could ever run this gang -- is left open, and the 3.0 GiB filler cannot fit
    in what remains of it, so nothing launches at all.
    """
    gpus = {
        0: _gpu(0, 10.0, budget_used=6.0, running_count=1),
        1: _gpu(1, 10.0),
        2: _gpu(2, 5.9, budget_used=2.0, running_count=1),
    }
    actual_free = {0: 4.0, 1: 10.0, 2: 5.6}  # GPU2 has only 0.3 GiB resident
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    filler = _t("filler", 3.0, timeout=15.0)
    assert _unschedulable_reason(gang, gpus) is None

    launches = _select_checked(
        [gang, filler], gpus, num_slots=8, running_count=2, actual_free=actual_free
    )

    assert launches == [(1, (2,))], (
        "the 5.9 GiB card cannot host a 6.0 GiB gang no matter what its live "
        "free reading says; it must not take a reservation slot from GPU0"
    )


def test_gang_falls_back_to_a_full_hold_when_only_some_cards_look_usable():
    """The fallback must trigger on ANY shortfall, not only on a total wipeout.

    Three 16 GiB cards, a gang wanting 12.0 on two of them, and neighbours
    holding 5.0 GiB on GPU0 and GPU1 until pass 40. Only GPU2 passes the
    eligibility filter, so the preferred list has one entry for a gang that
    needs two.

    One survivor is still a shortfall: the gang must fall back to the unfiltered
    list and hold two cards. A fallback that only fires when the filtered list is
    completely empty leaves the gang unprotected here, and the backfill takes the
    cards while it waits.
    """
    gpus = {0: _gpu(0, 16.0), 1: _gpu(1, 16.0), 2: _gpu(2, 16.0)}
    gang = _t("gang", 12.0, timeout=1800, gpus=2)
    assert _unschedulable_reason(gang, gpus) is None

    started = _drive_passes(
        gpus,
        [gang],
        backfill=lambda i: _t(f"fill{i}", 3.0, timeout=600.0),
        passes=400,
        external_hold=lambda now: {0: 5.0, 1: 5.0} if now < 40 else {},
    )

    assert "gang" in started, "gang never launched after its blockers left"
    assert started["gang"] <= 60, (
        f"gang launched on pass {started['gang']}: with only one card passing "
        "the filter it went unprotected instead of falling back to a full hold"
    )


def test_a_gang_holds_nothing_rather_than_hold_cards_it_can_never_run_on():
    """The size filter still has to be load bearing on its own.

    Preferring usable cards in the ordering hides the filter in every scenario
    where a better card exists, because a card too small to host a member can
    never be usable either -- the preference sorts it last anyway. The filter
    earns its keep exactly where the ordering cannot help: when nothing better
    is LEFT. Then a preference has no choice but to hand the slot over, and a
    filter refuses.

    GPU0/GPU1 are the only cards that could host this 6.0 GiB gang, and a
    higher-priority gang has already taken them. What remains for the second
    gang is two 5.9 GiB cards it could never run on. Holding them would bound
    backfill that has nowhere else to go, in exchange for a launch that can
    never happen -- so it holds nothing, and the filler lands.
    """
    gpus = {
        0: _gpu(0, 10.0, budget_used=5.0, running_count=1),
        1: _gpu(1, 10.0, budget_used=5.0, running_count=1),
        2: _gpu(2, 5.9),
        3: _gpu(3, 5.9),
    }
    actual_free = {0: 5.0, 1: 5.0, 2: 5.9, 3: 5.9}
    first = _t("first", 6.0, timeout=1800, gpus=2)
    second = _t("second", 6.0, timeout=90, gpus=2)
    filler = _t("filler", 2.0, timeout=15.0)
    assert _unschedulable_reason(second, gpus) is None

    launches = _select_checked(
        [first, second, filler],
        gpus,
        num_slots=8,
        running_count=2,
        actual_free=actual_free,
    )

    assert launches == [(2, (2,))], (
        "the second gang took a hold on cards it can never run on, and the "
        "filler -- which had nowhere else to go -- was blocked for a launch "
        "that could never happen"
    )


def test_a_card_whose_usable_capacity_exactly_meets_the_need_is_still_preferred():
    """The preference boundary, matching the filter's boundary.

    ``_can_ever_host`` admits a card whose physical size exactly equals the
    requirement, because pre-flight does. ``_usable_now`` has to agree at its own
    boundary for the same reason: a card carrying foreign memory that leaves
    exactly the requirement free is usable, and ranking it below a card that is
    genuinely short would hand the hold to the wrong one.

    GPU0 has 4.0 GiB of foreign memory on a 10.0 GiB card -- exactly 6.0 usable.
    GPU2 is the most attractive card by headroom and the least usable. A strict
    ``>`` at the boundary demotes GPU0 beneath it, the hold lands on {1, 2}, and
    the filler that should have had GPU2 is refused everywhere.
    """
    gpus = {
        0: _gpu(0, 10.0, budget_used=5.0, running_count=1),
        1: _gpu(1, 10.0, budget_used=5.0, running_count=1),
        2: _gpu(2, 10.0, running_count=1),
    }
    # GPU0: 4.0 GiB foreign -> exactly 6.0 usable. GPU2: 4.5 GiB foreign -> 5.5.
    actual_free = {0: 1.0, 1: 5.0, 2: 5.5}
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    filler = _t("filler", 3.0, timeout=15.0)
    assert _unschedulable_reason(gang, gpus) is None

    launches = _select_checked(
        [gang, filler], gpus, num_slots=8, running_count=3, actual_free=actual_free
    )

    assert launches == [(1, (2,))], (
        "the hold skipped the card with exactly enough usable capacity and took "
        "the emptiest-looking one instead, which cannot host a member at all"
    )


def test_a_card_exactly_the_size_of_the_requirement_can_still_be_reserved():
    """The boundary must agree with ``_unschedulable_reason``.

    Pre-flight counts a card as able to host the test when ``total_gib >=
    profiled_gib``, so a 6.0 GiB card is feasible for a 6.0 GiB gang -- as its
    sole occupant, since an idle card's cap is the whole card. A reservation
    filter using a strict ``>`` would disagree with that, and on a node whose
    only large-enough cards are exact fits it would refuse to protect a gang
    pre-flight had just declared schedulable.

    Here GPU1 and GPU2 are exact fits and GPU0 is busy and too small, so the
    hold has to land on {1, 2} and refuse the filler both places; the only card
    left for it is GPU0.
    """
    gpus = {
        0: _gpu(0, 5.0, budget_used=2.0, running_count=1),
        1: _gpu(1, 6.0, budget_used=1.0, running_count=1),
        2: _gpu(2, 6.0, budget_used=1.0, running_count=1),
    }
    gang = _t("gang", 6.0, timeout=1800, gpus=2)
    filler = _t("filler", 1.5, timeout=15.0)
    assert _unschedulable_reason(gang, gpus) is None, "pre-flight must accept it"

    launches = _select_checked([gang, filler], gpus, num_slots=8, running_count=3)

    assert launches == [(1, (0,))], (
        "a card whose capacity exactly equals the requirement is feasible per "
        "_unschedulable_reason and must remain reservable"
    )
