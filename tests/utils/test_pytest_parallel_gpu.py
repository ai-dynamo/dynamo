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
import random

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
    DEFAULT_GPU_COUNT,
    VRAM_MULTI_PROC_MARGIN,
    gpu_count_from_marker_names,
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
    # Two GPUs, both busy; a blocked 12 GiB gang needs headroom on BOTH. The
    # backfill cap (cap - required = 19 - 12 = 7) therefore applies to each
    # card, so only one 3.8 filler fits per card, not two.
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

    # The gang itself cannot run yet (19 - 8 = 11 < 12).
    assert all(idx != 0 for idx, _ in launches)
    # One filler per reserved card, and no third: 7 GiB of backfill room each.
    assert launches == [(1, (0,)), (2, (1,))]


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
