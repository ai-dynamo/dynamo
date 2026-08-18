# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit + makespan-simulation tests for the GPU-parallel scheduler.

Covers the pure scheduling core (``_select_launches`` / ``_priority_key``) and a
discrete-event makespan simulation on the real ``job-log.txt`` workload, showing
the VRAM-aware ordering beats the legacy timeout-sorted first-fit. No GPU or
``pynvml`` required -- the scheduler core is pure arithmetic.
"""

from __future__ import annotations

import pytest

from tests.utils.pytest_parallel_gpu import (
    _GpuState,
    _priority_key,
    _select_launches,
    _TestEntry,
)
from tests.utils.vram_utils import VRAM_MULTI_PROC_MARGIN

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


def _t(name: str, profiled: float, timeout: float = 600.0) -> _TestEntry:
    return _TestEntry(id=name, name=name, profiled_gib=profiled, timeout=timeout)


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

    assert launches == [(0, 0), (1, 0)]


def test_first_test_uses_full_card_then_multi_proc_margin():
    # A 20 GiB test fits only because it is first (full 24 GiB cap). Once it is
    # placed the cap drops to budget_multi (20.4), so the 4 GiB test is rejected.
    gpus = {0: _gpu(0, 24.0)}  # budget_multi = 20.4
    pending = [_t("anchor", 20.0), _t("extra", 4.0)]

    launches = _select(pending, gpus, num_slots=8)

    assert launches == [(0, 0)]


def test_multi_gpu_spreads_then_packs():
    # Two 22 GiB cards. 13 anchors GPU0; 12 cannot share it (19-13<12) so it
    # anchors GPU1; the 3.8 best-fits onto GPU1 (more free budget than GPU0).
    gpus = {0: _gpu(0, 22.0), 1: _gpu(1, 22.0)}
    pending = [_t("a", 13.0), _t("b", 12.0), _t("c", 3.8)]

    launches = _select(pending, gpus, num_slots=8)

    assert launches == [(0, 0), (1, 1), (2, 1)]


# --------------------------------------------------------------------------- #
# _select_launches: fillers
# --------------------------------------------------------------------------- #
def test_zero_vram_fillers_bypass_budget():
    # GPU budget fully committed to a running VRAM test, yet 0-GiB fillers must
    # still take free slots (they allocate no memory).
    gpus = {0: _gpu(0, 22.0, budget_used=19.0, running_count=1)}
    pending = [_t("f0", 0.0), _t("f1", 0.0)]

    launches = _select(pending, gpus, num_slots=8, running_count=1)

    assert launches == [(0, 0), (1, 0)]


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
    assert _select(pending, gpus, num_slots=8) == [(0, 0)]

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

    assert launches == [(1, 0)]  # only one 3.8 backfill; room held for the 12


def test_blocked_test_launches_once_occupant_frees():
    # Same setup, but now the 8 GiB occupant has freed (budget_used back to the
    # single 3.8 backfill). The 12 GiB test is now first and must launch.
    gpus = {0: _gpu(0, 22.0, budget_used=3.8, running_count=1)}
    pending = [_t("blocked12", 12.0), _t("fill_b", 3.8)]

    launches = _select(pending, gpus, num_slots=8, running_count=1)

    # 12 fits (19-3.8=15.2) and is highest priority; the extra 3.8 no longer fits
    # (19-15.8=3.2<3.8).
    assert launches == [(0, 0)]


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
    in its given order, no filler bypass, no reservation."""
    tent = {
        gi: {
            "budget": gs.budget_used,
            "free": actual_free[gi],
            "count": gs.running_count,
        }
        for gi, gs in gpu_states.items()
    }
    to_launch: list[tuple[int, int]] = []
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
            to_launch.append((i, best_gi))
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
            for idx, gi in sorted(sel, key=lambda x: x[0], reverse=True):
                t = pending.pop(idx)
                gpu_states[gi].budget_used += t.profiled_gib
                gpu_states[gi].running_count += 1
                running.append(
                    {"finish": now + t.runtime, "profiled": t.profiled_gib, "gpu": gi}
                )
            continue
        assert running, "deadlock: pending tests but nothing running"
        now = min(r["finish"] for r in running)
        still = []
        for r in running:
            if r["finish"] <= now:
                gpu_states[r["gpu"]].budget_used -= r["profiled"]
                gpu_states[r["gpu"]].running_count -= 1
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


# --------------------------------------------------------------------------- #
# effective_cpu_budget: don't let -n auto (host os.cpu_count) oversubscribe CPU
# --------------------------------------------------------------------------- #
def test_effective_cpu_budget_prefers_num_cpus_env():
    import os

    from tests.utils.vram_utils import effective_cpu_budget

    old = os.environ.get("NUM_CPUS")
    try:
        os.environ["NUM_CPUS"] = "4"
        assert effective_cpu_budget() == 4
        os.environ["NUM_CPUS"] = "7"
        assert effective_cpu_budget() == 7
        # Invalid value falls through to cgroup/os.cpu_count() and stays positive.
        os.environ["NUM_CPUS"] = "bogus"
        assert effective_cpu_budget() >= 1
    finally:
        if old is None:
            os.environ.pop("NUM_CPUS", None)
        else:
            os.environ["NUM_CPUS"] = old


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
