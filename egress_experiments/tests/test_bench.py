# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The benchmark is the shared measuring stick, so it gets its own guards.

Every experiment is scored by ``bench.run_bench``. If it can be gamed, or if
it silently measures the wrong stage, six parallel experiments produce six
incomparable numbers. These pin the properties that make it a fair stick.
"""

from __future__ import annotations

import pytest

from egress_experiments import architectures, bench, loop_meter
from egress_experiments.costs import Costs
from egress_experiments.harness import run_simulation

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


def test_baselines_are_registered_and_discovery_did_not_break():
    assert "baseline-push" in architectures.names()
    assert "baseline-pull" in architectures.names()
    assert architectures.IMPORT_ERRORS == {}, architectures.IMPORT_ERRORS


def test_the_meter_ticks_on_the_loop_and_only_there():
    """Score integrity: an item counts when the LOOP is done with it."""
    import threading

    result = run_simulation(bench._config("baseline-push", 32, Costs()))
    assert result.loop_item_times, "nothing ticked the meter"
    # asyncio.run puts the loop on the calling thread.
    assert set(result.loop_meter_threads) == {
        threading.current_thread().name
    }, f"meter ticked off the loop: {result.loop_meter_threads}"


def test_loop_exit_and_system_exit_are_different_measurements():
    """The reason the meter exists.

    Under saturation the tokio-side consumer lags, so counting items after it
    under-reports the loop and inflates apparent per-item loop cost. The two
    counts must therefore be allowed to differ, and the score uses the former.
    """
    result = run_simulation(bench._config("baseline-push", 240, Costs()))
    assert len(result.loop_item_times) >= result.responses


def test_work_is_conserved_not_deleted():
    """Total modelled work per item is the invariant experiments must respect."""
    result = run_simulation(bench._config("baseline-push", 64, Costs()))
    items = len(result.loop_item_times)
    total_us = sum(result.spin_us_by_thread.values()) / max(1, items)
    # handle_response + build_response + push_send, plus the amortised ingress
    # stages and the off-loop rust egress.
    assert 85.0 < total_us < 130.0, total_us


def test_score_is_measured_not_computed_from_costs():
    """Doubling the modelled cost must roughly halve the score.

    If the score were derived from ``Costs`` this would hold trivially; the
    point is that it holds while being measured from timestamps, which is what
    stops an experiment 'winning' by editing a constant.
    """
    fast = bench.run_bench("baseline-push", Costs(), ladder=(120,), warmup_s=0.3)
    slow = bench.run_bench(
        "baseline-push", Costs().with_scale(2.0), ladder=(120,), warmup_s=0.3
    )
    assert fast.items_per_s > slow.items_per_s
    assert 1.4 < fast.items_per_s / slow.items_per_s < 2.6


def test_bench_reports_saturation_rather_than_assuming_it():
    result = bench.run_bench("baseline-push", Costs(), ladder=(240,), warmup_s=0.3)
    assert result.saturated
    assert result.backlog_growth_per_s > 0
    assert result.items_per_s > 1000


def test_meter_reset_between_runs():
    loop_meter.reset()
    assert loop_meter.timestamps() == []
    run_simulation(bench._config("baseline-push", 16, Costs()))
    first = len(loop_meter.timestamps())
    run_simulation(bench._config("baseline-push", 16, Costs()))
    second = len(loop_meter.timestamps())
    # Not cumulative across runs: the second run must not inherit the first.
    assert second < first * 2
