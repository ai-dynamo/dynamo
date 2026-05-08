# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-path parity for ``ReplayPlannerAdapter`` (PR 8 sub-task 8-8).

Validates that replay on the orchestrator path produces the same
``scaling_events`` / ``total_ticks`` as replay on the PSM path when
fed an identical synthetic bridge trace. PSM path is the frozen
reference — orchestrator path must match tick-by-tick.

Uses a hand-rolled fake bridge instead of the Rust PyO3 pyclass so the
test runs with no extra build deps and is deterministic.
"""

from __future__ import annotations

from typing import Any, Iterable

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    WorkerCapabilities,
)
from dynamo.planner.offline.replay_adapter import (
    ReplayPlannerAdapter,
    ReplayPlannerReport,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Fake bridge
# ---------------------------------------------------------------------------


class _FakeReplayBridge:
    """Mimics the Rust ``PlannerReplayBridge.advance_to`` contract.

    Scripts yield deterministic trace ticks. Each call to ``advance_to``
    returns the next scripted tick, or ``{"is_done": True}`` once
    exhausted. ``apply_scaling`` just records the targets so the test
    can assert what the planner commanded.
    """

    def __init__(self, ticks: Iterable[dict[str, Any]]) -> None:
        self._iter = iter(ticks)
        self.apply_scaling_calls: list[tuple[int, int]] = []

    def advance_to(self, tick_ms: float) -> dict[str, Any]:
        try:
            t = next(self._iter)
        except StopIteration:
            return {"is_done": True}
        # Caller may request a later tick than the script has; we just
        # return the next scripted item and let the adapter interpret
        # timestamps from the payload.
        return t

    def finalize(self) -> dict[str, Any]:
        """Stub for ``ReplayPlannerAdapter.run`` end-of-trace report
        assembly. Real bridge returns the accumulated trace summary."""
        return {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "duration_s": 0.0,
            "avg_ttft_s": 0.0,
            "avg_itl_s": 0.0,
        }

    def drain_traffic(self) -> dict[str, Any]:
        """Empty-window traffic stub. Upstream commit `c388483ae`
        (KV-reuse awareness in load + throughput scaling) made every
        load-only tick request traffic_metrics, so the adapter now
        always calls ``bridge.drain_traffic`` — keep this even though
        the parity tests don't depend on real traffic content."""
        return {
            "duration_s": 0.0,
            "num_req": 0,
            "avg_isl": 0.0,
            "avg_osl": 0.0,
            "avg_ttft_ms": 0.0,
            "avg_itl_ms": 0.0,
            "avg_kv_hit_rate": 0.0,
        }

    # Not actually exercised in these parity tests (replay adapter
    # currently uses ``apply_scaling_agg`` / ``apply_scaling_disagg``
    # method names; keep this stub for completeness).
    def apply_scaling_agg(self, decode: int) -> None:
        self.apply_scaling_calls.append((0, decode))

    def apply_scaling_disagg(self, prefill: int, decode: int) -> None:
        self.apply_scaling_calls.append((prefill, decode))


def _tick(
    now_ms: float,
    active_d: int,
    active_p: int = 0,
    decode_kv: int = 0,
    prefill_tok: int = 0,
    is_done: bool = False,
) -> dict[str, Any]:
    """Build one scripted bridge tick. Field names match the Rust
    ``PlannerReplayBridge`` snapshot contract consumed by
    ``_build_fpm_from_dict``."""
    snapshot = {
        "worker_id": 1,
        "wall_time": 0.01 if (decode_kv > 0 or prefill_tok > 0) else 0.0,
        "num_prefill_requests": 0,
        "sum_prefill_tokens": prefill_tok,
        "var_prefill_length": 0.0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 1 if decode_kv > 0 else 0,
        "sum_decode_kv_tokens": decode_kv,
        "var_decode_kv_tokens": 0.0,
        "num_queued_prefill": 0,
        "sum_queued_prefill_tokens": 0,
        "var_queued_prefill_length": 0.0,
        "num_queued_decode": 0,
        "sum_queued_decode_kv_tokens": 0,
        "var_queued_decode_kv_tokens": 0.0,
    }
    return {
        "is_done": is_done,
        "now_ms": now_ms,
        "active_prefill_count": active_p,
        "active_decode_count": active_d,
        "prefill_fpm_snapshots": [snapshot] if prefill_tok > 0 else [],
        "decode_fpm_snapshots": [snapshot] if decode_kv > 0 else [],
        "accumulated_metrics": {
            "num_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_sum": 0.0,
            "ttft_sum": 0.0,
            "itl_sum": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _agg_config(use_orchestrator: bool, **overrides) -> PlannerConfig:
    """agg + throughput target (easy mode) keeps the test deterministic
    without needing benchmark data."""
    return PlannerConfig(
        environment="kubernetes",
        mode="agg",
        optimization_target="throughput",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        load_adjustment_interval=10,
        scheduling={"use_orchestrator": use_orchestrator},
        **overrides,
    )


def _caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1,
            max_num_batched_tokens=2048,
            max_kv_tokens=16384,
            max_num_seqs=64,
        )
    )


def _run(use_orchestrator: bool, ticks: list[dict[str, Any]]) -> ReplayPlannerReport:
    bridge = _FakeReplayBridge(ticks)
    adapter = ReplayPlannerAdapter(
        planner_config=_agg_config(use_orchestrator),
        bridge=bridge,
        capabilities=_caps(),
    )
    return adapter.run()


# ---------------------------------------------------------------------------
# Dual-path parity
# ---------------------------------------------------------------------------


def _empty_trace(n_ticks: int = 5) -> list[dict[str, Any]]:
    """n ticks with no FPM content (idle)."""
    return [_tick(now_ms=i * 10_000.0, active_d=1) for i in range(1, n_ticks + 1)]


def _loaded_trace(n_ticks: int = 5) -> list[dict[str, Any]]:
    """n ticks with non-zero FPM so the regression sees live data."""
    return [
        _tick(now_ms=i * 10_000.0, active_d=1, decode_kv=1000 + i * 100)
        for i in range(1, n_ticks + 1)
    ]


def test_empty_trace_same_total_ticks():
    psm = _run(use_orchestrator=False, ticks=_empty_trace(6))
    orch = _run(use_orchestrator=True, ticks=_empty_trace(6))
    assert psm.total_ticks == orch.total_ticks
    assert psm.total_ticks > 0


def test_empty_trace_no_scaling_events():
    psm = _run(use_orchestrator=False, ticks=_empty_trace(6))
    orch = _run(use_orchestrator=True, ticks=_empty_trace(6))
    # Idle workload doesn't trigger scaling on either path.
    assert psm.scaling_events == []
    assert orch.scaling_events == []


def test_loaded_trace_same_scaling_event_sequence():
    psm = _run(use_orchestrator=False, ticks=_loaded_trace(6))
    orch = _run(use_orchestrator=True, ticks=_loaded_trace(6))
    # Scaling events list tuples of (component, from, to) — whatever
    # PSM decides, orchestrator must match.
    psm_summary = [
        (e.component, e.from_count, e.to_count) for e in psm.scaling_events
    ]
    orch_summary = [
        (e.component, e.from_count, e.to_count) for e in orch.scaling_events
    ]
    assert psm_summary == orch_summary


def test_loaded_trace_same_total_ticks():
    psm = _run(use_orchestrator=False, ticks=_loaded_trace(6))
    orch = _run(use_orchestrator=True, ticks=_loaded_trace(6))
    assert psm.total_ticks == orch.total_ticks


# ---------------------------------------------------------------------------
# Path plumbing sanity
# ---------------------------------------------------------------------------


def test_orchestrator_path_constructs_engine():
    bridge = _FakeReplayBridge(_empty_trace(2))
    adapter = ReplayPlannerAdapter(
        planner_config=_agg_config(use_orchestrator=True),
        bridge=bridge,
        capabilities=_caps(),
    )
    # Engine is OrchestratorEngineAdapter when flag is true.
    from dynamo.planner.plugins.orchestrator.engine_adapter import (
        OrchestratorEngineAdapter,
    )

    assert isinstance(adapter._engine, OrchestratorEngineAdapter)
    assert adapter._sm is None
    assert adapter._loop is not None


def test_psm_path_constructs_state_machine():
    bridge = _FakeReplayBridge(_empty_trace(2))
    adapter = ReplayPlannerAdapter(
        planner_config=_agg_config(use_orchestrator=False),
        bridge=bridge,
        capabilities=_caps(),
    )
    from dynamo.planner.core.engine_protocol import _PSMEngineAdapter

    assert isinstance(adapter._engine, _PSMEngineAdapter)
    assert adapter._sm is not None
    assert adapter._loop is None
