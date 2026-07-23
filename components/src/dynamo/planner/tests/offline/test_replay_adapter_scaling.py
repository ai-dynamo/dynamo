# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    WorkerCapabilities,
)
from dynamo.planner.monitoring.traffic_metrics import Metrics
from dynamo.planner.offline.replay_adapter import (
    ReplayPlannerAdapter,
    ReplayPlannerReport,
    ScalingEvent,
    ScalingProposal,
)


class _FakeEngine:
    def __init__(self, effects: list[PlannerEffects]):
        self._effects = list(effects)
        self.inputs = []
        self.shutdown_calls = 0

    async def tick(self, tick, tick_input):
        self.inputs.append((tick, tick_input))
        return self._effects.pop(0)

    async def shutdown(self):
        self.shutdown_calls += 1


class _DisabledRecorder:
    enabled = False

    def finalize(self):
        return None


class _RecordingRecorder:
    enabled = True

    def __init__(self):
        self.records = []

    def record(self, *args):
        self.records.append(args)

    def finalize(self):
        return None


def _adapter(mode: str, *effects: PlannerEffects) -> ReplayPlannerAdapter:
    adapter = ReplayPlannerAdapter.__new__(ReplayPlannerAdapter)
    adapter._config = PlannerConfig(mode=mode, optimization_target="throughput")
    adapter._is_disagg = mode == "disagg"
    adapter._capabilities = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=2) if mode == "disagg" else None,
        decode=EngineCapabilities(num_gpu=4),
    )
    adapter._engine = _FakeEngine(list(effects))
    adapter._loop = asyncio.new_event_loop()
    adapter._owns_loop = True
    adapter._engine_shutdown = False
    adapter._orchestrator_bootstrapped = True
    adapter._pending_tick = ScheduledTick(at_s=5.0, need_worker_states=True)
    adapter._prefill_fpm_cache = {}
    adapter._decode_fpm_cache = {}
    adapter._pending_traffic = None
    adapter._scaling_target_prefill = None
    adapter._scaling_target_decode = None
    adapter._last_ready_prefill = None
    adapter._last_ready_decode = None
    adapter._last_total_prefill = None
    adapter._last_total_decode = None
    adapter._has_total_prefill = False
    adapter._has_total_decode = False
    adapter._scaling_events = []
    adapter._mediation_events = []
    adapter._diagnostics_log = []
    adapter._total_ticks = 0
    adapter._recorder = _DisabledRecorder()
    adapter._cumulative_gpu_hours = 0.0
    adapter._last_tick_s = 0.0
    adapter._last_traffic = Metrics()
    return adapter


def _result(
    *,
    ready_prefill: int,
    ready_decode: int,
    total_prefill: int | None = None,
    total_decode: int | None = None,
):
    result = {
        "now_ms": 5_000.0,
        "active_prefill_count": ready_prefill,
        "active_decode_count": ready_decode,
        "active_prefill_ids": list(range(ready_prefill)),
        "active_decode_ids": list(range(ready_decode)),
        "prefill_fpm_snapshots": [],
        "decode_fpm_snapshots": [],
    }
    if total_prefill is not None:
        result["total_prefill_count"] = total_prefill
    if total_decode is not None:
        result["total_decode_count"] = total_decode
    return result


def _effects(
    *,
    prefill: int | None = None,
    decode: int | None = None,
    next_at_s: float | None = 10.0,
) -> PlannerEffects:
    return PlannerEffects(
        scale_to=(
            ScalingDecision(num_prefill=prefill, num_decode=decode)
            if prefill is not None or decode is not None
            else None
        ),
        next_tick=(
            ScheduledTick(at_s=next_at_s, need_worker_states=True)
            if next_at_s is not None
            else None
        ),
        diagnostics=TickDiagnostics(),
    )


def test_rejected_proposal_does_not_mutate_expected_or_applied_events():
    adapter = _adapter("disagg", _effects(prefill=3))
    try:
        tick_result = adapter.propose_tick(_result(ready_prefill=1, ready_decode=2))
        proposal = tick_result.proposal

        assert proposal is not None
        assert proposal.target_prefill == 3
        assert proposal.target_decode is None
        assert adapter._scaling_target_prefill is None
        assert adapter._scaling_target_decode is None
        assert adapter._scaling_events == []

        adapter.observe_rejection(proposal, message="GPU budget breach")

        assert adapter._scaling_target_prefill is None
        assert adapter._scaling_target_decode is None
        assert adapter._scaling_events == []
        assert adapter._mediation_events[-1].status == "rejected"
    finally:
        adapter.close()


def test_report_keeps_legacy_positional_constructor_order():
    scaling_event = ScalingEvent(
        at_s=5.0,
        component="agg",
        from_count=1,
        to_count=2,
    )
    diagnostics = [TickDiagnostics()]

    report = ReplayPlannerReport(
        {"request_count": 1},
        [scaling_event],
        diagnostics,
        3,
        "planner.html",
    )

    assert report.scaling_events == [scaling_event]
    assert report.diagnostics_log is diagnostics
    assert report.total_ticks == 3
    assert report.html_report_path == "planner.html"
    assert report.mediation_events == []


def test_finalize_preserves_report_error_when_cleanup_also_fails():
    class ReportError(RuntimeError):
        pass

    class _FailingAdapter:
        def build_report(self, _trace_report):
            raise ReportError("report generation failed")

        def close(self):
            raise RuntimeError("engine shutdown failed")

    adapter = _FailingAdapter()
    with pytest.raises(ReportError, match="report generation failed") as exc_info:
        ReplayPlannerAdapter.finalize(adapter, {})  # type: ignore[arg-type]

    assert exc_info.value.__notes__ == [
        "Replay planner cleanup also failed: engine shutdown failed"
    ]


def test_partner_partial_commit_updates_adapter_that_did_not_tick():
    adapter = _adapter("disagg")
    try:
        rejected = ScalingProposal(
            at_s=5.0,
            current_prefill=1,
            current_decode=2,
            target_decode=5,
        )
        adapter.observe_rejection(rejected, message="waiting for a partner")

        # A later request selects only part of this cached intent. This adapter
        # did not run a local tick at the actuation timestamp.
        adapter.observe_committed_scale(
            at_s=7.0,
            target_decode=3,
            from_decode=2,
            reason="global_planner_partner",
        )

        assert adapter._scaling_target_decode == 3
        assert adapter._scaling_events[-1].component == "decode"
        assert adapter._scaling_events[-1].from_count == 2
        assert adapter._scaling_events[-1].to_count == 3
        assert adapter._scaling_events[-1].reason == "global_planner_partner"
    finally:
        adapter.close()


def test_partner_commit_stays_pending_until_a_later_inventory_observation():
    adapter = _adapter("agg")
    adapter._last_ready_decode = 2
    adapter._last_total_decode = 2
    adapter._has_total_decode = True
    adapter._scaling_target_decode = 3
    try:
        # This partner did not tick at t=7. Its last snapshot happens to match
        # the new target, but that stale snapshot cannot settle a new action.
        adapter.observe_committed_scale(
            at_s=7.0,
            target_decode=2,
            reason="global_planner_partner",
        )

        assert adapter._scaling_target_decode == 2
        assert adapter._scaling_events[-1].from_count == 3
        assert adapter._scaling_events[-1].to_count == 2

        # The first post-commit observation still sees a draining worker.
        adapter._observe_runtime_inventory(
            _result(
                ready_prefill=0,
                ready_decode=2,
                total_prefill=0,
                total_decode=3,
            )
        )
        assert adapter._scaling_target_decode == 2

        # Only a subsequent settled runtime snapshot retires the target.
        adapter._observe_runtime_inventory(
            _result(
                ready_prefill=0,
                ready_decode=2,
                total_prefill=0,
                total_decode=2,
            )
        )
        assert adapter._scaling_target_decode is None
    finally:
        adapter.close()


def test_proposal_preserves_component_mask():
    adapter = _adapter("disagg", _effects(prefill=3, decode=None))
    try:
        proposal = adapter.propose_tick(
            _result(ready_prefill=1, ready_decode=2)
        ).proposal

        assert proposal is not None
        assert proposal.target_prefill == 3
        assert proposal.target_decode is None
    finally:
        adapter.close()


def test_legacy_on_tick_immediately_commits_and_keeps_bridge_shape():
    adapter = _adapter("agg", _effects(decode=3, next_at_s=12.0))
    try:
        decision = adapter.on_tick(_result(ready_prefill=0, ready_decode=1))

        assert decision == {
            "target_prefill": None,
            "target_decode": 3,
            "next_tick_ms": 12_000.0,
        }
        assert adapter._scaling_target_decode == 3
        assert adapter._scaling_events[-1].component == "agg"
        assert adapter._scaling_events[-1].from_count == 1
        assert adapter._scaling_events[-1].to_count == 3
        assert adapter._mediation_events[-1].status == "approved"
    finally:
        adapter.close()


def test_total_lifecycle_count_keeps_target_in_progress_until_drained():
    adapter = _adapter("agg", _effects(), _effects())
    adapter._scaling_target_decode = 1
    try:
        adapter.propose_tick(
            _result(
                ready_prefill=0,
                ready_decode=1,
                total_prefill=0,
                total_decode=2,
            )
        )
        first_counts = adapter._engine.inputs[-1][1].worker_counts

        assert first_counts is not None
        assert first_counts.expected_num_decode == 1
        assert first_counts.decode_scaling_in_progress
        assert adapter._scaling_target_decode == 1

        adapter.propose_tick(
            _result(
                ready_prefill=0,
                ready_decode=1,
                total_prefill=0,
                total_decode=1,
            )
        )
        settled_counts = adapter._engine.inputs[-1][1].worker_counts

        assert settled_counts is not None
        assert settled_counts.expected_num_decode == 1
        assert not settled_counts.decode_scaling_in_progress
        assert adapter._scaling_target_decode is None
    finally:
        adapter.close()


def test_async_controller_surface_exposes_initial_tick_and_proposal():
    adapter = _adapter("agg", _effects(decode=2))

    async def drive():
        assert await adapter.initial_tick_ms_async() == 5_000.0

        tick_result = await adapter.propose_tick_async(
            _result(ready_prefill=0, ready_decode=1)
        )

        assert tick_result.proposal is not None
        assert tick_result.proposal.target_decode == 2
        assert adapter._scaling_target_decode is None
        await adapter.shutdown_async()

    try:
        asyncio.run(drive())
        assert adapter._engine.shutdown_calls == 1
    finally:
        adapter.close()


def test_gpu_hours_use_runtime_capabilities_not_planner_config_hints():
    adapter = _adapter("disagg")
    recorder = _RecordingRecorder()
    adapter._recorder = recorder
    adapter._config.prefill_engine_num_gpu = 100
    adapter._config.decode_engine_num_gpu = 100
    adapter._last_tick_s = 5.0
    try:
        adapter._record_diagnostics(
            TickInput(now_s=3_605.0),
            _effects(),
            {
                "active_prefill_count": 1,
                "active_decode_count": 2,
            },
            emit_diagnostics=True,
        )

        # One simulated hour at 1x2-GPU prefill + 2x4-GPU decode.
        assert adapter._cumulative_gpu_hours == 10.0
        assert len(recorder.records) == 1
    finally:
        adapter.close()
