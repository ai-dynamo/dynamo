# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BuiltinLoadPropose.

Algorithmic parity vs ``PSM._advance_load`` across mode × easy/sla ×
config toggle combos. Tests call the plugin's ``_advance_load`` sync
entry directly (bypassing the async Propose stage wiring) so the
algorithm can be validated with identical inputs against PSM.

PSM has cross-tick state (regressions / predictors / `_num_*_workers` /
`_throughput_lower_bound_*`); we seed them explicitly via `load_benchmark_fpms`
and `_update_inventory` before driving `_advance_load`, then seed the
plugin's matching hooks (`update_throughput_lower_bounds`, shared
regression objects) so a same-tick comparison is apples-to-apples.
"""

from __future__ import annotations

import pytest

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.builtins.load_propose import BuiltinLoadPropose
from dynamo.planner.plugins.types import OverrideType, ProposeStageRequest, PipelineContext

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeOrchestrator:
    def __init__(self, caps: WorkerCapabilities, regressions: dict):
        self._caps = caps
        self._store = dict(regressions)

    @property
    def capabilities(self):
        return self._caps

    def get_regression(self, kind):
        return self._store.get(kind)

    def update_regression(self, kind, model):
        self._store[kind] = model


def _make_caps(mode: str) -> WorkerCapabilities:
    decode = EngineCapabilities(
        num_gpu=1,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        context_length=4096,
        max_kv_tokens=16384,
    )
    prefill = EngineCapabilities(
        num_gpu=1,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        context_length=4096,
        max_kv_tokens=16384,
    )
    if mode == "agg":
        return WorkerCapabilities(decode=decode)
    if mode == "decode":
        return WorkerCapabilities(decode=decode)
    if mode == "prefill":
        return WorkerCapabilities(prefill=prefill)
    return WorkerCapabilities(prefill=prefill, decode=decode)


def _make_config(
    *,
    mode: str,
    enable_load: bool = True,
    enable_throughput: bool = False,
    optimization_target: str = "throughput",  # easy mode default
    max_gpu_budget: int = -1,
    min_endpoint: int = 1,
) -> PlannerConfig:
    return PlannerConfig(
        mode=mode,
        enable_load_scaling=enable_load,
        enable_throughput_scaling=enable_throughput,
        optimization_target=optimization_target,
        max_gpu_budget=max_gpu_budget,
        min_endpoint=min_endpoint,
    )


def _fpm(
    *,
    worker_id="w1",
    dp_rank=0,
    queued_prefill=0,
    scheduled_prefill=0,
    queued_decode_kv=0,
    scheduled_decode_kv=0,
) -> ForwardPassMetrics:
    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        wall_time=1.0,
        scheduled_requests=ScheduledRequestMetrics(
            sum_prefill_tokens=scheduled_prefill,
            num_prefill_requests=0,
            sum_decode_kv_tokens=scheduled_decode_kv,
            num_decode_requests=0,
        ),
        queued_requests=QueuedRequestMetrics(
            sum_prefill_tokens=queued_prefill,
            sum_decode_kv_tokens=queued_decode_kv,
        ),
    )


def _build_plugin_and_psm(config, caps):
    """Build a PSM + matching plugin that share the PSM's regression objects."""
    psm = PlannerStateMachine(config, caps)
    regressions = {}
    if hasattr(psm, "_agg_regression"):
        regressions["agg"] = psm._agg_regression
    if hasattr(psm, "_prefill_regression"):
        regressions["prefill"] = psm._prefill_regression
    if hasattr(psm, "_decode_regression"):
        regressions["decode"] = psm._decode_regression
    orch = _FakeOrchestrator(caps, regressions)
    plugin = BuiltinLoadPropose(orch, config)  # type: ignore[arg-type]
    return plugin, psm


def _sync_worker_state(psm, plugin_counts: WorkerCounts):
    """PSM derives ``_num_*_workers`` from ``_update_inventory``. Mirror that
    so plugin + PSM see the same worker counts."""
    psm._update_inventory(plugin_counts)


def _sync_throughput_bounds(psm, plugin: BuiltinLoadPropose, p: int, d: int):
    psm._throughput_lower_bound_p = p
    psm._throughput_lower_bound_d = d
    plugin.update_throughput_lower_bounds(p, d)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_scaling_disabled_returns_accept():
    config = _make_config(mode="agg", enable_load=False, enable_throughput=True)
    plugin = BuiltinLoadPropose(_FakeOrchestrator(_make_caps("agg"), {}), config)  # type: ignore[arg-type]
    plugin.prime_tick(FpmObservations(), WorkerCounts(ready_num_decode=1))
    resp = await plugin.Propose(ProposeStageRequest(context=PipelineContext()))
    assert resp.result_kind == "accept"


@pytest.mark.asyncio
async def test_not_primed_returns_accept():
    config = _make_config(mode="agg")
    plugin = BuiltinLoadPropose(_FakeOrchestrator(_make_caps("agg"), {}), config)  # type: ignore[arg-type]
    # prime_tick never called
    resp = await plugin.Propose(ProposeStageRequest(context=PipelineContext()))
    assert resp.result_kind == "accept"


# ---------------------------------------------------------------------------
# Parity vs PSM: agg easy-mode scale-up
# ---------------------------------------------------------------------------


def test_agg_easy_scale_up_parity():
    config = _make_config(
        mode="agg",
        enable_load=True,
        enable_throughput=False,
        optimization_target="throughput",  # easy
    )
    caps = _make_caps("agg")
    plugin, psm = _build_plugin_and_psm(config, caps)

    counts = WorkerCounts(ready_num_decode=2)
    _sync_worker_state(psm, counts)

    # High utilization (above _DECODE_THROUGHPUT_SCALE_UP=1.0 threshold).
    obs = FpmObservations(
        decode={
            ("w1", 0): _fpm(
                worker_id="w1",
                scheduled_decode_kv=8000,
                queued_decode_kv=8000,
                queued_prefill=500,
            ),
            ("w2", 0): _fpm(
                worker_id="w2",
                scheduled_decode_kv=8000,
                queued_decode_kv=8000,
                queued_prefill=500,
            ),
        }
    )

    psm_decision = psm._advance_load(obs)
    plugin_decision = plugin._advance_load(obs, counts)
    assert psm_decision == plugin_decision  # dataclass eq


def test_agg_easy_scale_down_parity():
    config = _make_config(
        mode="agg",
        enable_load=True,
        optimization_target="throughput",
    )
    caps = _make_caps("agg")
    plugin, psm = _build_plugin_and_psm(config, caps)

    counts = WorkerCounts(ready_num_decode=3)
    _sync_worker_state(psm, counts)

    # Low utilization (below _DECODE_THROUGHPUT_SCALE_DOWN=0.6).
    obs = FpmObservations(
        decode={
            ("w1", 0): _fpm(worker_id="w1", scheduled_decode_kv=100, queued_decode_kv=100),
            ("w2", 0): _fpm(worker_id="w2", scheduled_decode_kv=100, queued_decode_kv=100),
            ("w3", 0): _fpm(worker_id="w3", scheduled_decode_kv=100, queued_decode_kv=100),
        }
    )

    psm_decision = psm._advance_load(obs)
    plugin_decision = plugin._advance_load(obs, counts)
    assert psm_decision == plugin_decision


def test_agg_no_change_parity():
    config = _make_config(mode="agg", enable_load=True, optimization_target="throughput")
    caps = _make_caps("agg")
    plugin, psm = _build_plugin_and_psm(config, caps)

    counts = WorkerCounts(ready_num_decode=2)
    _sync_worker_state(psm, counts)

    # Mid utilization (~0.7; below up=1.0, above down=0.6) → no change.
    obs = FpmObservations(
        decode={
            ("w1", 0): _fpm(worker_id="w1", scheduled_decode_kv=5000, queued_decode_kv=5000),
            ("w2", 0): _fpm(worker_id="w2", scheduled_decode_kv=5000, queued_decode_kv=5000),
        }
    )

    # Both should return None (no change).
    assert psm._advance_load(obs) is None
    assert plugin._advance_load(obs, counts) is None


# ---------------------------------------------------------------------------
# Parity vs PSM: disagg easy-mode
# ---------------------------------------------------------------------------


def test_disagg_easy_per_component_parity():
    config = _make_config(mode="disagg", enable_load=True, optimization_target="throughput")
    caps = _make_caps("disagg")
    plugin, psm = _build_plugin_and_psm(config, caps)

    counts = WorkerCounts(ready_num_prefill=2, ready_num_decode=2)
    _sync_worker_state(psm, counts)

    # prefill queued >> ctx_len → scale up; decode utilization low → scale down.
    obs = FpmObservations(
        prefill={
            ("w1", 0): _fpm(worker_id="w1", queued_prefill=8000),
            ("w2", 0): _fpm(worker_id="w2", queued_prefill=8000),
        },
        decode={
            ("w1", 0): _fpm(worker_id="w1", scheduled_decode_kv=100, queued_decode_kv=100),
            ("w2", 0): _fpm(worker_id="w2", scheduled_decode_kv=100, queued_decode_kv=100),
        },
    )

    psm_decision = psm._advance_load(obs)
    plugin_decision = plugin._advance_load(obs, counts)
    assert psm_decision == plugin_decision


def test_disagg_no_fpm_parity():
    config = _make_config(mode="disagg", enable_load=True, optimization_target="throughput")
    caps = _make_caps("disagg")
    plugin, psm = _build_plugin_and_psm(config, caps)

    counts = WorkerCounts(ready_num_prefill=1, ready_num_decode=1)
    _sync_worker_state(psm, counts)

    obs = FpmObservations()  # no FPM for either
    assert psm._advance_load(obs) is None
    assert plugin._advance_load(obs, counts) is None


# ---------------------------------------------------------------------------
# scaling_in_progress: expected != ready → return None
# ---------------------------------------------------------------------------


def test_scaling_in_progress_single_returns_none_parity():
    config = _make_config(mode="decode", enable_load=True, optimization_target="throughput")
    caps = _make_caps("decode")
    plugin, psm = _build_plugin_and_psm(config, caps)

    counts = WorkerCounts(
        ready_num_decode=2,
        expected_num_decode=3,  # mismatch -> scaling_in_progress
    )
    _sync_worker_state(psm, counts)

    obs = FpmObservations(
        decode={
            ("w1", 0): _fpm(worker_id="w1", scheduled_decode_kv=9000, queued_decode_kv=9000),
            ("w2", 0): _fpm(worker_id="w2", scheduled_decode_kv=9000, queued_decode_kv=9000),
        }
    )

    assert psm._advance_load(obs) is None
    assert plugin._advance_load(obs, counts) is None


# ---------------------------------------------------------------------------
# worker_count mismatch → return None
# ---------------------------------------------------------------------------


def test_worker_count_mismatch_parity():
    config = _make_config(mode="decode", enable_load=True, optimization_target="throughput")
    caps = _make_caps("decode")
    plugin, psm = _build_plugin_and_psm(config, caps)

    counts = WorkerCounts(ready_num_decode=3)  # expect 3 workers
    _sync_worker_state(psm, counts)

    # Only 1 worker FPM reported — mismatch.
    obs = FpmObservations(
        decode={
            ("w1", 0): _fpm(worker_id="w1", scheduled_decode_kv=9000, queued_decode_kv=9000),
        }
    )

    assert psm._advance_load(obs) is None
    assert plugin._advance_load(obs, counts) is None


# ---------------------------------------------------------------------------
# Stage wiring: Propose output shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_emits_set_on_scale_up():
    config = _make_config(mode="agg", enable_load=True, optimization_target="throughput")
    caps = _make_caps("agg")
    plugin, _psm = _build_plugin_and_psm(config, caps)

    counts = WorkerCounts(ready_num_decode=2)
    obs = FpmObservations(
        decode={
            ("w1", 0): _fpm(worker_id="w1", scheduled_decode_kv=8000, queued_decode_kv=8000, queued_prefill=500),
            ("w2", 0): _fpm(worker_id="w2", scheduled_decode_kv=8000, queued_decode_kv=8000, queued_prefill=500),
        }
    )
    plugin.prime_tick(obs, counts)

    resp = await plugin.Propose(ProposeStageRequest(context=PipelineContext()))
    assert resp.result_kind == "override"
    assert len(resp.override.targets) == 1
    t = resp.override.targets[0]
    assert t.sub_component_type == "decode"
    assert t.type == OverrideType.SET
    assert t.replicas == 3  # scale up from 2 → 3


@pytest.mark.asyncio
async def test_propose_accept_when_no_change():
    config = _make_config(mode="agg", enable_load=True, optimization_target="throughput")
    caps = _make_caps("agg")
    plugin, _psm = _build_plugin_and_psm(config, caps)

    counts = WorkerCounts(ready_num_decode=2)
    obs = FpmObservations(
        decode={
            ("w1", 0): _fpm(worker_id="w1", scheduled_decode_kv=5000, queued_decode_kv=5000),
            ("w2", 0): _fpm(worker_id="w2", scheduled_decode_kv=5000, queued_decode_kv=5000),
        }
    )
    plugin.prime_tick(obs, counts)

    # Mid util → plugin's _advance_load returns None → Propose returns accept.
    resp = await plugin.Propose(ProposeStageRequest(context=PipelineContext()))
    # NOTE: agg easy scale-up = "engine above 1.0"; mid (~0.7) is below up,
    # above down; _agg_easy_decision returns None → _advance_load_agg returns
    # ScalingDecision(num_decode=num_workers) only when throughput_floor lifts
    # BUT enable_throughput=False here, so we just get None → Accept.
    assert resp.result_kind == "accept"
