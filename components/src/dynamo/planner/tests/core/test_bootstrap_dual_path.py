# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``NativePlannerBase._install_benchmark_fpms`` dual-path
routing + ``OrchestratorEngineAdapter.bootstrap_from_fpms``.

The 4 mode subclasses (``PrefillPlanner`` / ``DecodePlanner`` /
``AggPlanner`` / ``DisaggPlanner``) each call ``_install_benchmark_fpms``
with a different FPM subset — this file tests the routing logic
directly rather than the full mode subclass (which requires connector /
runtime / Prometheus wiring).
"""

from __future__ import annotations

import pytest

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.plugins.orchestrator.engine_adapter import (
    OrchestratorEngineAdapter,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _MinimalPlanner(NativePlannerBase):
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.runtime = None
        self.namespace = config.namespace
        self.model_name = None
        self.prefill_worker_info = _FakeWorkerInfo()
        self.decode_worker_info = _FakeWorkerInfo()
        self._state_machine = None
        self._engine = None
        self._last_worker_counts = None
        self.prometheus_port = 0
        self.prometheus_metrics = None


class _FakeWorkerInfo:
    def __init__(self):
        self.num_gpu = None
        self.max_num_batched_tokens = None
        self.max_num_seqs = None
        self.context_length = None
        self.max_kv_tokens = None


def _config(*, use_orchestrator: bool, mode: str = "disagg", target: str = "sla"):
    return PlannerConfig(
        environment="kubernetes",
        mode=mode,
        enable_load_scaling=True,
        enable_throughput_scaling=True if target == "sla" else True,
        optimization_target=target,
        scheduling={"use_orchestrator": use_orchestrator},
    )


def _fpm(worker_id="w1", dp_rank=0, wall_time=0.5):
    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        wall_time=wall_time,
        scheduled_requests=ScheduledRequestMetrics(
            sum_prefill_tokens=512,
            num_prefill_requests=1,
            sum_decode_kv_tokens=1024,
            num_decode_requests=4,
        ),
        queued_requests=QueuedRequestMetrics(
            sum_prefill_tokens=0,
            sum_decode_kv_tokens=0,
        ),
    )


# ---------------------------------------------------------------------------
# PSM path (default) — must match pre-PR-7 behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_psm_path_routes_prefill_fpms_to_psm():
    planner = _MinimalPlanner(_config(use_orchestrator=False, mode="prefill"))
    fpms = [_fpm() for _ in range(5)]
    await planner._install_benchmark_fpms(prefill_fpms=fpms)
    # PSM was lazily constructed + FPMs loaded.
    assert planner._state_machine is not None
    assert planner._state_machine._prefill_regression.num_observations > 0


@pytest.mark.asyncio
async def test_psm_path_routes_decode_fpms_to_psm():
    planner = _MinimalPlanner(_config(use_orchestrator=False, mode="decode"))
    fpms = [_fpm() for _ in range(5)]
    await planner._install_benchmark_fpms(decode_fpms=fpms)
    assert planner._state_machine._decode_regression.num_observations > 0


@pytest.mark.asyncio
async def test_psm_path_routes_agg_fpms_to_psm():
    planner = _MinimalPlanner(_config(use_orchestrator=False, mode="agg"))
    fpms = [_fpm() for _ in range(5)]
    await planner._install_benchmark_fpms(agg_fpms=fpms)
    assert planner._state_machine._agg_regression.num_observations > 0


@pytest.mark.asyncio
async def test_psm_path_empty_call_is_noop():
    """All FPM slots ``None`` → no PSM.load_benchmark_fpms call;
    PSM should still be constructed lazily (by ``state_machine`` access
    elsewhere) but no regression data loaded."""
    planner = _MinimalPlanner(_config(use_orchestrator=False, mode="disagg"))
    await planner._install_benchmark_fpms()
    # PSM not auto-constructed because no FPM was provided.
    assert planner._state_machine is None


@pytest.mark.asyncio
async def test_psm_path_disagg_passes_both_prefill_and_decode():
    planner = _MinimalPlanner(_config(use_orchestrator=False, mode="disagg"))
    p_fpms = [_fpm(worker_id="p1") for _ in range(5)]
    d_fpms = [_fpm(worker_id="d1") for _ in range(5)]
    await planner._install_benchmark_fpms(prefill_fpms=p_fpms, decode_fpms=d_fpms)
    assert planner._state_machine._prefill_regression.num_observations > 0
    assert planner._state_machine._decode_regression.num_observations > 0


# ---------------------------------------------------------------------------
# Orchestrator path — FPMs go through bootstrap_from_fpms to engine adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_path_installs_regressions_via_adapter():
    planner = _MinimalPlanner(_config(use_orchestrator=True, mode="disagg"))
    p_fpms = [_fpm(worker_id="p1") for _ in range(5)]
    d_fpms = [_fpm(worker_id="d1") for _ in range(5)]
    await planner._install_benchmark_fpms(prefill_fpms=p_fpms, decode_fpms=d_fpms)

    # PSM NOT constructed on the main planner (orchestrator path).
    assert planner._state_machine is None
    # Adapter was constructed + has regressions.
    engine = planner._engine
    assert isinstance(engine, OrchestratorEngineAdapter)
    orch = engine._orchestrator
    assert orch.get_regression("prefill") is not None
    assert orch.get_regression("decode") is not None


@pytest.mark.asyncio
async def test_orchestrator_path_agg_mode_installs_agg_regression():
    planner = _MinimalPlanner(_config(use_orchestrator=True, mode="agg"))
    fpms = [_fpm() for _ in range(5)]
    await planner._install_benchmark_fpms(agg_fpms=fpms)
    engine = planner._engine
    assert isinstance(engine, OrchestratorEngineAdapter)
    assert engine._orchestrator.get_regression("agg") is not None


@pytest.mark.asyncio
async def test_orchestrator_path_easy_mode_skips_regression_install():
    """Easy mode (``optimization_target != "sla"``) doesn't use
    regression models. The adapter's ``bootstrap_from_fpms`` should
    skip the throwaway-PSM step in that case."""
    planner = _MinimalPlanner(
        _config(use_orchestrator=True, mode="disagg", target="throughput")
    )
    p_fpms = [_fpm() for _ in range(5)]
    await planner._install_benchmark_fpms(prefill_fpms=p_fpms)
    engine = planner._engine
    assert isinstance(engine, OrchestratorEngineAdapter)
    # No regression installed in easy mode.
    assert engine._orchestrator.get_regression("prefill") is None


# ---------------------------------------------------------------------------
# Direct test of adapter.bootstrap_from_fpms
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_bootstrap_from_fpms_populates_regressions():
    from dynamo.planner.core.types import WorkerCapabilities, EngineCapabilities

    config = PlannerConfig(
        mode="disagg",
        enable_throughput_scaling=True,
        enable_load_scaling=False,
        optimization_target="sla",
    )
    caps = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048),
        decode=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384),
    )
    adapter = OrchestratorEngineAdapter(config, caps)
    await adapter.bootstrap_from_fpms(
        prefill_fpms=[_fpm() for _ in range(5)],
        decode_fpms=[_fpm() for _ in range(5)],
    )
    assert adapter._orchestrator.get_regression("prefill") is not None
    assert adapter._orchestrator.get_regression("decode") is not None
    # Agg was not passed, so nothing there.
    assert adapter._orchestrator.get_regression("agg") is None
    await adapter.shutdown()
