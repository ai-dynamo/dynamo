# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``BuiltinThroughputPropose._last_throughput_diagnostics``
(PR 8 follow-up A2 — Accept-path observability parity with PSM).

Mirrors ``test_load_propose_diagnostics.py``: every exit branch of
``Propose`` / ``_throughput_*`` / ``_compute_*_replicas`` must leave a
reason populated under the right slot so
``OrchestratorEngineAdapter._project_throughput_diagnostics`` can
project it onto ``TickDiagnostics.throughput_decision_reason*`` without
falling back to None.

Vocabulary asserted matches PSM's ``_diag_throughput_reason`` strings
(``disabled``/``no_traffic_data``/``predict_failed``/``model_not_ready``/
``set_lower_bound``/``scale``) so existing dashboards keep working.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    WorkerCapabilities,
)
from dynamo.planner.plugins.builtins.throughput_propose import (
    BuiltinThroughputPropose,
)
from dynamo.planner.plugins.types import (
    ObservationData,
    PipelineContext,
    PredictionData,
    ProposeStageRequest,
    TrafficMetrics,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeOrch:
    def __init__(
        self,
        caps: WorkerCapabilities | None = None,
        regressions: dict | None = None,
    ) -> None:
        self._caps = caps
        self._reg = regressions or {}
        self._lower_bounds: dict[str, int] = {"prefill": 1, "decode": 1}

    @property
    def capabilities(self):
        return self._caps

    def get_regression(self, kind):
        return self._reg.get(kind)

    def update_regression(self, kind, model):
        self._reg[kind] = model

    def set_throughput_lower_bound(self, side: str, value: int) -> None:
        self._lower_bounds[side] = value

    def get_throughput_lower_bound(self, side: str) -> int:
        return self._lower_bounds.get(side, 1)


def _config(
    mode: str = "disagg",
    *,
    enable_throughput: bool = True,
    enable_load: bool = False,
    optimization_target: str = "sla",
) -> PlannerConfig:
    return PlannerConfig(
        environment="kubernetes",
        mode=mode,
        enable_throughput_scaling=enable_throughput,
        enable_load_scaling=enable_load,
        optimization_target=optimization_target,
        max_gpu_budget=-1,
        min_endpoint=1,
    )


def _caps(mode: str) -> WorkerCapabilities:
    e = EngineCapabilities(
        num_gpu=1,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        context_length=4096,
        max_kv_tokens=16384,
    )
    if mode == "prefill":
        return WorkerCapabilities(prefill=e)
    if mode == "decode" or mode == "agg":
        return WorkerCapabilities(decode=e)
    return WorkerCapabilities(prefill=e, decode=e)


def _ctx(*, num_req: float = 100, isl: float = 500, osl: float = 100,
         duration_s: float = 10.0, with_predictions: bool = True,
         with_traffic: bool = True) -> PipelineContext:
    obs = ObservationData(
        traffic=(
            TrafficMetrics(
                duration_s=duration_s, num_req=num_req, isl=isl, osl=osl
            )
            if with_traffic else None
        )
    )
    preds = (
        PredictionData(
            predicted_num_req=num_req, predicted_isl=isl, predicted_osl=osl,
        )
        if with_predictions else PredictionData()
    )
    return PipelineContext(observations=obs, predictions=preds)


def _plugin(
    cfg: PlannerConfig, caps: WorkerCapabilities | None = None,
    regressions: dict | None = None,
) -> BuiltinThroughputPropose:
    orch = _FakeOrch(caps=caps, regressions=regressions)
    return BuiltinThroughputPropose(orch, cfg)  # type: ignore[arg-type]


def _diag(plugin: BuiltinThroughputPropose) -> dict:
    return plugin._last_throughput_diagnostics


# ---------------------------------------------------------------------------
# Empty-init invariant
# ---------------------------------------------------------------------------


def test_diagnostics_initialized_empty():
    plugin = _plugin(_config(mode="disagg", enable_load=True))
    assert _diag(plugin) == {"agg": None, "prefill": None, "decode": None}


# ---------------------------------------------------------------------------
# Skip branches: ``disabled`` / ``predict_failed`` / ``no_traffic_data``
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_stamps_disagg_reason():
    cfg = _config(mode="disagg", enable_throughput=False, enable_load=True)
    plugin = _plugin(cfg)
    await plugin.Propose(ProposeStageRequest(context=_ctx()))
    assert _diag(plugin) == {"agg": None, "prefill": "disabled", "decode": "disabled"}


@pytest.mark.asyncio
async def test_disabled_stamps_agg_reason():
    cfg = _config(mode="agg", enable_throughput=False, enable_load=True)
    plugin = _plugin(cfg)
    await plugin.Propose(ProposeStageRequest(context=_ctx()))
    assert _diag(plugin)["agg"] == "disabled"


@pytest.mark.asyncio
async def test_predict_failed_when_no_predictions():
    cfg = _config(mode="disagg")
    plugin = _plugin(cfg)
    await plugin.Propose(
        ProposeStageRequest(context=_ctx(with_predictions=False))
    )
    d = _diag(plugin)
    assert d["prefill"] == "predict_failed"
    assert d["decode"] == "predict_failed"


@pytest.mark.asyncio
async def test_no_traffic_data_when_zero_duration():
    cfg = _config(mode="agg")
    plugin = _plugin(cfg)
    await plugin.Propose(
        ProposeStageRequest(context=_ctx(duration_s=0))
    )
    assert _diag(plugin)["agg"] == "no_traffic_data"


@pytest.mark.asyncio
async def test_predict_failed_when_pipeline_context_empty():
    cfg = _config(mode="prefill")
    plugin = _plugin(cfg)
    await plugin.Propose(ProposeStageRequest(context=PipelineContext()))
    assert _diag(plugin)["prefill"] == "predict_failed"


# ---------------------------------------------------------------------------
# ``model_not_ready`` — regression / capabilities missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_not_ready_disagg_no_regressions():
    cfg = _config(mode="disagg")
    # No regressions installed → both _compute_* return None and stamp.
    plugin = _plugin(cfg, caps=_caps("disagg"), regressions={})
    await plugin.Propose(ProposeStageRequest(context=_ctx()))
    d = _diag(plugin)
    assert d["prefill"] == "model_not_ready"
    assert d["decode"] == "model_not_ready"


@pytest.mark.asyncio
async def test_model_not_ready_agg_no_caps():
    cfg = _config(mode="agg")
    # No capabilities → ``_throughput_agg`` short-circuits with model_not_ready.
    plugin = _plugin(cfg, caps=None, regressions={})
    await plugin.Propose(ProposeStageRequest(context=_ctx()))
    assert _diag(plugin)["agg"] == "model_not_ready"


# ---------------------------------------------------------------------------
# Accept-with-decision: ``set_lower_bound`` / ``scale``
# ---------------------------------------------------------------------------


def _stub_prefill_regression(rps: float = 50.0, ttft_ms: float = 100.0) -> MagicMock:
    reg = MagicMock()
    reg.find_best_engine_prefill_rps = MagicMock(
        return_value=(rps, ttft_ms)
    )
    return reg


def _stub_decode_regression(rps: float = 30.0, itl_ms: float = 5.0) -> MagicMock:
    reg = MagicMock()
    reg.find_best_engine_decode_rps = MagicMock(
        return_value=(rps, itl_ms)
    )
    return reg


def _stub_agg_regression(rps: float = 20.0, ttft_ms: float = 100.0,
                         itl_ms: float = 5.0) -> MagicMock:
    reg = MagicMock()
    reg.find_best_engine_agg_rps = MagicMock(
        return_value=(rps, ttft_ms, itl_ms)
    )
    return reg


@pytest.mark.asyncio
async def test_set_lower_bound_disagg_when_load_scaling_enabled():
    cfg = _config(mode="disagg", enable_load=True)
    plugin = _plugin(
        cfg,
        caps=_caps("disagg"),
        regressions={
            "prefill": _stub_prefill_regression(),
            "decode": _stub_decode_regression(),
        },
    )
    resp = await plugin.Propose(ProposeStageRequest(context=_ctx()))
    # enable_load → emit Accept (lower_bound side-effect, no SET).
    assert resp.result_kind == "accept"
    d = _diag(plugin)
    assert d["prefill"] == "set_lower_bound"
    assert d["decode"] == "set_lower_bound"


@pytest.mark.asyncio
async def test_scale_disagg_when_load_scaling_disabled():
    cfg = _config(mode="disagg", enable_load=False)
    plugin = _plugin(
        cfg,
        caps=_caps("disagg"),
        regressions={
            "prefill": _stub_prefill_regression(),
            "decode": _stub_decode_regression(),
        },
    )
    resp = await plugin.Propose(ProposeStageRequest(context=_ctx()))
    assert resp.result_kind == "override"
    d = _diag(plugin)
    assert d["prefill"] == "scale"
    assert d["decode"] == "scale"


@pytest.mark.asyncio
async def test_scale_agg_routes_through_agg_slot():
    cfg = _config(mode="agg", enable_load=False)
    plugin = _plugin(
        cfg,
        caps=_caps("agg"),
        regressions={"agg": _stub_agg_regression()},
    )
    resp = await plugin.Propose(ProposeStageRequest(context=_ctx()))
    assert resp.result_kind == "override"
    # Agg scenario stamps the ``agg`` slot, NOT decode — so the adapter
    # projects to ``throughput_decision_reason`` (aggregate) directly.
    d = _diag(plugin)
    assert d["agg"] == "scale"
    assert d["prefill"] is None  # never written in agg mode


@pytest.mark.asyncio
async def test_set_lower_bound_agg_when_load_scaling_enabled():
    cfg = _config(mode="agg", enable_load=True)
    plugin = _plugin(
        cfg,
        caps=_caps("agg"),
        regressions={"agg": _stub_agg_regression()},
    )
    resp = await plugin.Propose(ProposeStageRequest(context=_ctx()))
    assert resp.result_kind == "accept"
    assert _diag(plugin)["agg"] == "set_lower_bound"


# ---------------------------------------------------------------------------
# Reset-per-tick contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnostics_reset_between_ticks():
    """Stale reason from tick N must not leak into tick N+1."""
    cfg = _config(mode="disagg", enable_throughput=False, enable_load=True)
    plugin = _plugin(cfg)
    await plugin.Propose(ProposeStageRequest(context=_ctx()))
    assert _diag(plugin)["prefill"] == "disabled"

    # Tick 2: flip to enabled-but-no-traffic — must NOT see "disabled" again.
    plugin._config.enable_throughput_scaling = True  # type: ignore[attr-defined]
    await plugin.Propose(
        ProposeStageRequest(context=_ctx(duration_s=0))
    )
    d = _diag(plugin)
    assert d["prefill"] == "no_traffic_data"
    assert d["decode"] == "no_traffic_data"
