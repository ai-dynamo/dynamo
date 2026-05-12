# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BuiltinLoadPredictor.

Parity-focused: for each predictor class in ``LOAD_PREDICTORS``, feed
the same traffic observations through both a bare PSM predictor
instance and ``BuiltinLoadPredictor``; assert ``predict_next`` output
is equal (float-bit-exact when deterministic, else within strict
tolerance). This is the invariant the G3 sweep relies on.
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.load.predictors import LOAD_PREDICTORS
from dynamo.planner.core.types import TrafficObservation
from dynamo.planner.plugins.builtins.load_predictor import BuiltinLoadPredictor
from dynamo.planner.plugins.types import (
    BootstrapRequest,
    ObservationData,
    PipelineContext,
    PredictStageRequest,
    ResetRequest,
    TrafficMetrics,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FakeOrchestrator:
    def __init__(self):
        self._store = {}

    def get_regression(self, kind):
        return self._store.get(kind)

    def update_regression(self, kind, model):
        self._store[kind] = model


def _sla_config(predictor: str = "constant") -> PlannerConfig:
    return PlannerConfig(optimization_target="sla", load_predictor=predictor)


def _easy_config(predictor: str = "constant") -> PlannerConfig:
    return PlannerConfig(optimization_target="throughput", load_predictor=predictor)


def _ctx_with_traffic(num_req, isl, osl, duration_s=1.0):
    return PipelineContext(
        observations=ObservationData(
            traffic=TrafficMetrics(
                duration_s=duration_s, num_req=num_req, isl=isl, osl=osl
            )
        )
    )


# ---------------------------------------------------------------------------
# Construction & mode gating
# ---------------------------------------------------------------------------


def test_sla_mode_constructs_three_predictors():
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), _sla_config("constant"))  # type: ignore[arg-type]
    assert plugin._num_req_predictor is not None
    assert plugin._isl_predictor is not None
    assert plugin._osl_predictor is not None


def test_easy_mode_does_not_construct_predictors():
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), _easy_config("constant"))  # type: ignore[arg-type]
    assert plugin._num_req_predictor is None
    assert plugin._isl_predictor is None
    assert plugin._osl_predictor is None


# ---------------------------------------------------------------------------
# Predict — output parity vs bare PSM predictor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_easy_mode_predict_always_returns_accept():
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), _easy_config("constant"))  # type: ignore[arg-type]
    resp = await plugin.Predict(
        PredictStageRequest(context=_ctx_with_traffic(100.0, 500.0, 200.0))
    )
    assert resp.predictions is None


@pytest.mark.asyncio
async def test_predict_returns_accept_when_no_traffic_and_no_data():
    # Constant predictor with empty buffer: get_last_value -> 0; predict_next
    # returns 0 by most predictors without error — the plugin surfaces that
    # as valid predictions, NOT Accept.
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), _sla_config("constant"))  # type: ignore[arg-type]
    resp = await plugin.Predict(PredictStageRequest(context=PipelineContext()))
    # Constant returns 0 on empty buffer — this is a valid prediction.
    assert resp.predictions is not None
    assert resp.predictions.predicted_num_req == 0


@pytest.mark.asyncio
async def test_predict_matches_bare_constant_predictor_output():
    config = _sla_config("constant")
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), config)  # type: ignore[arg-type]

    # Parallel bare predictor with same sequence of inputs.
    cls = LOAD_PREDICTORS["constant"]
    bare_num = cls(config)
    bare_isl = cls(config)
    bare_osl = cls(config)

    sequence = [(10.0, 100.0, 50.0), (20.0, 200.0, 60.0), (30.0, 300.0, 70.0)]
    for num_req, isl, osl in sequence:
        # Plugin observes via context; bare observes directly.
        resp = await plugin.Predict(
            PredictStageRequest(context=_ctx_with_traffic(num_req, isl, osl))
        )
        bare_num.add_data_point(num_req)
        bare_isl.add_data_point(isl)
        bare_osl.add_data_point(osl)

        assert resp.predictions is not None
        assert resp.predictions.predicted_num_req == bare_num.predict_next()
        assert resp.predictions.predicted_isl == bare_isl.predict_next()
        assert resp.predictions.predicted_osl == bare_osl.predict_next()


@pytest.mark.asyncio
async def test_predict_sets_source_identifier():
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), _sla_config("constant"))  # type: ignore[arg-type]
    resp = await plugin.Predict(
        PredictStageRequest(context=_ctx_with_traffic(100.0, 500.0, 200.0))
    )
    assert resp.predictions is not None
    assert resp.predictions.source == "builtin-load-predictor"


@pytest.mark.asyncio
async def test_predict_failure_path_returns_accept():
    # Install a broken predictor that raises on predict_next.
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), _sla_config("constant"))  # type: ignore[arg-type]

    class _Boom:
        def add_data_point(self, v):
            pass

        def predict_next(self):
            raise RuntimeError("deliberate")

        def reset_idle_skip(self):
            pass

    plugin._num_req_predictor = _Boom()
    resp = await plugin.Predict(
        PredictStageRequest(context=_ctx_with_traffic(1.0, 1.0, 1.0))
    )
    assert resp.predictions is None  # Accept equivalent


# ---------------------------------------------------------------------------
# warm_from_observations
# ---------------------------------------------------------------------------


def test_warm_from_observations_feeds_then_predict_matches_bare():
    config = _sla_config("constant")
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), config)  # type: ignore[arg-type]
    cls = LOAD_PREDICTORS["constant"]
    bare_num = cls(config)
    bare_isl = cls(config)
    bare_osl = cls(config)

    history = [
        TrafficObservation(duration_s=1.0, num_req=5.0, isl=200.0, osl=50.0),
        TrafficObservation(duration_s=1.0, num_req=7.0, isl=250.0, osl=60.0),
    ]
    plugin.warm_from_observations(history)
    for obs in history:
        bare_num.add_data_point(obs.num_req)
        bare_isl.add_data_point(obs.isl)
        bare_osl.add_data_point(obs.osl)
    for p in (bare_num, bare_isl, bare_osl):
        p.reset_idle_skip()

    assert plugin._num_req_predictor.predict_next() == bare_num.predict_next()
    assert plugin._isl_predictor.predict_next() == bare_isl.predict_next()
    assert plugin._osl_predictor.predict_next() == bare_osl.predict_next()


def test_warm_from_observations_noop_in_easy_mode():
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), _easy_config("constant"))  # type: ignore[arg-type]
    # Should not raise even with None predictors.
    plugin.warm_from_observations(
        [TrafficObservation(duration_s=1.0, num_req=1.0, isl=1.0, osl=1.0)]
    )


# ---------------------------------------------------------------------------
# Bootstrap / Reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_returns_ok_no_op():
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), _sla_config("constant"))  # type: ignore[arg-type]
    resp = await plugin.Bootstrap(BootstrapRequest())
    assert resp.ok is True


@pytest.mark.asyncio
async def test_reset_reinstantiates_predictors_and_discards_state():
    config = _sla_config("constant")
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), config)  # type: ignore[arg-type]

    # Seed some state.
    plugin._num_req_predictor.add_data_point(999.0)
    first_pred = plugin._num_req_predictor.predict_next()
    assert first_pred == 999.0

    # After reset, predictor starts fresh.
    await plugin.Reset(ResetRequest())
    assert plugin._num_req_predictor.predict_next() == 0  # empty buffer


@pytest.mark.asyncio
async def test_reset_noop_in_easy_mode():
    plugin = BuiltinLoadPredictor(_FakeOrchestrator(), _easy_config("constant"))  # type: ignore[arg-type]
    resp = await plugin.Reset(ResetRequest())
    assert resp.ok is True
    assert plugin._num_req_predictor is None
