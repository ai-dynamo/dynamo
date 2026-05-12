# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BuiltinLoadPredictor`` — real PREDICT builtin.

Ports PSM's load-prediction pipeline (``_observe_traffic`` +
``_predict_load`` + ``warm_load_predictors``) into a standalone plugin.
Shares the existing ``LOAD_PREDICTORS`` dict in ``core/load/predictors``
so the algorithm choices (constant / ARIMA / Prophet / Kalman) and their
implementations are unchanged — this plugin only decouples predictor
lifecycle from ``PlannerStateMachine``.

Lifecycle
---------

- Construction: if ``optimization_target == "sla"``, instantiate three
  predictors (``num_req`` / ``isl`` / ``osl``) using the class picked
  by ``config.load_predictor``. In non-sla ("easy") mode the plugin
  exists but always emits AcceptResult — matches PSM's
  ``_is_easy`` skip.
- ``Bootstrap`` (RPC): no-op by default. Warm-up from historical
  observations is done via the Python-level helper
  ``warm_from_observations`` which the orchestrator's
  ``bootstrap_plugins`` wiring calls directly; keeping warm-up off the
  RPC path avoids serialising ``list[TrafficObservation]`` through
  proto ``bytes``.
- ``Reset`` (RPC): re-instantiates the three predictors using the
  same config — mirrors PSM's ``config.reload`` semantics.

Stage
-----

``Predict(request)``: reads ``request.context.observations.traffic``,
adds the sample to each predictor, calls ``predict_next`` on each,
and returns ``PredictStageResponse`` with a populated
``PredictionData``. On any exception (e.g. insufficient data points)
returns an empty ``PredictStageResponse`` which chain_augment treats
as ACCEPT — matching PSM's ``_diag_throughput_reason="predict_failed"``
+ ``return None`` path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from dynamo.planner.core.load.predictors import LOAD_PREDICTORS
from dynamo.planner.core.types import TrafficObservation
from dynamo.planner.plugins.builtins.base import BuiltinPluginBase
from dynamo.planner.plugins.types import (
    BootstrapRequest,
    BootstrapResponse,
    PredictionData,
    PredictStageRequest,
    PredictStageResponse,
    ResetRequest,
    ResetResponse,
)

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.plugins.orchestrator.orchestrator import (
        LocalPlannerOrchestrator,
    )

log = logging.getLogger(__name__)


class BuiltinLoadPredictor(BuiltinPluginBase):
    """PREDICT-stage builtin wrapping ``LOAD_PREDICTORS`` instances."""

    def __init__(
        self,
        orchestrator: "LocalPlannerOrchestrator",
        config: "PlannerConfig",
    ) -> None:
        super().__init__(orchestrator, config)
        self._num_req_predictor = None
        self._isl_predictor = None
        self._osl_predictor = None
        if not self._is_easy():
            self._construct_predictors()

    # ------------------------------------------------------------------
    # Predictor lifecycle helpers
    # ------------------------------------------------------------------

    def _is_easy(self) -> bool:
        return self._config.optimization_target != "sla"

    def _construct_predictors(self) -> None:
        cls = LOAD_PREDICTORS[self._config.load_predictor]
        self._num_req_predictor = cls(self._config)
        self._isl_predictor = cls(self._config)
        self._osl_predictor = cls(self._config)

    def warm_from_observations(
        self, observations: Sequence[TrafficObservation]
    ) -> None:
        """Python-level helper used by ``LocalPlannerOrchestrator``
        bootstrap wiring to prime the predictors before the first
        tick. Mirrors PSM's ``warm_load_predictors``:

        - No-op in easy mode.
        - Feed each observation through ``add_data_point``.
        - Call ``reset_idle_skip`` so the transition to live traffic
          doesn't re-enter the "skip leading zeros" state.
        """
        if self._is_easy():
            log.debug(
                "BuiltinLoadPredictor.warm_from_observations: skipping (easy mode)"
            )
            return
        assert self._num_req_predictor is not None  # narrow for type-checker
        assert self._isl_predictor is not None
        assert self._osl_predictor is not None
        for obs in observations:
            self._num_req_predictor.add_data_point(obs.num_req)
            self._isl_predictor.add_data_point(obs.isl)
            self._osl_predictor.add_data_point(obs.osl)
        for p in (
            self._num_req_predictor,
            self._isl_predictor,
            self._osl_predictor,
        ):
            p.reset_idle_skip()

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    async def Predict(
        self, request: PredictStageRequest
    ) -> PredictStageResponse:
        if self._is_easy():
            return PredictStageResponse()

        ctx = request.context
        traffic = None
        if ctx is not None and ctx.observations is not None:
            traffic = ctx.observations.traffic

        if traffic is not None:
            # PSM.on_tick feeds traffic into the predictor INSIDE
            # _advance_throughput; here we mirror that side effect on
            # the PREDICT plugin's own predictors.
            assert self._num_req_predictor is not None
            assert self._isl_predictor is not None
            assert self._osl_predictor is not None
            self._num_req_predictor.add_data_point(traffic.num_req)
            self._isl_predictor.add_data_point(traffic.isl)
            self._osl_predictor.add_data_point(traffic.osl)

        try:
            assert self._num_req_predictor is not None
            assert self._isl_predictor is not None
            assert self._osl_predictor is not None
            nr = self._num_req_predictor.predict_next()
            isl = self._isl_predictor.predict_next()
            osl = self._osl_predictor.predict_next()
        except Exception as exc:
            # Matches PSM behaviour: predict_failed → no predictions.
            log.info(
                "BuiltinLoadPredictor.Predict: predict_next failed (%s); "
                "returning ACCEPT",
                exc,
            )
            return PredictStageResponse()

        return PredictStageResponse(
            predictions=PredictionData(
                predicted_num_req=nr,
                predicted_isl=isl,
                predicted_osl=osl,
                source="builtin-load-predictor",
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle RPCs
    # ------------------------------------------------------------------

    async def Bootstrap(self, request: BootstrapRequest) -> BootstrapResponse:
        # In-process warmup uses ``warm_from_observations``. The RPC
        # form stays a no-op until an out-of-process deployment needs
        # to ship TrafficObservation history over the wire, at which
        # point we'll define a proto-level encoding for
        # ``BootstrapRequest.bootstrap_data``.
        return BootstrapResponse(ok=True)

    async def Reset(self, request: ResetRequest) -> ResetResponse:
        if not self._is_easy():
            self._construct_predictors()
        return ResetResponse(ok=True)


__all__ = ["BuiltinLoadPredictor"]
