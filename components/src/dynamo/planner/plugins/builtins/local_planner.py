# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builtin plugins that implement the local planner's current algorithms."""

from __future__ import annotations

import logging
import math

from dynamo.common.forward_pass_metrics import ForwardPassMetrics
from dynamo.common.forward_pass_metrics import decode as decode_fpm
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.load.predictors import LOAD_PREDICTORS
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import (
    FpmObservations,
    ScalingDecision,
    TrafficObservation,
    WorkerCapabilities,
)
from dynamo.planner.plugins.types import (
    AcceptResult,
    ComponentTarget,
    OverrideResult,
    OverrideType,
    PredictionData,
    PredictStageRequest,
    PredictStageResponse,
    ProposeStageRequest,
    ProposeStageResponse,
    TrafficMetrics,
)

log = logging.getLogger(__name__)


def _traffic_observation(metrics: TrafficMetrics) -> TrafficObservation:
    return TrafficObservation(
        duration_s=metrics.duration_s,
        num_req=metrics.num_req,
        isl=metrics.isl,
        osl=metrics.osl,
        kv_hit_rate=metrics.kv_hit_rate,
        accept_length=metrics.accept_length,
    )


def _decision_targets(decision: ScalingDecision) -> list[ComponentTarget]:
    targets: list[ComponentTarget] = []
    if decision.num_prefill is not None:
        targets.append(
            ComponentTarget(
                sub_component_type="prefill",
                replicas=decision.num_prefill,
                type=OverrideType.SET,
            )
        )
    if decision.num_decode is not None:
        targets.append(
            ComponentTarget(
                sub_component_type="decode",
                replicas=decision.num_decode,
                type=OverrideType.SET,
            )
        )
    return targets


def _decode_fpm_map(raw: dict[str, bytes]) -> dict[tuple[str, int], ForwardPassMetrics]:
    out: dict[tuple[str, int], ForwardPassMetrics] = {}
    for key, payload in raw.items():
        worker_id, sep, dp_rank = key.rpartition("/")
        if not sep:
            log.warning("Skipping FPM payload with invalid key %r", key)
            continue
        try:
            rank = int(dp_rank)
        except ValueError:
            log.warning("Skipping FPM payload with invalid dp_rank %r", key)
            continue
        decoded = decode_fpm(payload)
        if decoded is not None:
            out[(worker_id, rank)] = decoded
    return out


def _fpm_observations(req: ProposeStageRequest) -> FpmObservations:
    ctx = req.context
    obs = ctx.observations if ctx is not None else None
    fpm = obs.fpm if obs is not None else None
    if fpm is None:
        return FpmObservations()
    prefill = _decode_fpm_map(fpm.prefill_engines) if fpm.prefill_engines else None
    decode = _decode_fpm_map(fpm.decode_engines) if fpm.decode_engines else None
    return FpmObservations(prefill=prefill, decode=decode)


class BuiltinLoadPredict:
    """PREDICT plugin for throughput-scaling traffic forecasts.

    This plugin owns the predictor history. Downstream builtins consume its
    output through ``PredictionData`` instead of reading predictor state from
    ``PlannerScalingState``.
    """

    plugin_id = "builtin_load_predict"

    def __init__(self, config: PlannerConfig, capabilities: WorkerCapabilities) -> None:
        self._config = config
        self._capabilities = capabilities
        predictor_cls = LOAD_PREDICTORS[config.load_predictor]
        self._num_req_predictor = predictor_cls(config)
        self._isl_predictor = predictor_cls(config)
        self._osl_predictor = predictor_cls(config)
        self._last_observed_isl: float | None = None
        self._last_observed_osl: float | None = None
        self._seen_live_traffic = False
        self._last_kv_hit_rate: float | None = None
        self._last_accept_length: float = 1.0

    def update_capabilities(self, capabilities: WorkerCapabilities) -> None:
        self._capabilities = capabilities
        self._last_accept_length = self._clamp_accept_length(self._last_accept_length)

    def warm_from_observations(self, observations: list[TrafficObservation]) -> None:
        if self._config.optimization_target != "sla":
            return
        for obs in observations:
            self._observe_load(obs)
        log.info("Warmed builtin load predictors with %d intervals", len(observations))
        for predictor in (
            self._num_req_predictor,
            self._isl_predictor,
            self._osl_predictor,
        ):
            if hasattr(predictor, "reset_idle_skip"):
                predictor.reset_idle_skip()
        self._seen_live_traffic = False

    def _observe_load(self, traffic: TrafficObservation) -> None:
        self._num_req_predictor.add_data_point(traffic.num_req)

        if traffic.num_req > 0:
            self._seen_live_traffic = True
            if math.isfinite(traffic.isl):
                self._last_observed_isl = traffic.isl
            if math.isfinite(traffic.osl):
                self._last_observed_osl = traffic.osl

        isl = traffic.isl
        osl = traffic.osl
        if traffic.num_req == 0 and self._seen_live_traffic:
            # Average request lengths are undefined in an idle window. Carry the
            # last observed request shape forward so time-based predictors still
            # advance once per interval without learning artificial zero lengths.
            isl = self._last_observed_isl if self._last_observed_isl is not None else 0
            osl = self._last_observed_osl if self._last_observed_osl is not None else 0

        if math.isfinite(isl):
            self._isl_predictor.add_data_point(isl)
        if math.isfinite(osl):
            self._osl_predictor.add_data_point(osl)

    def _observe_traffic(self, traffic: TrafficObservation) -> None:
        self._observe_load(traffic)
        if traffic.kv_hit_rate is not None and not math.isnan(traffic.kv_hit_rate):
            self._last_kv_hit_rate = traffic.kv_hit_rate
        if traffic.accept_length is not None and math.isfinite(traffic.accept_length):
            self._last_accept_length = self._clamp_accept_length(traffic.accept_length)

    def _predict_load(self) -> tuple[float | None, float | None, float | None]:
        try:
            nr = self._num_req_predictor.predict_next()
            isl = self._isl_predictor.predict_next()
            osl = self._osl_predictor.predict_next()
            log.info(
                "Predicted load: num_req=%.2f, isl=%.2f, osl=%.2f",
                nr,
                isl,
                osl,
            )
            return nr, isl, osl
        except (
            ArithmeticError,
            IndexError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            log.error("Failed to predict load: %s", exc)
            return None, None, None

    def _effective_speculative_nextn(self) -> int:
        d_caps = self._capabilities.decode
        if d_caps and d_caps.speculative_nextn and d_caps.speculative_nextn > 0:
            return d_caps.speculative_nextn
        return max(0, int(self._config.speculative_nextn))

    def _clamp_accept_length(self, accept_length: float | None) -> float:
        nextn = self._effective_speculative_nextn()
        if nextn <= 0:
            return 1.0
        if accept_length is None or not math.isfinite(accept_length):
            return 1.0
        return min(max(float(accept_length), 1.0), float(nextn + 1))

    async def Predict(self, req: PredictStageRequest) -> PredictStageResponse:
        # Keep builtin stage methods async: InProcessTransport runs sync plugin
        # methods in a thread pool, but these methods mutate plugin-local state.
        if not self._config.enable_throughput_scaling:
            return PredictStageResponse()
        ctx = req.context
        obs = ctx.observations if ctx is not None else None
        metrics = obs.traffic if obs is not None else None
        if metrics is None:
            return PredictStageResponse(reason="no_traffic_data")

        self._observe_traffic(_traffic_observation(metrics))
        nr, isl, osl = self._predict_load()
        if nr is None or isl is None or osl is None:
            return PredictStageResponse(reason="predict_failed")

        return PredictStageResponse(
            predictions=PredictionData(
                predicted_num_req=nr,
                predicted_isl=isl,
                predicted_osl=osl,
                predicted_kv_hit_rate=self._last_kv_hit_rate,
                predicted_accept_length=self._clamp_accept_length(
                    self._last_accept_length
                ),
                source=self.plugin_id,
            ),
            reason="predicted",
        )


class BuiltinThroughputPropose:
    """PROPOSE plugin for the current throughput-scaling algorithm."""

    plugin_id = "builtin_throughput_propose"

    def __init__(self, config: PlannerConfig, state: PlannerScalingState) -> None:
        self._config = config
        self._state = state

    async def Propose(self, req: ProposeStageRequest) -> ProposeStageResponse:
        # See BuiltinLoadPredict.Predict for why builtin methods are async.
        if not self._config.enable_throughput_scaling:
            return ProposeStageResponse(accept=AcceptResult())

        ctx = req.context
        obs = ctx.observations if ctx is not None else None
        metrics = obs.traffic if obs is not None else None
        predictions = ctx.predictions if ctx is not None else None
        if metrics is None or predictions is None:
            return ProposeStageResponse(accept=AcceptResult())

        decision = self._state.advance_throughput_from_prediction(
            _traffic_observation(metrics),
            predicted_num_req=predictions.predicted_num_req,
            predicted_isl=predictions.predicted_isl,
            predicted_osl=predictions.predicted_osl,
            predicted_kv_hit_rate=predictions.predicted_kv_hit_rate,
            predicted_accept_length=predictions.predicted_accept_length,
        )
        if decision is None:
            return ProposeStageResponse(accept=AcceptResult())

        targets = _decision_targets(decision)
        if not targets:
            return ProposeStageResponse(accept=AcceptResult())
        return ProposeStageResponse(
            override=OverrideResult(targets=targets, reason="throughput_scale")
        )


class BuiltinLoadPropose:
    """PROPOSE plugin for FPM-driven load scaling."""

    plugin_id = "builtin_load_propose"

    def __init__(self, config: PlannerConfig, state: PlannerScalingState) -> None:
        self._config = config
        self._state = state

    async def Propose(self, req: ProposeStageRequest) -> ProposeStageResponse:
        # See BuiltinLoadPredict.Predict for why builtin methods are async.
        if not self._config.enable_load_scaling:
            return ProposeStageResponse(accept=AcceptResult())

        ctx = req.context
        obs = ctx.observations if ctx is not None else None

        # Load-only deployments consume a cheap traffic observation on each
        # load tick so SLA load scaling can keep last-value KV hit rate and
        # speculative accept length fresh. Mixed mode consumes traffic in the
        # throughput/predict branch instead.
        if (
            self._config.optimization_target == "sla"
            and not self._config.enable_throughput_scaling
            and obs is not None
            and obs.traffic is not None
        ):
            traffic = _traffic_observation(obs.traffic)
            self._state.observe_runtime_metadata(
                kv_hit_rate=traffic.kv_hit_rate,
                accept_length=traffic.accept_length,
            )

        predictions = ctx.predictions if ctx is not None else None
        if (
            self._config.enable_throughput_scaling
            and obs is not None
            and obs.traffic is not None
            and predictions is not None
        ):
            # PROPOSE plugins fan out concurrently. Refresh the throughput
            # lower bound here too, before load scaling reads it, so combined
            # throughput+load ticks preserve the intended local planner ordering.
            self._state.advance_throughput_from_prediction(
                _traffic_observation(obs.traffic),
                predicted_num_req=predictions.predicted_num_req,
                predicted_isl=predictions.predicted_isl,
                predicted_osl=predictions.predicted_osl,
                predicted_kv_hit_rate=predictions.predicted_kv_hit_rate,
                predicted_accept_length=predictions.predicted_accept_length,
            )

        decision = self._state.advance_load(
            _fpm_observations(req),
            predicted_kv_hit_rate=(
                predictions.predicted_kv_hit_rate if predictions is not None else None
            ),
            predicted_accept_length=(
                predictions.predicted_accept_length if predictions is not None else None
            ),
        )
        if decision is None:
            return ProposeStageResponse(accept=AcceptResult())

        targets = _decision_targets(decision)
        if not targets:
            return ProposeStageResponse(accept=AcceptResult())
        return ProposeStageResponse(
            override=OverrideResult(targets=targets, reason="load_scale")
        )
