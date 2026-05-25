# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost-eval service core.

Holds the prefill- and agg-side regression models, runs a background loop
draining FPM events from each pool's ``FpmEventSubscriber``, and serves
slow-path requests from the KV router's RegressionConditionalPrefillPolicy.

Per-pool model. v1 doesn't do per-worker regressions; the AGG regression sees
all decode-pool FPMs and the PREFILL regression sees all prefill-pool FPMs.
That matches how Planner uses the same model classes today.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from dynamo.common.forward_pass_metrics import decode as decode_fpm
from dynamo.cost_eval.config import CostEvalConfig
from dynamo.cost_eval.wire import CostEvalRequest, CostEvalResponse

if TYPE_CHECKING:
    from dynamo.llm import FpmEventSubscriber
    from dynamo.planner.core.perf_model.agg import AggRegressionModel
    from dynamo.planner.core.perf_model.prefill import PrefillRegressionModel
    from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


# Convert seconds (regression output) to milliseconds (wire format).
_S_TO_MS = 1000.0


class CostEvalService:
    """In-process service holding the two regressions + their FPM feeders.

    Lifecycle:
        ``await service.start()`` — wires up FPM subscribers, begins draining.
        ``await service.run()``   — serves until cancelled. Combines the FPM
                                    drain loop with the endpoint handler.

    The endpoint handler is exposed via ``handle_request_bytes`` and adapted
    by ``__main__.py`` to dynamo's endpoint plane.
    """

    def __init__(self, runtime: "DistributedRuntime", config: CostEvalConfig):
        self._runtime = runtime
        self._config = config

        # Lazy-imported here (rather than at module scope) because
        # ``dynamo.planner.core.__init__`` pulls in heavy stats deps
        # (pmdarima) that the cost-eval service doesn't otherwise need.
        # Production deployments must still install the planner's container
        # deps; this only buys us a clean module-level import in dev/tests.
        from dynamo.planner.core.perf_model.agg import AggRegressionModel
        from dynamo.planner.core.perf_model.prefill import PrefillRegressionModel

        self._prefill_regression: "PrefillRegressionModel" = PrefillRegressionModel(
            max_num_fpm_samples=config.max_num_fpm_samples,
            min_observations=config.min_observations,
        )
        self._agg_regression: "AggRegressionModel" = AggRegressionModel(
            max_num_fpm_samples=config.max_num_fpm_samples,
            min_observations=config.min_observations,
        )

        self._prefill_sub: Optional["FpmEventSubscriber"] = None
        self._decode_sub: Optional["FpmEventSubscriber"] = None
        # Debug-only flags: log the first decoded FPM payload from each pool
        # so we can confirm wall_time / iter timing fields are populated.
        # If wall_time == 0.0 on every payload, the regression's filter at
        # perf_model/base.py:176 silently drops observations and warmth is
        # never reached.
        self._sampled_prefill_fpm: bool = False
        self._sampled_decode_fpm: bool = False
        # Active-FPM debug counters: log the first N FPMs with wall_time>0 from
        # each pool so we can audit the raw numbers feeding the regression.
        self._active_prefill_logged: int = 0
        self._active_decode_logged: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create + start the FPM subscribers. Idempotent."""
        from dynamo.llm import FpmEventSubscriber  # local import: PyO3 module

        if self._prefill_sub is None:
            endpoint = self._runtime.endpoint(
                f"{self._config.namespace}"
                f".{self._config.prefill_component_name}"
                f".{self._config.prefill_endpoint_name}"
            )
            self._prefill_sub = FpmEventSubscriber(endpoint)
            self._prefill_sub.start_tracking()
            logger.info(
                "Prefill FPM subscriber tracking %s.%s.%s",
                self._config.namespace,
                self._config.prefill_component_name,
                self._config.prefill_endpoint_name,
            )

        if self._decode_sub is None:
            endpoint = self._runtime.endpoint(
                f"{self._config.namespace}"
                f".{self._config.decode_component_name}"
                f".{self._config.decode_endpoint_name}"
            )
            self._decode_sub = FpmEventSubscriber(endpoint)
            self._decode_sub.start_tracking()
            logger.info(
                "Decode FPM subscriber tracking %s.%s.%s",
                self._config.namespace,
                self._config.decode_component_name,
                self._config.decode_endpoint_name,
            )

    # ------------------------------------------------------------------
    # FPM drain loop
    # ------------------------------------------------------------------

    async def _fpm_drain_loop(self) -> None:
        """Periodically pull recent FPMs from both subscribers and feed them
        into the corresponding regression's ``add_observation``."""
        tick = 0
        while True:
            prefill_added, decode_added = self._drain_once()
            tick += 1
            # Log every tick during warmup (until both warm) so we can see
            # FPM observations flowing; after warmth, log every 10s so we
            # don't spam.
            both_warm = (
                self._prefill_regression.has_sufficient_data()
                and self._agg_regression.has_sufficient_data()
            )
            if not both_warm or tick % 10 == 0:
                logger.info(
                    "fpm_drain tick=%d prefill_added=%d decode_added=%d "
                    "prefill_obs=%d decode_obs=%d "
                    "prefill_warm=%s decode_warm=%s",
                    tick,
                    prefill_added,
                    decode_added,
                    # Surface the regression's actual counted observations so we
                    # can see when add_observation silently drops FPMs (e.g.
                    # wall_time==0.0 filter at perf_model/base.py:176).
                    self._prefill_regression._total_observations,
                    self._agg_regression._total_observations,
                    self._prefill_regression.has_sufficient_data(),
                    self._agg_regression.has_sufficient_data(),
                )
            await asyncio.sleep(self._config.fpm_poll_interval_s)

    def _drain_once(self) -> tuple[int, int]:
        prefill_added = 0
        decode_added = 0
        if self._prefill_sub is not None:
            for raw in self._prefill_sub.get_recent_stats().values():
                fpm = decode_fpm(raw)
                if fpm is not None:
                    # Log up to 5 *active* (wall_time>0) FPMs from the prefill
                    # side so we can audit what the regression is actually
                    # learning from (vs the initial idle heartbeat at wall_time=0).
                    if fpm.wall_time > 0.0 and self._active_prefill_logged < 5:
                        logger.info(
                            "active prefill FPM #%d: counter_id=%d wall_time=%.6fs "
                            "sched_prefill_tokens=%d sched_num_prefill=%d "
                            "queued_prefill_tokens=%d",
                            self._active_prefill_logged,
                            fpm.counter_id,
                            fpm.wall_time,
                            fpm.scheduled_requests.sum_prefill_tokens,
                            fpm.scheduled_requests.num_prefill_requests,
                            fpm.queued_requests.sum_prefill_tokens,
                        )
                        self._active_prefill_logged += 1
                    self._prefill_regression.add_observation(fpm)
                    prefill_added += 1
        if self._decode_sub is not None:
            for raw in self._decode_sub.get_recent_stats().values():
                fpm = decode_fpm(raw)
                if fpm is not None:
                    if fpm.wall_time > 0.0 and self._active_decode_logged < 5:
                        logger.info(
                            "active decode FPM #%d: counter_id=%d wall_time=%.6fs "
                            "sched_decode_kv_tokens=%d sched_num_decode=%d",
                            self._active_decode_logged,
                            fpm.counter_id,
                            fpm.wall_time,
                            fpm.scheduled_requests.sum_decode_kv_tokens,
                            fpm.scheduled_requests.num_decode_requests,
                        )
                        self._active_decode_logged += 1
                    self._agg_regression.add_observation(fpm)
                    decode_added += 1
        return prefill_added, decode_added

    # ------------------------------------------------------------------
    # Slow-path RPC
    # ------------------------------------------------------------------

    def evaluate(self, request: CostEvalRequest) -> CostEvalResponse:
        """Pure-Python core of the slow-path RPC. Side-effect free; safe to
        call from anywhere (tests pass synthetic requests directly).
        """
        max_batched = self._config.max_num_batched_tokens

        agg_ttft_s = self._agg_regression.estimate_next_ttft(
            queued_prefill_tokens=0,  # tracked sidecar-side from FPM; TODO: derive
            max_num_batched_tokens=max_batched,
            current_decode_kv=0,  # tracked sidecar-side from FPM; TODO: derive
            kv_hit_rate=request.agg_kv_hit_rate,
        )
        disagg_ttft_s = self._prefill_regression.estimate_next_ttft(
            queued_prefill_tokens=0,  # tracked sidecar-side from FPM; TODO: derive
            max_num_batched_tokens=max_batched,
            kv_hit_rate=request.disagg_kv_hit_rate,
        )

        response = CostEvalResponse(
            agg_ttft_ms=(agg_ttft_s * _S_TO_MS if agg_ttft_s is not None else None),
            disagg_ttft_ms=(
                disagg_ttft_s * _S_TO_MS if disagg_ttft_s is not None else None
            ),
            agg_warm=self._agg_regression.has_sufficient_data(),
            disagg_warm=self._prefill_regression.has_sufficient_data(),
        )
        logger.info(
            "evaluate request_id=%r prompt_tokens=%d agg_kvh=%.3f disagg_kvh=%.3f "
            "→ agg_ttft=%s disagg_ttft=%s agg_warm=%s disagg_warm=%s",
            request.request_id,
            request.prompt_tokens,
            request.agg_kv_hit_rate,
            request.disagg_kv_hit_rate,
            f"{response.agg_ttft_ms:.2f}ms"
            if response.agg_ttft_ms is not None
            else "None",
            f"{response.disagg_ttft_ms:.2f}ms"
            if response.disagg_ttft_ms is not None
            else "None",
            response.agg_warm,
            response.disagg_warm,
        )
        return response

    def evaluate_safe(self, request: CostEvalRequest) -> CostEvalResponse:
        """Wrapper around ``evaluate`` that converts any internal exception
        into a conservative ``CostEvalResponse.unavailable()`` rather than
        propagating it to the request-plane handler (which would surface as
        a transport error on the Rust side and serialize the same fallback
        via timeout anyway, but at higher latency). Logs the exception."""
        try:
            return self.evaluate(request)
        except Exception:
            logger.exception(
                "Regression evaluate failed for request_id=%r; returning unavailable",
                request.request_id,
            )
            return CostEvalResponse.unavailable()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the FPM drain loop until cancelled.

        Endpoint serving is started separately in ``__main__.py`` so the two
        can be ``asyncio.gather``'d alongside the runtime's health endpoint.
        """
        await self._fpm_drain_loop()
