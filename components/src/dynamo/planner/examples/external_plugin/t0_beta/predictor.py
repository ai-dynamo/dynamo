# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, regularly sampled traffic history for an external PREDICT plugin."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Sequence

from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb

Traffic = tuple[float, float, float]
Forecast = Callable[[Sequence[Traffic], float], Traffic]
logger = logging.getLogger(__name__)


class T0Predictor:
    """Forecast one interval, abstaining until a contiguous history is available.

    The synchronous forecast function runs in one background thread. Cancellation
    of a gRPC request does not stop a PyTorch kernel, so retain its task and refuse
    overlapping inference rather than queueing work or returning stale results.
    """

    def __init__(
        self,
        forecast: Forecast,
        *,
        interval_seconds: float = 60.0,
        context_length: int = 512,
        min_history: int = 32,
        quantile: float = 0.9,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive")
        if not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("quantile must be finite and between zero and one")
        if not 2 <= min_history <= context_length:
            raise ValueError("require 2 <= min_history <= context_length")
        self._forecast = forecast
        self._interval = interval_seconds
        self._min_history = min_history
        self._quantile = quantile
        self._clock = clock
        self._history: deque[Traffic] = deque(maxlen=context_length)
        self._last_sample: float | None = None
        self._last_request_id: str | None = None
        self._inference: asyncio.Task[Traffic] | None = None

    def _reset_history(self) -> None:
        self._history.clear()
        self._last_sample = None
        self._last_request_id = None

    @staticmethod
    def _report_failure(task: asyncio.Task[Traffic]) -> None:
        # Also retrieve failures from tasks whose requesting RPC was cancelled.
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("t0 inference failed: %s", error)

    async def predict(self, request: pb.PredictStageRequest) -> pb.PredictStageResponse:
        """Return only traffic predictions; the builtin supplies runtime metadata."""
        observations = request.context.observations
        if not observations.HasField("traffic"):
            self._reset_history()
            return pb.PredictStageResponse(reason="missing_traffic")
        traffic = observations.traffic
        sample: Traffic = (traffic.num_req, traffic.isl, traffic.osl)
        if (
            not math.isclose(traffic.duration_s, self._interval, rel_tol=1e-6)
            or not all(math.isfinite(value) and value >= 0 for value in sample)
            or (traffic.num_req > 0 and (traffic.isl <= 0 or traffic.osl <= 0))
        ):
            self._reset_history()
            logger.warning(
                "Discarding invalid traffic or mismatched observation window"
            )
            return pb.PredictStageResponse(reason="invalid_traffic")

        now = self._clock()
        request_id = request.context.request_id
        if request_id and request_id == self._last_request_id:
            return pb.PredictStageResponse(reason="duplicate_tick")
        if self._last_sample is not None:
            elapsed = now - self._last_sample
            # Allow scheduling jitter, but never train on overlapping ticks or
            # compress a missed observation into the next regular time step.
            if elapsed < self._interval * 0.5:
                return pb.PredictStageResponse(reason="early_tick")
            if elapsed > self._interval * 1.5:
                logger.warning("Traffic sampling gap; restarting t0 history warmup")
                self._reset_history()
        self._history.append(sample)
        self._last_sample = now
        self._last_request_id = request_id
        if len(self._history) < self._min_history:
            return pb.PredictStageResponse(reason="warming_up")
        if self._inference is not None and not self._inference.done():
            return pb.PredictStageResponse(reason="inference_busy")

        self._inference = asyncio.create_task(
            asyncio.to_thread(self._forecast, tuple(self._history), self._quantile)
        )
        self._inference.add_done_callback(self._report_failure)
        num_req, isl, osl = await asyncio.shield(self._inference)
        if not all(math.isfinite(value) for value in (num_req, isl, osl)):
            raise ValueError("t0 predictions must be finite")
        return pb.PredictStageResponse(
            predictions=pb.PredictionData(
                predicted_num_req=max(0.0, num_req),
                predicted_isl=max(1.0, isl),
                predicted_osl=max(1.0, osl),
                source="t0-beta",
            ),
            reason="predicted",
            # Keep builtin history warm and let it fill KV/accept-length fields.
            final=False,
        )

    async def close(self) -> None:
        """Wait for any in-flight model work before releasing server resources."""
        if self._inference is not None:
            # Failures have already been logged and propagated to their RPC.
            await asyncio.gather(self._inference, return_exceptions=True)
