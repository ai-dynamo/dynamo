# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Jev decision engine on Dynamo's ordinary Planner tick seam.

Only aggregate observations and bounded history enter the request. Jev chooses
an absolute target for each pool from code-generated feasible options. No native
Planner recommendation or future trace data is supplied to the model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
from autoscaling_arena.adapters._regression_bootstrap import _NoopRegressionBootstrap
from autoscaling_arena.adapters.aggregation import (
    aggregate_kv_util,
    aggregate_queue_depth,
    aggregate_queued_prefill_tokens,
    current_replica_target,
)

from dynamo.planner.core.types import (
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
    WorkerCapabilities,
)


def _probability(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _validate_answer(answer: Any, options: dict[str, Any]) -> None:
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("expected a Choice answer")
    probabilities = answer.get("probabilities")
    if (
        not isinstance(probabilities, dict)
        or set(probabilities) != set(options)
        or not all(_probability(p) for p in probabilities.values())
        or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-3)
        or not _probability(answer.get("confidence"))
        or not isinstance(answer.get("choice"), str)
        or answer["choice"] not in options
    ):
        raise ValueError("invalid Choice distribution or selection")
    if probabilities[answer["choice"]] < max(probabilities.values()):
        raise ValueError("Choice selection is not a highest-probability option")


class JevAutoscaler(_NoopRegressionBootstrap):
    """Select replica targets with TypeSafe; benchmark failures abort by default.

    ``failure_mode=hold`` is an explicit availability experiment, not a fallback
    to native Planner. API wall time is recorded but does not advance replay
    time. This adapter measures policy quality under synchronous observations.
    """

    def __init__(
        self,
        *,
        mode: str = "disagg",
        capabilities: WorkerCapabilities | None = None,
        min_prefill: int = 1,
        max_prefill: int = 16,
        min_decode: int = 1,
        max_decode: int = 8,
        poll_interval_s: float = 15,
        step: int = 1,
        model: str = "jev-1.13.0",
        timeout_s: float = 5,
        min_confidence: float = 0,
        history_ticks: int = 4,
        max_calls: int = 500,
        failure_mode: str = "raise",
        decision_log: Path | None = None,
        context: dict[str, Any] | None = None,
        client: Any = None,
    ) -> None:
        if mode not in {"agg", "disagg"}:
            raise ValueError("mode must be agg or disagg")
        for name, value in {
            "min_prefill": min_prefill,
            "max_prefill": max_prefill,
            "min_decode": min_decode,
            "max_decode": max_decode,
            "step": step,
            "history_ticks": history_ticks,
            "max_calls": max_calls,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if min_prefill > max_prefill or min_decode > max_decode:
            raise ValueError("replica minimum must not exceed maximum")
        for value in (poll_interval_s, timeout_s):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "poll interval and timeout must be finite and positive"
                )
        if not _probability(min_confidence) or failure_mode not in {"raise", "hold"}:
            raise ValueError("invalid confidence threshold or failure mode")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be nonempty")
        if client is None:
            key = os.environ.get("TYPESAFE_API_KEY", "").strip()
            if not key:
                raise ValueError("Jev requires TYPESAFE_API_KEY in the environment")
            client = httpx.AsyncClient(
                base_url="https://api.typesafe.ai",
                headers={"Authorization": f"Bearer {key}"},
                timeout=timeout_s,
                follow_redirects=False,
            )
        self._client = client
        self._mode = mode
        self._caps = capabilities
        self._bounds = {
            "prefill": (min_prefill, max_prefill),
            "decode": (min_decode, max_decode),
        }
        self._interval = poll_interval_s
        self._step = step
        self._model = model
        self._timeout = timeout_s
        self._confidence = min_confidence
        self._history: deque[dict[str, Any]] = deque(maxlen=history_ticks)
        self._max_calls = max_calls
        self._calls = 0
        self._failure_mode = failure_mode
        self._context = context or {}
        self._log = decision_log.open("x", encoding="utf-8") if decision_log else None

    def initial_tick(self, start_s: float) -> ScheduledTick:
        return ScheduledTick(
            at_s=start_s,
            need_worker_states=True,
            need_worker_fpm=True,
            need_traffic_metrics=True,
            use_full_traffic_metrics=True,
            traffic_metrics_duration_s=self._interval,
        )

    def _snapshot(self, tick_input: TickInput) -> dict[str, Any]:
        pools = {}
        for role in self._roles:
            wc = tick_input.worker_counts
            observed_pool = "all" if self._mode == "agg" else role
            fpm = tick_input.fpm_observations
            observations = getattr(fpm, role, None) if fpm else None
            ready = getattr(wc, f"ready_num_{role}", None)
            queue = (
                aggregate_queue_depth(fpm, pool=observed_pool) if observations else None
            )
            pools[role] = {
                "ready": ready,
                "expected": getattr(wc, f"expected_num_{role}", None),
                "scaling_in_progress": getattr(wc, f"{role}_scaling_in_progress", None),
                "waiting_requests": queue,
                "waiting_requests_per_ready_worker": queue / ready
                if queue is not None and ready
                else None,
                "queued_prefill_tokens": aggregate_queued_prefill_tokens(
                    fpm, pool=observed_pool
                )
                if observations
                else None,
                "mean_kv_utilization": aggregate_kv_util(
                    fpm, self._caps, pool=observed_pool
                ),
            }
        traffic = asdict(tick_input.traffic) if tick_input.traffic else None
        if traffic and traffic["duration_s"] > 0:
            traffic["requests_per_second"] = traffic["num_req"] / traffic["duration_s"]
        return {"now_s": tick_input.now_s, "pools": pools, "traffic": traffic}

    @property
    def _roles(self) -> tuple[str, ...]:
        return ("prefill", "decode") if self._mode == "disagg" else ("decode",)

    async def tick(
        self, scheduled_tick: ScheduledTick, tick_input: TickInput
    ) -> PlannerEffects:
        if self._calls >= self._max_calls:
            raise RuntimeError(
                "Jev max_calls exhausted; increase the explicit run budget"
            )
        snapshot = self._snapshot(tick_input)
        current = {
            role: current_replica_target(
                tick_input.worker_counts, role=role, minimum=self._bounds[role][0]
            )
            for role in self._roles
        }
        questions = {}
        for role in self._roles:
            lo, hi = self._bounds[role]
            if not lo <= current[role] <= hi:
                raise ValueError("observed fleet is outside configured replica bounds")
            targets = sorted(
                {
                    max(lo, current[role] - self._step),
                    current[role],
                    min(hi, current[role] + self._step),
                }
            )
            questions[role] = {
                "type": "choice",
                "instructions": (
                    f"Choose the desired total replica count for the {role} pool for the next control interval. "
                    "Protect the latency objectives while avoiding idle GPU allocation. "
                    f"Use `current.pools.{role}`, `current.traffic`, and `history` to judge whether capacity should increase, hold, or decrease. "
                    "The latency objectives and worker costs are in `context`. "
                    "Expected replicas include workers still starting; ready replicas serve traffic. "
                    "Starting workers need the configured cold-start delay. Avoid cancelling needed pending capacity. "
                    "A null observation is unknown, not zero. SLO values are objectives, not measured latency. "
                    "Prefer holding when observations are insufficient. Each option is an absolute target."
                ),
                "criteria": {
                    str(n): {
                        "target_replicas": n,
                        "change_from_expected": n - current[role],
                    }
                    for n in targets
                },
            }
        payload = {
            "model": self._model,
            "state": {
                "topology": self._mode,
                "control_interval_s": self._interval,
                "context": self._context,
                "current": snapshot,
                "history": list(self._history),
            },
            "questions": questions,
        }
        # Reject non-finite observations before sending or writing a request.
        serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
        record: dict[str, Any] = {
            "schema_version": 1,
            "tick_s": scheduled_tick.at_s,
            "request": payload,
            "request_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "status": "ok",
            "gated_pools": [],
        }
        target = dict(current)
        started = time.perf_counter()
        self._calls += 1
        failure = None
        try:

            async def evaluate():
                response = await self._client.post("/v1/systemone", json=payload)
                response.raise_for_status()
                return response.json()

            result = await asyncio.wait_for(evaluate(), timeout=self._timeout)
            if not isinstance(result, dict) or not isinstance(result.get("model"), str):
                raise TypeError("missing response model")
            if (
                self._model not in {"jev-latest", "jev-preview"}
                and result["model"] != self._model
            ):
                raise ValueError("response model differs from the pinned model")
            answers = result.get("answers")
            if not isinstance(answers, dict) or set(answers) != set(questions):
                raise ValueError("missing or unexpected answers")
            for role in self._roles:
                _validate_answer(answers[role], questions[role]["criteria"])
            if result.get("usage") is not None and not isinstance(
                result["usage"], dict
            ):
                raise TypeError("invalid token usage")
            record.update(
                model=result["model"], answers=answers, usage=result.get("usage")
            )
            for role in self._roles:
                if answers[role]["confidence"] >= self._confidence:
                    target[role] = int(answers[role]["choice"])
                else:
                    record["gated_pools"].append(role)
        except (httpx.HTTPError, ValueError, TypeError, asyncio.TimeoutError) as exc:
            # Record categories, never exception text or HTTP bodies that could
            # include credentials. A failed call leaves the whole fleet intact.
            failure = type(exc).__name__
            record.update(status="error", error_type=failure)
            target = dict(current)
        finally:
            record.update(wall_latency_s=time.perf_counter() - started, targets=target)
            if self._log:
                self._log.write(json.dumps(record, allow_nan=False) + "\n")
                self._log.flush()
        self._history.append(
            {**snapshot, "targets": target, "decision_status": record["status"]}
        )
        if failure and self._failure_mode == "raise":
            raise RuntimeError(
                f"Jev decision failed ({failure}); see decision log"
            ) from None
        return PlannerEffects(
            scale_to=ScalingDecision(
                num_prefill=target.get("prefill"), num_decode=target["decode"]
            ),
            next_tick=self.initial_tick(scheduled_tick.at_s + self._interval),
        )

    async def shutdown(self) -> None:
        try:
            await self._client.aclose()
        finally:
            if self._log:
                self._log.close()
