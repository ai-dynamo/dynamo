# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditional Planner updates with off-loop predictor preparation."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.control_api import _MinimumEndpointUnavailableError
from dynamo.planner.offline.trace_data import extract_traffic_observations_from_trace
from dynamo.planner.plugins.builtins.local_planner import BuiltinLoadPredict

if TYPE_CHECKING:
    from dynamo.planner.core.base import NativePlannerBase
    from dynamo.planner.core.types import WorkerCapabilities
    from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)

PREDICTOR_FIELDS = frozenset(
    {
        "load_predictor",
        "load_predictor_log1p",
        "prophet_window_size",
        "kalman_q_level",
        "kalman_q_trend",
        "kalman_r",
        "kalman_min_points",
    }
)
RUNTIME_FIELDS = PREDICTOR_FIELDS | {
    "min_endpoint",
    "prefill_min_endpoint",
    "decode_min_endpoint",
    "min_gpu_budget",
    "max_gpu_budget",
    "max_throughput_scaling_replicas",
    "ttft_ms",
    "itl_ms",
    "load_scaling_down_sensitivity",
    "prefill_scale_up_queue_tokens",
    "prefill_scale_down_queue_tokens",
    "decode_scale_up_kv_rate",
    "decode_scale_down_kv_rate",
}
PREPARATION_TIMEOUT_SECONDS = 120.0
OPERATION_HISTORY_SIZE = 32


class ConfigurationConflict(ValueError):
    pass


class ReconfigureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    instance_id: str
    expected_revision: int = Field(ge=0)
    request_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
    )
    config: dict[str, Any]
    warmup_trace: str | None = Field(default=None, min_length=1)
    cold_start: bool = False


def _candidate_config(config: PlannerConfig, updates: dict[str, Any]) -> PlannerConfig:
    if not updates or (unknown := updates.keys() - RUNTIME_FIELDS):
        raise ValueError(
            f"provide supported runtime fields; unknown fields: {sorted(unknown) if updates else []}"
        )
    active_endpoints = {
        "agg": {"min_endpoint"},
        "disagg": {"prefill_min_endpoint", "decode_min_endpoint"},
        "prefill": {"prefill_min_endpoint"},
        "decode": {"decode_min_endpoint"},
    }[config.mode]
    if (
        inactive := (
            updates.keys()
            & {"min_endpoint", "prefill_min_endpoint", "decode_min_endpoint"}
        )
        - active_endpoints
    ):
        raise ValueError(
            f"endpoint fields are inactive in {config.mode} mode: {sorted(inactive)}"
        )
    try:
        candidate = PlannerConfig.model_validate(
            config.model_dump() | updates, strict=True
        )
    except ValidationError as error:
        # Full PlannerConfig also contains credentials; never echo that input
        # through the runtime API when a model-level validator rejects a patch.
        messages = [item["msg"] for item in error.errors(include_input=False)]
        raise ValueError("; ".join(messages)) from None
    for name in ("ttft_ms", "itl_ms", "kalman_q_level", "kalman_q_trend", "kalman_r"):
        value = getattr(candidate, name)
        if (
            not math.isfinite(value)
            or value < 0
            or (name in {"ttft_ms", "itl_ms"} and value == 0)
        ):
            raise ValueError(
                f"{name} must be finite and {'positive' if name in {'ttft_ms', 'itl_ms'} else 'nonnegative'}"
            )
    if candidate.kalman_min_points < 1:
        raise ValueError("kalman_min_points must be positive")
    if candidate.min_gpu_budget < -1 or candidate.max_gpu_budget < -1:
        raise ValueError("GPU budgets must be -1 (disabled) or nonnegative")
    for name in ("decode_scale_up_kv_rate", "decode_scale_down_kv_rate"):
        value = getattr(candidate, name)
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    return candidate


def _prepare_predictor(
    config: PlannerConfig,
    capabilities: WorkerCapabilities,
    trace: str | None,
) -> BuiltinLoadPredict:
    predictor = BuiltinLoadPredict(config, capabilities)
    if trace is not None:
        observations = extract_traffic_observations_from_trace(
            trace, config.throughput_adjustment_interval_seconds
        )
        predictor.prepare_reconfiguration(observations)
    return predictor


class PlannerReconfiguration:
    def __init__(self, planner: NativePlannerBase) -> None:
        self.planner = planner
        self.instance_id = str(uuid.uuid4())
        self._operations: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._task: asyncio.Task[None] | None = None
        self._preparation: asyncio.Task[BuiltinLoadPredict] | None = None
        self._closed = False
        self._registered_runtime: DistributedRuntime | None = None

    async def get_config(self) -> dict[str, Any]:
        async with self.planner._bounded_config_lock():
            return {
                "instance_id": self.instance_id,
                "revision": self.planner._config_generation,
                "config": self.planner.config.model_dump(include=RUNTIME_FIELDS),
                "supported_fields": sorted(RUNTIME_FIELDS),
            }

    def _check_revision(self, request: ReconfigureRequest) -> None:
        if self._closed:
            raise _MinimumEndpointUnavailableError("planner is shutting down")
        if (
            request.instance_id != self.instance_id
            or request.expected_revision != self.planner._config_generation
        ):
            raise ConfigurationConflict(
                "Planner instance or configuration revision changed; read config again"
            )

    async def submit(self, request: ReconfigureRequest) -> dict[str, Any]:
        request = request.model_copy(deep=True)
        fingerprint = json.dumps(request.model_dump(), sort_keys=True, allow_nan=False)
        async with self.planner._bounded_config_lock():
            self._check_revision_for_retry(request)
            if previous := self._operations.get(request.request_id):
                if previous["fingerprint"] != fingerprint:
                    raise ConfigurationConflict(
                        "request_id was already used for a different update"
                    )
                return self._public_operation(previous)
            self._check_revision(request)
            if any(
                task is not None and not task.done()
                for task in (self._task, self._preparation)
            ):
                raise ConfigurationConflict(
                    "another configuration update is still preparing"
                )
            candidate = _candidate_config(self.planner.config, request.config)
            self.planner.validate_runtime_candidate(candidate)
            changed_predictor = (
                any(
                    getattr(candidate, field) != getattr(self.planner.config, field)
                    for field in PREDICTOR_FIELDS
                )
                or request.warmup_trace is not None
                or request.cold_start
            )
            if request.warmup_trace is not None and request.cold_start:
                raise ValueError("provide warmup_trace or cold_start, not both")
            if changed_predictor:
                if (
                    candidate.optimization_target != "sla"
                    or not candidate.enable_throughput_scaling
                ):
                    raise ValueError(
                        "predictor updates require SLA mode with throughput scaling enabled"
                    )
                if request.warmup_trace is None and not request.cold_start:
                    raise ValueError(
                        "predictor replacement requires warmup_trace or explicit cold_start: true"
                    )
            operation = {
                "operation_id": request.request_id,
                "instance_id": self.instance_id,
                "expected_revision": request.expected_revision,
                "state": "preparing",
                "fingerprint": fingerprint,
            }
            self._operations[request.request_id] = operation
            while len(self._operations) > OPERATION_HISTORY_SIZE:
                self._operations.popitem(last=False)
            self._task = asyncio.create_task(
                self._apply(request, candidate, changed_predictor, operation)
            )
            self._task.add_done_callback(lambda task: self._finish(task, operation))
            return self._public_operation(operation)

    def _check_revision_for_retry(self, request: ReconfigureRequest) -> None:
        # A completed request remains retryable after its own revision increment.
        if self._closed:
            raise _MinimumEndpointUnavailableError("planner is shutting down")
        if request.instance_id != self.instance_id:
            raise ConfigurationConflict("Planner instance changed; read config again")

    async def _apply(self, request, candidate, changed_predictor, operation) -> None:
        predictor = None
        if changed_predictor:
            operation["state"] = (
                "warming" if request.warmup_trace is not None else "preparing"
            )
            capabilities = self.planner._build_worker_capabilities()
            self._preparation = asyncio.create_task(
                asyncio.to_thread(
                    _prepare_predictor, candidate, capabilities, request.warmup_trace
                )
            )
            # A timed-out thread cannot be killed. Keep tracking it and reject a
            # second preparation until it exits; it never owns active state.
            self._preparation.add_done_callback(_consume_preparation_result)
            predictor = await asyncio.wait_for(
                asyncio.shield(self._preparation), PREPARATION_TIMEOUT_SECONDS
            )
        async with self.planner._effect_admission_lock:
            async with self.planner._bounded_config_lock():
                self._check_revision(request)
                self.planner.validate_runtime_candidate(candidate)
                self.planner.apply_runtime_candidate(
                    candidate, request.config.keys(), predictor
                )
                operation["state"] = "applied"
                operation["applied_revision"] = self.planner._config_generation

    @staticmethod
    def _finish(task: asyncio.Task[None], operation: dict[str, Any]) -> None:
        if task.cancelled():
            operation.update(state="failed", error="planner stopped before activation")
        elif error := task.exception():
            operation.update(
                state="failed",
                error=str(error) or type(error).__name__,
                error_type="conflict"
                if isinstance(error, ConfigurationConflict)
                else type(error).__name__,
            )
            logger.warning(
                "Planner reconfiguration %s failed: %s",
                operation["operation_id"],
                error,
            )

    @staticmethod
    def _public_operation(operation: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in operation.items() if key != "fingerprint"}

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        return self._public_operation(self._operations[operation_id])

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        method, path = request["method"], request["path"]
        try:
            if path == "config":
                if method == "GET":
                    return {"status": 200, "body": await self.get_config()}
                if method == "PATCH":
                    update = ReconfigureRequest.model_validate(request["body"])
                    return {"status": 202, "body": await self.submit(update)}
                return {"status": 405, "body": {"error": "use GET or PATCH"}}
            if path.startswith("operations/"):
                if method != "GET":
                    return {"status": 405, "body": {"error": "use GET"}}
                return {
                    "status": 200,
                    "body": self.get_operation(path.removeprefix("operations/")),
                }
            return {"status": 404, "body": {"error": "control resource not found"}}
        except ConfigurationConflict as error:
            return {"status": 409, "body": {"error": str(error)}}
        except (ValueError, ValidationError) as error:
            return {"status": 422, "body": {"error": str(error)}}
        except _MinimumEndpointUnavailableError as error:
            return {"status": 503, "body": {"error": str(error)}}
        except KeyError:
            return {
                "status": 404,
                "body": {"error": "operation not retained; read current config"},
            }

    def register(self, runtime: DistributedRuntime) -> None:
        reference = weakref.ref(self)

        async def handler(request):
            controller = reference()
            if controller is None or controller._closed:
                return {"status": 503, "body": {"error": "planner is unavailable"}}
            return await controller.handle_request(request)

        runtime.register_control_handler("planner", handler)
        self._registered_runtime = runtime

    async def close(self) -> None:
        self._closed = True
        if self._registered_runtime is not None:
            self._registered_runtime.unregister_control_handler("planner")
            self._registered_runtime = None
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


def _consume_preparation_result(task: asyncio.Task[BuiltinLoadPredict]) -> None:
    if not task.cancelled():
        task.exception()
