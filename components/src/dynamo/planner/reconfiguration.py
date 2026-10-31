# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditional Planner updates with off-loop predictor preparation."""

from __future__ import annotations

import asyncio
import glob
import gzip
import io
import json
import logging
import os
import stat
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.control_api import _MinimumEndpointUnavailableError
from dynamo.planner.core.budget import minimum_power_footprint_fits
from dynamo.planner.offline.trace_data import extract_traffic_observations_from_bytes

if TYPE_CHECKING:
    from dynamo.planner.core.base import NativePlannerBase
    from dynamo.planner.core.types import TrafficObservation, WorkerCapabilities
    from dynamo.planner.plugins.builtins.local_planner import BuiltinLoadPredict
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
MAX_WARMUP_TRACE_BYTES = 16 * 1024 * 1024
MAX_WARMUP_INTERVALS = 4096


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


def candidate_config(config: PlannerConfig, updates: dict[str, Any]) -> PlannerConfig:
    """Validate ``config`` with runtime ``updates`` applied, strictly.

    Rejects unknown or mode-inactive fields and re-runs every
    ``PlannerConfig`` validator on the full merged config.
    """

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
            _dump_with_excluded(config) | updates, strict=True
        )
    except ValidationError as error:
        # Full PlannerConfig also contains credentials; never echo that input
        # through the runtime API when a model-level validator rejects a patch.
        messages = [item["msg"] for item in error.errors(include_input=False)]
        raise ValueError("; ".join(messages)) from None
    return candidate


# Backward-compatible private name.
_candidate_config = candidate_config


def _dump_with_excluded(model: BaseModel) -> dict[str, Any]:
    """``model.model_dump()`` with every ``exclude=True`` field kept.

    Field-level ``exclude`` keeps secrets and environment-derived settings
    out of serialized configs. A revalidated plain dump would lose them: a
    ``batch_scheduling.redis.url`` written in the config file would fail the
    non-advisory batch scheduling checks, and the other excluded fields would
    be read from the environment again.
    """

    dumped = model.model_dump()
    _restore_excluded(model, dumped)
    return dumped


def _restore_excluded(model: BaseModel, dumped: dict[str, Any]) -> None:
    for name, info in type(model).model_fields.items():
        value = getattr(model, name)
        if info.exclude is True:
            dumped[name] = value
        elif isinstance(value, BaseModel):
            _restore_excluded(value, dumped[name])
        elif isinstance(value, list):
            for item, item_dump in zip(value, dumped[name]):
                if isinstance(item, BaseModel):
                    _restore_excluded(item, item_dump)


def runtime_configuration_errors(
    prefill_min_endpoint: int | None,
    decode_min_endpoint: int | None,
    min_gpu_budget: int,
    max_gpu_budget: int,
    power_budget: int | None,
    capabilities: WorkerCapabilities,
) -> list[str]:
    """Return why endpoint floors and budgets cannot run on ``capabilities``.

    Checks budget ordering, the GPU footprint of the minimum endpoints against
    ``max_gpu_budget`` and, when ``power_budget`` is set and per-replica watts
    are known, their power footprint. An empty list means the values fit.
    """

    errors: list[str] = []

    if min_gpu_budget >= 0 and max_gpu_budget >= 0 and min_gpu_budget > max_gpu_budget:
        errors.append(
            f"min_gpu_budget={min_gpu_budget} exceeds "
            f"max_gpu_budget={max_gpu_budget}"
        )

    required_gpus = 0
    if prefill_min_endpoint is not None and capabilities.prefill is not None:
        p_gpu = capabilities.prefill.resolved_gpu_cost_per_replica
        if p_gpu is not None:
            required_gpus += prefill_min_endpoint * p_gpu
    if decode_min_endpoint is not None and capabilities.decode is not None:
        d_gpu = capabilities.decode.resolved_gpu_cost_per_replica
        if d_gpu is not None:
            required_gpus += decode_min_endpoint * d_gpu
    if max_gpu_budget >= 0 and required_gpus > max_gpu_budget:
        errors.append(
            "minimum endpoint footprint requires "
            f"{required_gpus} GPUs, exceeding max_gpu_budget="
            f"{max_gpu_budget}"
        )

    if power_budget is not None:
        p_watts = (
            capabilities.prefill.power_watts_per_replica
            if capabilities.prefill is not None
            else None
        )
        d_watts = (
            capabilities.decode.power_watts_per_replica
            if capabilities.decode is not None
            else None
        )
        power_known = (prefill_min_endpoint is None or p_watts is not None) and (
            decode_min_endpoint is None or d_watts is not None
        )
        prefill_floor = prefill_min_endpoint or 0
        decode_floor = decode_min_endpoint or 0
        if power_known and not minimum_power_footprint_fits(
            total_budget=power_budget,
            prefill_min_endpoint=prefill_floor,
            decode_min_endpoint=decode_floor,
            p_watts=p_watts,
            d_watts=d_watts,
        ):
            required_watts = prefill_floor * (p_watts or 0) + decode_floor * (
                d_watts or 0
            )
            errors.append(
                "minimum endpoint footprint requires "
                f"{required_watts}W, exceeding total_gpu_power_limit="
                f"{power_budget}W"
            )
    return errors


def runtime_candidate_errors(
    candidate: PlannerConfig, capabilities: WorkerCapabilities
) -> list[str]:
    """``runtime_configuration_errors`` for a full candidate configuration."""

    prefill_min_endpoint, decode_min_endpoint = candidate.active_min_endpoints()
    return runtime_configuration_errors(
        prefill_min_endpoint,
        decode_min_endpoint,
        candidate.min_gpu_budget,
        candidate.max_gpu_budget,
        candidate.total_gpu_power_limit,
        capabilities,
    )


def prepare_predictor(
    config: PlannerConfig,
    capabilities: WorkerCapabilities,
    observations: Sequence[TrafficObservation] | None,
) -> BuiltinLoadPredict:
    """Build ``config``'s load predictor, warmed with ``observations``.

    ``None`` cold-starts the predictor. A sequence warms it and raises when it
    has too few effective points, so a predictor is never swapped in cold by
    accident.
    """

    # Preserve the core's lazy plugin-loading boundary.
    from dynamo.planner.plugins.builtins.local_planner import BuiltinLoadPredict

    predictor = BuiltinLoadPredict(config, capabilities)
    if observations is not None:
        predictor.prepare_reconfiguration(list(observations))
    return predictor


def _prepare_predictor(
    config: PlannerConfig,
    capabilities: WorkerCapabilities,
    trace: str | None,
) -> BuiltinLoadPredict:
    observations = None
    if trace is not None:
        observations = _load_runtime_warmup(
            trace,
            config.throughput_adjustment_interval_seconds,
            config.runtime_warmup_trace_dir,
        )
    return prepare_predictor(config, capabilities, observations)


def _load_runtime_warmup(
    trace: str, interval: int, directory: str | None
) -> list[TrafficObservation]:
    if directory is None:
        raise ValueError(
            "configure runtime_warmup_trace_dir before supplying warmup_trace"
        )
    root = Path(directory).absolute()
    relative = Path(trace)
    if relative.is_absolute():
        try:
            relative = relative.relative_to(root)
        except ValueError:
            raise ValueError(
                "warmup_trace must be inside runtime_warmup_trace_dir"
            ) from None
    if not relative.parts or ".." in relative.parts or glob.has_magic(str(relative)):
        raise ValueError(
            "warmup_trace must name one file inside runtime_warmup_trace_dir"
        )
    try:
        payload = _read_warmup_file(root, relative)
    except OSError:
        # Do not turn caller-selected paths into a pod filesystem probe.
        raise ValueError(
            "warmup trace is unavailable in the configured directory"
        ) from None
    if trace.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
            payload = compressed.read(MAX_WARMUP_TRACE_BYTES + 1)
        if len(payload) > MAX_WARMUP_TRACE_BYTES:
            raise ValueError(
                f"decompressed warmup trace exceeds {MAX_WARMUP_TRACE_BYTES} bytes"
            )
    return extract_traffic_observations_from_bytes(
        payload, interval, max_intervals=MAX_WARMUP_INTERVALS
    )


def _read_warmup_file(root: Path, relative: Path) -> bytes:
    # Resolve each caller-controlled component relative to an open directory,
    # without following symlinks. A rename cannot redirect a later open outside
    # the configured root. Nonblocking open prevents FIFOs from stalling reads.
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            relative.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory
        )
    finally:
        os.close(directory)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("warmup_trace must name a regular file")
        if metadata.st_size > MAX_WARMUP_TRACE_BYTES:
            raise ValueError(f"warmup trace exceeds {MAX_WARMUP_TRACE_BYTES} bytes")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            payload = source.read(MAX_WARMUP_TRACE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) > MAX_WARMUP_TRACE_BYTES:
        raise ValueError(f"warmup trace exceeds {MAX_WARMUP_TRACE_BYTES} bytes")
    return payload


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
                "config": self.planner.config.model_dump(include=set(RUNTIME_FIELDS)),
                "supported_fields": sorted(RUNTIME_FIELDS),
            }

    def _check_revision(self, request: ReconfigureRequest) -> None:
        self._check_instance(request)
        if request.expected_revision != self.planner._config_generation:
            raise ConfigurationConflict(
                "Planner instance or configuration revision changed; read config again"
            )

    async def submit(self, request: ReconfigureRequest) -> dict[str, Any]:
        request = request.model_copy(deep=True)
        fingerprint = json.dumps(request.model_dump(), sort_keys=True, allow_nan=False)
        async with self.planner._bounded_config_lock():
            self._check_instance(request)
            if previous := self._operations.get(request.request_id):
                if previous["fingerprint"] != fingerprint:
                    raise ConfigurationConflict(
                        "request_id was already used for a different update"
                    )
                return self._public_operation(previous)
            self._check_revision(request)
            if self._task is not None and not self._task.done():
                raise ConfigurationConflict(
                    "another configuration update is still preparing"
                )
            candidate = candidate_config(self.planner.config, request.config)
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
                if self._preparation is not None and not self._preparation.done():
                    raise ConfigurationConflict("another predictor is still preparing")
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

    def _check_instance(self, request: ReconfigureRequest) -> None:
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

    def get_operation(self, operation_id: str) -> dict[str, Any] | None:
        operation = self._operations.get(operation_id)
        return self._public_operation(operation) if operation is not None else None

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
                operation = self.get_operation(path.removeprefix("operations/"))
                if operation is None:
                    return {
                        "status": 404,
                        "body": {
                            "error": "operation not retained; read current config"
                        },
                    }
                return {"status": 200, "body": operation}
            return {"status": 404, "body": {"error": "control resource not found"}}
        except ConfigurationConflict as error:
            return {"status": 409, "body": {"error": str(error)}}
        except ValueError as error:
            return {"status": 422, "body": {"error": str(error)}}
        except _MinimumEndpointUnavailableError as error:
            return {"status": 503, "body": {"error": str(error)}}

    def register(self, runtime: DistributedRuntime) -> None:
        reference = weakref.ref(self)

        async def handler(request):
            controller = reference()
            if controller is None:
                return {"status": 503, "body": {"error": "planner is unavailable"}}
            return await controller.handle_request(request)

        runtime.register_engine_http_handler("planner", handler)
        self._registered_runtime = runtime

    async def close(self) -> None:
        self._closed = True
        if self._registered_runtime is not None:
            self._registered_runtime.unregister_engine_http_handler("planner")
            self._registered_runtime = None
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


def _consume_preparation_result(task: asyncio.Task[BuiltinLoadPredict]) -> None:
    if not task.cancelled():
        task.exception()
