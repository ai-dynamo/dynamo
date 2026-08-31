#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the standalone, single-pool Batch Planner POC control loop."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import importlib
import json
import math
import re
import secrets
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Protocol, Self
from urllib.parse import urlsplit

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DYNAMO_REPO_ROOT = Path(__file__).resolve().parents[6]
DYNAMO_COMPONENTS_SRC = DYNAMO_REPO_ROOT / "components" / "src"
if DYNAMO_COMPONENTS_SRC.is_dir():
    # Keep this standalone POC runnable from a source checkout without an
    # editable ai-dynamo install. Dependencies still come from the caller's
    # selected Python environment.
    sys.path.insert(0, str(DYNAMO_COMPONENTS_SRC))
HF_CREDENTIAL_RE = re.compile(r"\bhf_[A-Za-z0-9][A-Za-z0-9_%.-]{15,}")
HF_ASSIGNMENT_RE = re.compile(
    r"(?im)(\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\b\s*[:=]\s*)[^\s,}\]]+"
)
BEARER_RE = re.compile(r"(?i)(\bBearer\s+)[^\s\"']+")
URL_USERINFO_RE = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/\s@]+@")


class ControlLoopConfigurationError(ValueError):
    """The requested loop configuration is unsafe or incomplete."""


class ControlLoopFailure(RuntimeError):
    """One control-loop phase failed after the loop started."""

    def __init__(self, phase: str, cause: BaseException) -> None:
        self.phase = phase
        self.cause = cause
        super().__init__(f"{phase} failed: {sanitize_text(str(cause))}")


def prepare_source_checkout_planner_imports() -> None:
    """Load only Planner's source-only packages when no wheel is required.

    ``dynamo.planner.__init__`` imports runtime connectors backed by the native
    ``dynamo._core`` extension. This standalone POC only needs the pure-Python
    policy and environment packages, so a source checkout can expose those
    subpackages without building the unrelated runtime extension.
    """

    planner_root = DYNAMO_COMPONENTS_SRC / "dynamo" / "planner"
    if not planner_root.is_dir() or "dynamo.planner" in sys.modules:
        return

    dynamo_package = importlib.import_module("dynamo")
    planner_package = ModuleType("dynamo.planner")
    planner_package.__file__ = str(planner_root / "__init__.py")
    planner_package.__package__ = "dynamo.planner"
    planner_package.__path__ = [str(planner_root)]
    sys.modules["dynamo.planner"] = planner_package
    dynamo_package.planner = planner_package

    # ``core.__init__`` also exports the native-runtime-backed state machine.
    # Mount the core directory as a package so this POC can import only its
    # pure policy and contract modules.
    core_package = ModuleType("dynamo.planner.core")
    core_package.__file__ = str(planner_root / "core" / "__init__.py")
    core_package.__package__ = "dynamo.planner.core"
    core_package.__path__ = [str(planner_root / "core")]
    sys.modules["dynamo.planner.core"] = core_package
    planner_package.core = core_package


class SchedulingCollector(Protocol):
    async def collect(self) -> Any: ...


class DrainLimitActuator(Protocol):
    async def apply_drain_limit(self, decision: Any) -> None: ...


class SchedulingPolicy(Protocol):
    def __call__(
        self,
        tick_input: Any,
        config: Any,
        *,
        decision_id: str,
    ) -> Any: ...


TickInputBuilder = Callable[[float, int, Any], Any]
PauseDecisionBuilder = Callable[[str, float, float, str], Any]


@dataclasses.dataclass(frozen=True)
class ControlLoopSettings:
    """Safe, non-secret inputs shared by the live runner and tests."""

    run_id: str
    pool_id: str
    work_class: str
    safe_rps_per_ready_replica: float
    ready_replicas: int
    online_offered_rps: float
    iterations: int
    interval_seconds: float
    drain_lease_duration_s: float
    apply_drain_limit: bool = False
    cold_start_margin_s: float = 0.0
    finalization_margin_s: float = 0.0
    max_observation_age_s: float = 60.0
    min_replicas: int = 0
    max_replicas: int | None = None
    max_batch_admission_rps: float | None = None
    tenant: str = "planner-poc-baseline"

    def public_record(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "work_class": self.work_class,
            "safe_rps_per_ready_replica": self.safe_rps_per_ready_replica,
            "ready_replicas": self.ready_replicas,
            "online_offered_rps": self.online_offered_rps,
            "iterations": self.iterations,
            "interval_seconds": self.interval_seconds,
            "drain_lease_duration_s": self.drain_lease_duration_s,
            "cold_start_margin_s": self.cold_start_margin_s,
            "finalization_margin_s": self.finalization_margin_s,
            "max_observation_age_s": self.max_observation_age_s,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "max_batch_admission_rps": self.max_batch_admission_rps,
            "tenant": self.tenant,
        }


class JsonlDecisionRecorder:
    """Write strict, sanitized JSON Lines without overwriting prior evidence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8", buffering=1)

    def append(self, record: Mapping[str, Any]) -> None:
        payload = to_json_value(record)
        self._handle.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class StaticOnlineTrafficSource:
    """Expose a caller-configured offered rate through the public source protocol."""

    def __init__(
        self,
        *,
        pool_id: str,
        online_offered_rps: float,
        demand_factory: Callable[..., Any],
    ) -> None:
        self._pool_id = pool_id
        self._online_offered_rps = online_offered_rps
        self._demand_factory = demand_factory

    async def collect_online_traffic(self, *, observed_at_s: float) -> list[Any]:
        return [
            self._demand_factory(
                observed_at_s=observed_at_s,
                pool_id=self._pool_id,
                online_offered_rps=self._online_offered_rps,
            )
        ]


class PrometheusHttpQuery:
    """Call Prometheus instant-query API with an existing aiohttp session."""

    def __init__(self, session: Any, base_url: str) -> None:
        self._session = session
        self._endpoint = f"{base_url.rstrip('/')}/api/v1/query"

    async def __call__(self, query: str) -> object:
        async with self._session.get(
            self._endpoint,
            params={"query": query},
        ) as response:
            response.raise_for_status()
            return await response.json()


def sanitize_text(value: str) -> str:
    """Remove common credential shapes from errors before output or recording."""

    result = URL_USERINFO_RE.sub(r"\1<redacted>@", value)
    result = HF_ASSIGNMENT_RE.sub(r"\1<redacted>", result)
    result = HF_CREDENTIAL_RE.sub("<redacted-hugging-face-credential>", result)
    return BEARER_RE.sub(r"\1<redacted>", result)


def to_json_value(value: Any) -> Any:
    """Convert dataclasses and non-finite floats into strict JSON values."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(str(value))


def utc_timestamp(value_s: float) -> str:
    """Format a wall-clock timestamp as RFC 3339 UTC."""

    return (
        dt.datetime.fromtimestamp(value_s, tz=dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def make_run_id(now_s: float | None = None, suffix: str | None = None) -> str:
    """Create a collision-resistant UTC control-loop run identifier."""

    observed_s = time.time() if now_s is None else now_s
    timestamp = dt.datetime.fromtimestamp(observed_s, tz=dt.timezone.utc)
    random_suffix = suffix or secrets.token_hex(3)
    return (
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-planner-loop-{random_suffix.lower()}"
    )


def validate_settings(settings: ControlLoopSettings) -> None:
    """Reject configurations that cannot renew a valid lease safely."""

    if not settings.run_id:
        raise ControlLoopConfigurationError("run_id must not be empty")
    if not settings.pool_id or not settings.work_class or not settings.tenant:
        raise ControlLoopConfigurationError(
            "pool_id, work_class, and tenant must not be empty"
        )
    if settings.iterations <= 0:
        raise ControlLoopConfigurationError("iterations must be positive")
    if not math.isfinite(settings.interval_seconds) or settings.interval_seconds <= 0:
        raise ControlLoopConfigurationError(
            "interval_seconds must be positive and finite"
        )
    if (
        not math.isfinite(settings.drain_lease_duration_s)
        or settings.drain_lease_duration_s <= settings.interval_seconds
    ):
        raise ControlLoopConfigurationError(
            "drain lease duration must be finite and greater than the loop interval"
        )
    if settings.ready_replicas < 0:
        raise ControlLoopConfigurationError("ready_replicas must be non-negative")
    if settings.min_replicas < 0:
        raise ControlLoopConfigurationError("min_replicas must be non-negative")
    if (
        settings.max_replicas is not None
        and settings.max_replicas < settings.min_replicas
    ):
        raise ControlLoopConfigurationError(
            "max_replicas must be greater than or equal to min_replicas"
        )
    for name, value, allow_zero in (
        (
            "safe_rps_per_ready_replica",
            settings.safe_rps_per_ready_replica,
            False,
        ),
        ("online_offered_rps", settings.online_offered_rps, True),
        ("cold_start_margin_s", settings.cold_start_margin_s, True),
        ("finalization_margin_s", settings.finalization_margin_s, True),
        ("max_observation_age_s", settings.max_observation_age_s, True),
    ):
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ControlLoopConfigurationError(
                f"{name} must be {qualifier} and finite"
            )
    if settings.max_batch_admission_rps is not None and (
        not math.isfinite(settings.max_batch_admission_rps)
        or settings.max_batch_admission_rps < 0
    ):
        raise ControlLoopConfigurationError(
            "max_batch_admission_rps must be non-negative and finite"
        )


def validate_plan(
    plan: Any,
    settings: ControlLoopSettings,
    tick_now_s: float,
    decision_id: str,
) -> None:
    """Validate the policy boundary before a Redis mutation is possible."""

    replica_floor = getattr(plan, "replica_floor", None)
    if (
        isinstance(replica_floor, bool)
        or not isinstance(replica_floor, int)
        or replica_floor < 0
    ):
        raise ValueError("policy returned an invalid replica_floor")
    decision = getattr(plan, "drain_limit", None)
    if decision is None:
        raise ValueError("policy returned no drain_limit")
    if getattr(decision, "pool_id", None) != settings.pool_id:
        raise ValueError("policy drain_limit targets the wrong pool")
    rate = getattr(decision, "max_admission_rps", None)
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate < 0
    ):
        raise ValueError("policy returned an invalid drain rate")
    valid_until_s = getattr(decision, "valid_until_s", None)
    if (
        isinstance(valid_until_s, bool)
        or not isinstance(valid_until_s, (int, float))
        or not math.isfinite(valid_until_s)
        or valid_until_s <= tick_now_s
    ):
        raise ValueError("policy returned an expired or invalid drain lease")
    if getattr(decision, "decision_id", None) != decision_id:
        raise ValueError("policy returned a mismatched decision_id")


def decision_record(
    *,
    settings: ControlLoopSettings,
    iteration: int,
    recorded_at_s: float,
    status: str,
    observation: Any,
    plan: Any,
    drain_limit_applied: bool,
    error: Mapping[str, Any] | None = None,
    fail_closed_pause: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one stable audit record without endpoint or credential fields."""

    decision = None
    diagnostics = None
    if plan is not None:
        decision = {
            "replica_floor_advisory": plan.replica_floor,
            "replica_scaling_applied": False,
            "drain_limit": plan.drain_limit,
        }
        diagnostics = plan.diagnostics
    return {
        "schema_version": "1.0",
        "run_id": settings.run_id,
        "iteration": iteration,
        "recorded_at": utc_timestamp(recorded_at_s),
        "mode": "apply" if settings.apply_drain_limit else "dry_run",
        "status": status,
        "inputs": settings.public_record(),
        "observation": observation,
        "decision": decision,
        "diagnostics": diagnostics,
        "actuation": {
            "drain_limit_applied": drain_limit_applied,
            "replica_scaling_applied": False,
        },
        "error": error,
        "fail_closed_pause": fail_closed_pause,
    }


async def _best_effort_fail_closed_pause(
    *,
    settings: ControlLoopSettings,
    phase: str,
    decision_id: str,
    actuator: DrainLimitActuator | None,
    pause_decision_builder: PauseDecisionBuilder,
    clock: Callable[[], float],
) -> dict[str, Any] | None:
    if (
        not settings.apply_drain_limit
        or actuator is None
        or phase not in {"collection", "policy"}
    ):
        return None

    result: dict[str, Any] = {
        "attempted": True,
        "applied": False,
        "decision": None,
        "error": None,
    }
    try:
        now_s = clock()
        pause = pause_decision_builder(
            settings.pool_id,
            0.0,
            now_s + settings.drain_lease_duration_s,
            f"{decision_id}-fail-closed",
        )
        result["decision"] = pause
        await actuator.apply_drain_limit(pause)
        result["applied"] = True
    except Exception as error:  # noqa: BLE001 - best-effort failure is evidence
        result["error"] = {
            "type": type(error).__name__,
            "message": sanitize_text(str(error)),
        }
    return result


async def run_control_loop(
    *,
    settings: ControlLoopSettings,
    collector: SchedulingCollector,
    policy_config: Any,
    policy: SchedulingPolicy,
    tick_input_builder: TickInputBuilder,
    pause_decision_builder: PauseDecisionBuilder,
    recorder: JsonlDecisionRecorder,
    actuator: DrainLimitActuator | None = None,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    status_sink: Callable[[str], None] | None = None,
) -> None:
    """Collect, decide, optionally renew the leased drain cap, and record."""

    validate_settings(settings)
    if settings.apply_drain_limit and actuator is None:
        raise ControlLoopConfigurationError(
            "apply mode requires a drain-limit actuator"
        )

    for iteration in range(1, settings.iterations + 1):
        if iteration > 1:
            try:
                await sleeper(settings.interval_seconds)
            except Exception as error:
                failed_at_s = clock()
                recorder.append(
                    decision_record(
                        settings=settings,
                        iteration=iteration,
                        recorded_at_s=failed_at_s,
                        status="error",
                        observation=None,
                        plan=None,
                        drain_limit_applied=False,
                        error={
                            "phase": "interval",
                            "type": type(error).__name__,
                            "message": sanitize_text(str(error)),
                        },
                    )
                )
                raise ControlLoopFailure("interval", error) from error

        decision_id = f"{settings.run_id}-{iteration:06d}"
        phase = "collection"
        observation = None
        plan = None
        tick_now_s: float | None = None
        drain_limit_applied = False
        try:
            observation = await collector.collect()
            phase = "policy"
            tick_now_s = clock()
            tick_input = tick_input_builder(
                tick_now_s,
                settings.ready_replicas,
                observation,
            )
            plan = policy(
                tick_input,
                policy_config,
                decision_id=decision_id,
            )
            validate_plan(plan, settings, tick_now_s, decision_id)
            phase = "actuation"
            if settings.apply_drain_limit:
                assert actuator is not None
                await actuator.apply_drain_limit(plan.drain_limit)
                drain_limit_applied = True
        except Exception as error:
            failure_at_s = clock()
            pause_result = await _best_effort_fail_closed_pause(
                settings=settings,
                phase=phase,
                decision_id=decision_id,
                actuator=actuator,
                pause_decision_builder=pause_decision_builder,
                clock=clock,
            )
            recorder.append(
                decision_record(
                    settings=settings,
                    iteration=iteration,
                    recorded_at_s=failure_at_s,
                    status="error",
                    observation=observation,
                    plan=plan,
                    drain_limit_applied=drain_limit_applied,
                    error={
                        "phase": phase,
                        "type": type(error).__name__,
                        "message": sanitize_text(str(error)),
                    },
                    fail_closed_pause=pause_result,
                )
            )
            raise ControlLoopFailure(phase, error) from error

        assert tick_now_s is not None
        status = "applied" if settings.apply_drain_limit else "planned"
        recorder.append(
            decision_record(
                settings=settings,
                iteration=iteration,
                recorded_at_s=tick_now_s,
                status=status,
                observation=observation,
                plan=plan,
                drain_limit_applied=drain_limit_applied,
            )
        )
        if status_sink is not None:
            status_sink(
                f"iteration {iteration}/{settings.iterations}: {status}; "
                f"drain_limit_rps={plan.drain_limit.max_admission_rps:.6g}; "
                f"lease_until={plan.drain_limit.valid_until_s:.3f}; "
                f"replica_floor_advisory={plan.replica_floor}"
            )


def _validate_http_base_url(name: str, value: str) -> None:
    try:
        parsed = urlsplit(value)
        _port = parsed.port
    except ValueError as error:
        raise ControlLoopConfigurationError(f"{name} is invalid: {error}") from error
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ControlLoopConfigurationError(f"{name} must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ControlLoopConfigurationError(
            f"{name} must not contain credentials, a query, or a fragment"
        )


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def non_negative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be non-negative and finite")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--work-class", required=True)
    parser.add_argument(
        "--safe-rps-per-ready-replica",
        required=True,
        type=positive_float,
    )
    parser.add_argument("--ready-replicas", required=True, type=non_negative_int)
    parser.add_argument(
        "--online-offered-rps",
        required=True,
        type=non_negative_float,
    )
    parser.add_argument("--batch-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--tenant", default="planner-poc-baseline")
    parser.add_argument("--iterations", type=positive_int, default=1)
    parser.add_argument(
        "--interval",
        "--interval-seconds",
        dest="interval_seconds",
        type=positive_float,
        default=10.0,
    )
    parser.add_argument("--drain-lease-seconds", type=positive_float, default=30.0)
    parser.add_argument(
        "--max-observation-age-seconds",
        type=non_negative_float,
        default=60.0,
    )
    parser.add_argument(
        "--cold-start-margin-seconds",
        type=non_negative_float,
        default=0.0,
    )
    parser.add_argument(
        "--finalization-margin-seconds",
        type=non_negative_float,
        default=0.0,
    )
    parser.add_argument("--min-replicas", type=non_negative_int, default=0)
    parser.add_argument("--max-replicas", type=non_negative_int)
    parser.add_argument(
        "--max-batch-admission-rps",
        type=non_negative_float,
    )
    parser.add_argument(
        "--prometheus-observation-window-seconds",
        type=positive_int,
        default=90,
    )
    parser.add_argument("--http-timeout-seconds", type=positive_float, default=10.0)
    parser.add_argument("--batch-page-size", type=positive_int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        help="Decision JSONL path; defaults to a unique raw experiment run",
    )
    parser.add_argument(
        "--apply-drain-limit",
        action="store_true",
        help="Apply and renew the leased Redis drain limit; default is dry-run",
    )
    parser.add_argument("--redis-url")
    parser.add_argument(
        "--redis-control-key",
        help="Exact single-pool llm-d Async control key; required in apply mode",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    _validate_http_base_url("--batch-base-url", args.batch_base_url)
    _validate_http_base_url("--prometheus-url", args.prometheus_url)
    if args.interval_seconds >= args.drain_lease_seconds:
        raise ControlLoopConfigurationError(
            "--interval must be less than --drain-lease-seconds"
        )
    if args.batch_page_size > 100:
        raise ControlLoopConfigurationError("--batch-page-size cannot exceed 100")
    max_replicas = (
        args.ready_replicas if args.max_replicas is None else args.max_replicas
    )
    if max_replicas < args.min_replicas:
        raise ControlLoopConfigurationError(
            "--max-replicas must be greater than or equal to --min-replicas"
        )
    args.max_replicas = max_replicas
    if args.apply_drain_limit and (not args.redis_url or not args.redis_control_key):
        raise ControlLoopConfigurationError(
            "--apply-drain-limit requires --redis-url and --redis-control-key"
        )
    if not args.apply_drain_limit and (args.redis_url or args.redis_control_key):
        raise ControlLoopConfigurationError(
            "Redis options require the explicit --apply-drain-limit flag"
        )


def _control_key_for_pool(
    requested_pool: str,
    configured_pool: str,
    control_key: str,
) -> str:
    if requested_pool != configured_pool:
        raise ValueError(
            f"actuator requested pool {requested_pool!r}, expected {configured_pool!r}"
        )
    return control_key


async def run_live(
    args: argparse.Namespace,
    settings: ControlLoopSettings,
    recorder: JsonlDecisionRecorder,
) -> None:
    """Wire the stable Planner public APIs to the standalone loop."""

    try:
        prepare_source_checkout_planner_imports()
        import aiohttp
        from dynamo.planner.core.batch_policy import (
            BatchSchedulingPolicyConfig,
            plan_batch_schedule,
        )
        from dynamo.planner.core.types import (
            BatchDrainLimitDecision,
            PoolTrafficDemand,
            TickInput,
            WorkerCounts,
        )
        from dynamo.planner.environment.batch import (
            BatchGatewayJobSource,
            BatchSchedulingCollector,
            LlmdAsyncPrometheusSource,
            RedisLeasedDrainLimitActuator,
        )
    except ImportError as error:
        raise RuntimeError(
            "the control loop requires the Dynamo Planner package and aiohttp"
        ) from error

    timeout = aiohttp.ClientTimeout(total=args.http_timeout_seconds)
    actuator = None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        jobs = BatchGatewayJobSource(
            base_url=args.batch_base_url,
            session=session,
            pool_resolver=lambda _job: args.pool,
            work_class_resolver=lambda _job: args.work_class,
            headers={"X-MaaS-Username": args.tenant},
            page_size=args.batch_page_size,
        )
        online = StaticOnlineTrafficSource(
            pool_id=args.pool,
            online_offered_rps=args.online_offered_rps,
            demand_factory=PoolTrafficDemand,
        )
        feedback = LlmdAsyncPrometheusSource(
            pools=[args.pool],
            query=PrometheusHttpQuery(session, args.prometheus_url),
            observation_window_s=args.prometheus_observation_window_seconds,
            max_sample_age_s=args.max_observation_age_seconds,
        )
        collector = BatchSchedulingCollector(
            batch_jobs=jobs,
            online_traffic=online,
            dispatcher_feedback=feedback,
        )
        policy_config = BatchSchedulingPolicyConfig(
            pool_id=args.pool,
            work_class=args.work_class,
            safe_rps_per_ready_replica=args.safe_rps_per_ready_replica,
            cold_start_margin_s=args.cold_start_margin_seconds,
            finalization_margin_s=args.finalization_margin_seconds,
            max_observation_age_s=args.max_observation_age_seconds,
            drain_lease_duration_s=args.drain_lease_seconds,
            min_replicas=args.min_replicas,
            max_replicas=args.max_replicas,
            max_batch_admission_rps=args.max_batch_admission_rps,
        )
        if args.apply_drain_limit:
            assert args.redis_url is not None
            assert args.redis_control_key is not None
            actuator = RedisLeasedDrainLimitActuator.from_url(
                args.redis_url,
                control_key_resolver=lambda pool_id: _control_key_for_pool(
                    pool_id,
                    args.pool,
                    args.redis_control_key,
                ),
                decode_responses=True,
            )

        try:
            await run_control_loop(
                settings=settings,
                collector=collector,
                policy_config=policy_config,
                policy=plan_batch_schedule,
                tick_input_builder=lambda now_s, ready, observation: TickInput(
                    now_s=now_s,
                    worker_counts=WorkerCounts(ready_num_decode=ready),
                    batch=observation,
                ),
                pause_decision_builder=lambda pool_id, rate, valid_until_s, decision_id: (
                    BatchDrainLimitDecision(
                        pool_id=pool_id,
                        max_admission_rps=rate,
                        valid_until_s=valid_until_s,
                        decision_id=decision_id,
                    )
                ),
                recorder=recorder,
                actuator=actuator,
                status_sink=print,
            )
        finally:
            if actuator is not None:
                await actuator.aclose()


def _startup_error_record(
    settings: ControlLoopSettings,
    error: BaseException,
) -> dict[str, Any]:
    recorded_at_s = time.time()
    return {
        "schema_version": "1.0",
        "run_id": settings.run_id,
        "iteration": 0,
        "recorded_at": utc_timestamp(recorded_at_s),
        "mode": "apply" if settings.apply_drain_limit else "dry_run",
        "status": "error",
        "inputs": settings.public_record(),
        "observation": None,
        "decision": None,
        "diagnostics": None,
        "actuation": {
            "drain_limit_applied": False,
            "replica_scaling_applied": False,
        },
        "error": {
            "phase": "startup",
            "type": type(error).__name__,
            "message": sanitize_text(str(error)),
        },
        "fail_closed_pause": None,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    live_runner: Callable[
        [argparse.Namespace, ControlLoopSettings, JsonlDecisionRecorder],
        Awaitable[None],
    ] = run_live,
) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
    except ControlLoopConfigurationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    run_id = make_run_id()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else EXPERIMENT_ROOT
        / "results"
        / "raw"
        / run_id
        / "control-loop-decisions.jsonl"
    )
    settings = ControlLoopSettings(
        run_id=run_id,
        pool_id=args.pool,
        work_class=args.work_class,
        safe_rps_per_ready_replica=args.safe_rps_per_ready_replica,
        ready_replicas=args.ready_replicas,
        online_offered_rps=args.online_offered_rps,
        iterations=args.iterations,
        interval_seconds=args.interval_seconds,
        drain_lease_duration_s=args.drain_lease_seconds,
        apply_drain_limit=args.apply_drain_limit,
        cold_start_margin_s=args.cold_start_margin_seconds,
        finalization_margin_s=args.finalization_margin_seconds,
        max_observation_age_s=args.max_observation_age_seconds,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        max_batch_admission_rps=args.max_batch_admission_rps,
        tenant=args.tenant,
    )

    try:
        with JsonlDecisionRecorder(output) as recorder:
            print(f"run ID: {run_id}")
            print(f"mode: {'apply' if args.apply_drain_limit else 'dry-run'}")
            print(f"decision records: {output}")
            try:
                asyncio.run(live_runner(args, settings, recorder))
            except ControlLoopFailure as error:
                print(f"error: {error}", file=sys.stderr)
                return 1
            except KeyboardInterrupt:
                print("control loop interrupted", file=sys.stderr)
                return 130
            except Exception as error:  # noqa: BLE001 - startup needs evidence
                recorder.append(_startup_error_record(settings, error))
                print(
                    f"error: {type(error).__name__}: {sanitize_text(str(error))}",
                    file=sys.stderr,
                )
                return 1
    except OSError as error:
        print(f"error: cannot create decision output: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
