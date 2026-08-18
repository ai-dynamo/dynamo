# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wait for exact Power Agent evidence before starting a worker backend."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, NoReturn, Sequence

REPORT_VERSION = 1
MAX_REPORT_BYTES = 64 * 1024
MAIN_CONTAINER_NAME = "main"

DGD_UID_ENV = "DYNAMO_POWER_DGD_UID"
COMPONENT_ENV = "DYNAMO_POWER_COMPONENT"
EXPECTED_GPU_COUNT_ENV = "DYNAMO_POWER_EXPECTED_GPU_COUNT"
IN_GATE_BOUND_WATTS_ENV = "DYNAMO_POWER_IN_GATE_BOUND_WATTS_PER_GPU"

POWER_GATE_VOLUME_PATH = Path("/var/run/dynamo/power-gate")
POD_UID_FILE = POWER_GATE_VOLUME_PATH / "pod-uid"
REPORT_FILE = POWER_GATE_VOLUME_PATH / "report"
TERMINATION_MESSAGE_FILE = Path("/dev/termination-log")
MAX_TERMINATION_MESSAGE_BYTES = 1024

REPORT_FRESHNESS_LIMIT = timedelta(minutes=1)
GATE_TIMEOUT_SECONDS = 180.0
POLL_INTERVAL_SECONDS = 0.5

_TOP_LEVEL_FIELDS = frozenset(
    {
        "version",
        "dgdUID",
        "component",
        "podUID",
        "node",
        "allocationID",
        "gpus",
    }
)
_GPU_FIELDS = frozenset(
    {
        "uuid",
        "requestedWatts",
        "targetWatts",
        "constraintMinWatts",
        "constraintMaxWatts",
        "policyOutcome",
        "writeOutcome",
        "readbackOutcome",
        "enforcedCapWatts",
        "actuator",
        "observedAt",
    }
)
_ACTUATORS = frozenset({"nvml", "dcgm"})


class GateConfigurationError(ValueError):
    """The operator-injected gate inputs or original command are invalid."""


class GateRejection(ValueError):
    """A report candidate does not prove safe backend startup."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class GateTimeout(RuntimeError):
    """No acceptable report arrived within the bounded startup wait."""

    def __init__(self, last_reason: str) -> None:
        super().__init__(last_reason)
        self.last_reason = last_reason


@dataclass(frozen=True)
class GateContext:
    """Immutable identity and bound injected by the operator."""

    dgd_uid: str
    component: str
    expected_gpu_count: int
    in_gate_bound_watts_per_gpu: int

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "GateContext":
        values: dict[str, str] = {}
        for name in (
            DGD_UID_ENV,
            COMPONENT_ENV,
            EXPECTED_GPU_COUNT_ENV,
            IN_GATE_BOUND_WATTS_ENV,
        ):
            value = environment.get(name)
            if value is None or not value.strip():
                raise GateConfigurationError(
                    f"missing reserved environment variable {name}"
                )
            values[name] = value

        expected_gpu_count = _positive_decimal(
            values[EXPECTED_GPU_COUNT_ENV], EXPECTED_GPU_COUNT_ENV
        )
        in_gate_bound = _positive_decimal(
            values[IN_GATE_BOUND_WATTS_ENV], IN_GATE_BOUND_WATTS_ENV
        )
        return cls(
            dgd_uid=values[DGD_UID_ENV],
            component=values[COMPONENT_ENV],
            expected_gpu_count=expected_gpu_count,
            in_gate_bound_watts_per_gpu=in_gate_bound,
        )


@dataclass(frozen=True)
class GateConfig:
    """Bounded wait parameters and Downward API file locations."""

    context: GateContext
    pod_uid_file: Path = POD_UID_FILE
    report_file: Path = REPORT_FILE
    freshness_limit: timedelta = REPORT_FRESHNESS_LIMIT
    timeout_seconds: float = GATE_TIMEOUT_SECONDS
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if self.freshness_limit <= timedelta(0):
            raise GateConfigurationError("report freshness limit must be positive")
        if self.timeout_seconds < 0:
            raise GateConfigurationError("gate timeout must be nonnegative")
        if self.poll_interval_seconds <= 0:
            raise GateConfigurationError("gate poll interval must be positive")


def _positive_decimal(value: str, name: str) -> int:
    if not value.isdecimal():
        raise GateConfigurationError(f"{name} must be a positive decimal integer")
    parsed = int(value)
    if parsed <= 0:
        raise GateConfigurationError(f"{name} must be a positive decimal integer")
    return parsed


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_bounded_text(path: Path, byte_limit: int, missing_reason: str) -> str:
    try:
        with path.open("rb") as stream:
            encoded = stream.read(byte_limit + 1)
    except OSError as exc:
        raise GateRejection(missing_reason) from exc
    if not encoded:
        raise GateRejection(missing_reason)
    if len(encoded) > byte_limit:
        raise GateRejection("report_oversized")
    try:
        return encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateRejection("report_malformed") from exc


def _positive_report_integer(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise GateRejection("report_malformed")
    return value


def _parse_observed_at(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise GateRejection("report_malformed")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        observed_at = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise GateRejection("report_malformed") from exc
    if observed_at.tzinfo is None:
        raise GateRejection("report_malformed")
    return observed_at.astimezone(timezone.utc)


def validate_report(
    report: object,
    *,
    context: GateContext,
    pod_uid: str,
    now: datetime,
    freshness_limit: timedelta,
) -> None:
    """Reject unless every assigned GPU has fresh, safe enforcement evidence."""
    if not isinstance(report, dict) or set(report) != _TOP_LEVEL_FIELDS:
        raise GateRejection("report_malformed")
    if type(report["version"]) is not int or report["version"] != REPORT_VERSION:
        raise GateRejection("report_version_mismatch")
    if report["dgdUID"] != context.dgd_uid:
        raise GateRejection("dgd_identity_mismatch")
    if report["component"] != context.component:
        raise GateRejection("component_identity_mismatch")
    if report["podUID"] != pod_uid:
        raise GateRejection("pod_identity_mismatch")
    if not isinstance(report["node"], str) or not report["node"]:
        raise GateRejection("report_malformed")
    if not isinstance(report["allocationID"], str) or not report["allocationID"]:
        raise GateRejection("allocation_identity_mismatch")

    gpus = report["gpus"]
    if not isinstance(gpus, list) or len(gpus) != context.expected_gpu_count:
        raise GateRejection("gpu_count_mismatch")

    now_utc = now.astimezone(timezone.utc)
    gpu_uuids: list[str] = []
    for gpu in gpus:
        if not isinstance(gpu, dict) or set(gpu) != _GPU_FIELDS:
            raise GateRejection("report_malformed")
        gpu_uuid = gpu["uuid"]
        if not isinstance(gpu_uuid, str) or not gpu_uuid or gpu_uuid in gpu_uuids:
            raise GateRejection("allocation_identity_mismatch")
        gpu_uuids.append(gpu_uuid)

        requested_watts = _positive_report_integer(gpu["requestedWatts"])
        target_watts = _positive_report_integer(gpu["targetWatts"])
        constraint_min_watts = _positive_report_integer(gpu["constraintMinWatts"])
        constraint_max_watts = _positive_report_integer(gpu["constraintMaxWatts"])
        invalid_constraint = constraint_max_watts < constraint_min_watts
        target_out_of_bounds = not (
            constraint_min_watts <= target_watts <= constraint_max_watts
        )
        if invalid_constraint or target_out_of_bounds or requested_watts <= 0:
            raise GateRejection("report_malformed")

        if gpu["policyOutcome"] != "annotated":
            raise GateRejection("policy_outcome_not_annotated")
        if gpu["writeOutcome"] != "succeeded":
            raise GateRejection("write_not_succeeded")
        if gpu["readbackOutcome"] != "succeeded":
            raise GateRejection("readback_not_succeeded")
        if not isinstance(gpu["actuator"], str) or gpu["actuator"] not in _ACTUATORS:
            raise GateRejection("report_malformed")

        enforced_cap_watts = _positive_report_integer(gpu["enforcedCapWatts"])
        if not constraint_min_watts <= enforced_cap_watts <= constraint_max_watts:
            raise GateRejection("report_malformed")
        if enforced_cap_watts > context.in_gate_bound_watts_per_gpu:
            raise GateRejection("enforced_cap_above_bound")

        observed_at = _parse_observed_at(gpu["observedAt"])
        age = now_utc - observed_at
        if age < timedelta(0) or age > freshness_limit:
            raise GateRejection("report_not_fresh")

    expected_allocation_id = (
        f"{pod_uid}/{MAIN_CONTAINER_NAME}/{','.join(sorted(gpu_uuids))}"
    )
    if report["allocationID"] != expected_allocation_id:
        raise GateRejection("allocation_identity_mismatch")


def _load_and_validate_report(config: GateConfig, now: datetime) -> None:
    pod_uid = _read_bounded_text(config.pod_uid_file, 1024, "pod_uid_missing").strip()
    if not pod_uid:
        raise GateRejection("pod_uid_missing")
    encoded_report = _read_bounded_text(
        config.report_file, MAX_REPORT_BYTES, "report_missing"
    )
    try:
        report = json.loads(encoded_report)
    except ValueError as exc:
        raise GateRejection("report_malformed") from exc
    validate_report(
        report,
        context=config.context,
        pod_uid=pod_uid,
        now=now,
        freshness_limit=config.freshness_limit,
    )


def wait_for_report(
    config: GateConfig,
    *,
    now: Callable[[], datetime] = _utc_now,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll the projected report until it is safe or the wait expires."""
    deadline = monotonic() + config.timeout_seconds
    last_reason = "report_missing"
    while True:
        try:
            _load_and_validate_report(config, now())
            return
        except GateRejection as exc:
            last_reason = exc.reason

        remaining = deadline - monotonic()
        if remaining <= 0:
            raise GateTimeout(last_reason)
        sleep(min(config.poll_interval_seconds, remaining))


def original_command(argv: Sequence[str]) -> tuple[str, ...]:
    """Return the structurally wrapped command after the required separator."""
    if len(argv) < 2 or argv[0] != "--" or not argv[1]:
        raise GateConfigurationError("expected -- followed by the original command")
    return tuple(argv[1:])


def _emit_failure(kind: str, detail: object) -> None:
    """Expose one bounded stable failure through logs and Pod termination status."""
    normalized_detail = " ".join(str(detail).splitlines()).strip()
    message = f"dynamo-power-gate: {kind}: {normalized_detail}"
    encoded = message.encode("utf-8")[:MAX_TERMINATION_MESSAGE_BYTES]
    bounded_message = encoded.decode("utf-8", errors="ignore")
    print(bounded_message, file=sys.stderr)
    try:
        TERMINATION_MESSAGE_FILE.write_text(
            f"{bounded_message}\n",
            encoding="utf-8",
        )
    except OSError:
        # Failure reporting must never hide the original stable exit reason.
        pass


def run_gate(
    command: Sequence[str],
    config: GateConfig,
    *,
    exec_process: Callable[[str, Sequence[str]], NoReturn] = os.execvp,
    now: Callable[[], datetime] = _utc_now,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> NoReturn:
    """Wait for enforcement, then replace this process with the backend."""
    if not command or not command[0]:
        raise GateConfigurationError("original command is empty")
    wait_for_report(
        config,
        now=now,
        monotonic=monotonic,
        sleep=sleep,
    )
    exec_process(command[0], command)
    raise RuntimeError("exec_process returned unexpectedly")


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``dynamo-power-gate``."""
    arguments = sys.argv[1:] if argv is None else argv
    try:
        command = original_command(arguments)
        context = GateContext.from_environment(os.environ)
        config = GateConfig(context=context)
        run_gate(command, config)
    except GateConfigurationError as exc:
        _emit_failure("configuration_error", exc)
        return 2
    except GateTimeout as exc:
        _emit_failure("enforcement_timeout", exc.last_reason)
        return 1
    except OSError as exc:
        _emit_failure("exec_failed", exc)
        return 127 if isinstance(exc, FileNotFoundError) else 126
    return 0
