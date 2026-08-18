# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, identity-bound Power Agent reports on workload Pods."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from actuator import ApplyResult

REPORT_ANNOTATION_KEY = "dynamo.nvidia.com/gpu-power-enforcement-report"
REPORT_VERSION = 1
MAX_REPORT_BYTES = 64 * 1024
REPORT_REFRESH_INTERVAL_S = 30
PATCH_CONFLICT_RETRIES = 3
PATCH_REQUEST_TIMEOUT_S = 5

MAIN_CONTAINER_NAME = "main"
DGD_UID_ENV = "DYNAMO_POWER_DGD_UID"
COMPONENT_ENV = "DYNAMO_POWER_COMPONENT"
EXPECTED_GPU_COUNT_ENV = "DYNAMO_POWER_EXPECTED_GPU_COUNT"
IN_GATE_BOUND_WATTS_ENV = "DYNAMO_POWER_IN_GATE_BOUND_WATTS_PER_GPU"
POWER_GATE_ENV_NAMES = frozenset(
    {
        DGD_UID_ENV,
        COMPONENT_ENV,
        EXPECTED_GPU_COUNT_ENV,
        IN_GATE_BOUND_WATTS_ENV,
    }
)


@dataclass(frozen=True)
class PowerGateContext:
    dgd_uid: str
    component: str
    expected_gpu_count: int
    in_gate_bound_watts_per_gpu: int


def _main_container(pod):
    for container in getattr(getattr(pod, "spec", None), "containers", []) or []:
        if getattr(container, "name", "") == MAIN_CONTAINER_NAME:
            return container
    return None


def power_gate_context_from_pod(pod) -> Optional[PowerGateContext]:
    """Return injected transactional context, or ``None`` for a static Pod.

    Presence of any reserved variable makes the Pod transactional. A partial
    or invalid set is rejected instead of falling back to the Phase 1 path.
    """
    container = _main_container(pod)
    if container is None:
        return None
    reserved = [
        item
        for item in (getattr(container, "env", None) or [])
        if getattr(item, "name", "") in POWER_GATE_ENV_NAMES
    ]
    values = {
        getattr(item, "name", ""): getattr(item, "value", None) for item in reserved
    }
    if not values:
        return None
    if len(values) != len(reserved):
        raise ValueError("transactional Pod has duplicate power-gate context")
    if set(values) != POWER_GATE_ENV_NAMES or any(
        not isinstance(values[name], str) or not values[name]
        for name in POWER_GATE_ENV_NAMES
    ):
        raise ValueError("transactional Pod has incomplete power-gate context")
    try:
        expected_gpu_count = int(values[EXPECTED_GPU_COUNT_ENV])
        in_gate_bound = int(values[IN_GATE_BOUND_WATTS_ENV])
    except (TypeError, ValueError) as exc:
        raise ValueError("transactional Pod has non-integer power-gate bounds") from exc
    if expected_gpu_count <= 0 or in_gate_bound <= 0:
        raise ValueError("transactional Pod power-gate bounds must be positive")

    labels = getattr(getattr(pod, "metadata", None), "labels", None) or {}
    labeled_component = labels.get("nvidia.com/dynamo-component", "")
    if labeled_component != values[COMPONENT_ENV]:
        raise ValueError("transactional Pod component does not match its label")
    return PowerGateContext(
        dgd_uid=values[DGD_UID_ENV],
        component=values[COMPONENT_ENV],
        expected_gpu_count=expected_gpu_count,
        in_gate_bound_watts_per_gpu=in_gate_bound,
    )


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("ApplyResult observed_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_report(
    *,
    context: PowerGateContext,
    pod_uid: str,
    node_name: str,
    allocation_id: str,
    allocation_gpu_uuids: Iterable[str],
    results: Iterable[ApplyResult],
) -> dict:
    """Build one atomic report after every allocated GPU produced a result."""
    expected = tuple(sorted(allocation_gpu_uuids))
    if not pod_uid or not node_name or not allocation_id or not expected:
        raise ValueError("complete Pod and allocation identity is required")
    if len(set(expected)) != len(expected):
        raise ValueError("allocation GPU UUIDs must be unique")

    by_uuid: dict[str, ApplyResult] = {}
    for result in results:
        if not result.gpu_uuid or result.gpu_uuid in by_uuid:
            raise ValueError("ApplyResult GPU UUIDs must be nonempty and unique")
        by_uuid[result.gpu_uuid] = result
    if tuple(sorted(by_uuid)) != expected:
        raise ValueError("ApplyResults must exactly cover the allocation GPU UUID set")
    if len(expected) != context.expected_gpu_count:
        raise ValueError("allocation GPU count does not match injected expectation")

    gpus = []
    for gpu_uuid in expected:
        result = by_uuid[gpu_uuid]
        gpus.append(
            {
                "uuid": result.gpu_uuid,
                "requestedWatts": result.requested_watts,
                "targetWatts": result.target_watts,
                "constraintMinWatts": result.constraint_min_watts,
                "constraintMaxWatts": result.constraint_max_watts,
                "policyOutcome": result.policy_outcome,
                "writeOutcome": result.write_outcome,
                "readbackOutcome": result.readback_outcome,
                "enforcedCapWatts": result.enforced_cap_watts,
                "actuator": result.actuator,
                "observedAt": _rfc3339(result.observed_at),
            }
        )
    return {
        "version": REPORT_VERSION,
        "dgdUID": context.dgd_uid,
        "component": context.component,
        "podUID": pod_uid,
        "node": node_name,
        "allocationID": allocation_id,
        "gpus": gpus,
    }


def encode_report(report: dict) -> str:
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    size = len(encoded.encode("utf-8"))
    if size > MAX_REPORT_BYTES:
        raise ValueError(f"Agent report size {size} exceeds {MAX_REPORT_BYTES} bytes")
    return encoded


def _semantic_report(report: dict) -> dict:
    semantic = dict(report)
    semantic["gpus"] = [
        {key: value for key, value in gpu.items() if key != "observedAt"}
        for gpu in report.get("gpus", [])
    ]
    return semantic


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("observedAt must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("observedAt must contain a timezone")
    return parsed.astimezone(timezone.utc)


def report_patch_required(
    existing_encoded: Optional[str],
    new_report: dict,
    *,
    now: Optional[datetime] = None,
    refresh_interval_s: int = REPORT_REFRESH_INTERVAL_S,
) -> bool:
    """Suppress semantic no-ops until the bounded freshness refresh is due."""
    if not existing_encoded:
        return True
    try:
        existing = json.loads(existing_encoded)
        if not isinstance(existing, dict):
            return True
        if _semantic_report(existing) != _semantic_report(new_report):
            return True
        observed = [
            _parse_timestamp(gpu.get("observedAt")) for gpu in existing.get("gpus", [])
        ]
        if not observed:
            return True
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return True
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    if any(value > current for value in observed):
        return True
    oldest = min(observed)
    return (current - oldest).total_seconds() >= refresh_interval_s


class PodReportPatcher:
    """Own exactly one annotation and retry optimistic Pod patch conflicts."""

    def __init__(self, core_v1, conflict_retries: int = PATCH_CONFLICT_RETRIES) -> None:
        if conflict_retries < 1:
            raise ValueError("conflict_retries must be positive")
        self._core_v1 = core_v1
        self._conflict_retries = conflict_retries

    def publish(self, pod, report: dict, *, now: Optional[datetime] = None):
        encoded = encode_report(report)
        current = pod
        original_uid = getattr(pod.metadata, "uid", None)
        for attempt in range(self._conflict_retries):
            metadata = current.metadata
            if getattr(metadata, "uid", None) != original_uid:
                raise RuntimeError("Pod UID changed during report conflict retry")
            annotations = metadata.annotations or {}
            if not report_patch_required(
                annotations.get(REPORT_ANNOTATION_KEY), report, now=now
            ):
                return current, False
            body = {
                "metadata": {
                    "resourceVersion": metadata.resource_version,
                    "annotations": {REPORT_ANNOTATION_KEY: encoded},
                }
            }
            try:
                updated = self._core_v1.patch_namespaced_pod(
                    name=metadata.name,
                    namespace=metadata.namespace,
                    body=body,
                    _request_timeout=PATCH_REQUEST_TIMEOUT_S,
                )
                return updated, True
            except Exception as exc:
                if (
                    getattr(exc, "status", None) != 409
                    or attempt + 1 >= self._conflict_retries
                ):
                    raise
                current = self._core_v1.read_namespaced_pod(
                    name=metadata.name,
                    namespace=metadata.namespace,
                    _request_timeout=PATCH_REQUEST_TIMEOUT_S,
                )
        raise AssertionError("unreachable")
