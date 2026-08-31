# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report models used by the Dynamo replay API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_primitive(asdict(value))
    if isinstance(value, Enum):
        return _to_primitive(value.value)
    if isinstance(value, dict):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(item) for item in value]
    return value


@dataclass
class PlannerReplayDetails:
    metadata: dict[str, Any] = field(default_factory=dict)
    ticks: list[dict[str, Any]] = field(default_factory=list)
    scaling_events: list[Any] = field(default_factory=list)
    lifecycle_operations: list[dict[str, Any]] = field(default_factory=list)
    total_ticks: int = 0
    html_report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": _to_primitive(self.metadata),
            "ticks": _to_primitive(self.ticks),
            "scaling_events": _to_primitive(self.scaling_events),
            "lifecycle_operations": _to_primitive(self.lifecycle_operations),
            "total_ticks": self.total_ticks,
            "html_report_path": self.html_report_path,
        }


@dataclass
class ReplayTelemetryDetails:
    """Periodic observability samples captured independently of Planner ticks."""

    sample_interval_ms: float
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_interval_ms": self.sample_interval_ms,
            "samples": _to_primitive(self.samples),
        }


@dataclass
class ReplayReport:
    summary: dict[str, Any]
    per_request: list[dict[str, Any]] | None
    coverage: dict[str, Any]
    planner: PlannerReplayDetails | None
    telemetry: ReplayTelemetryDetails | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "summary": self.summary,
            "per_request": self.per_request,
            "coverage": self.coverage,
            "planner": None if self.planner is None else self.planner.to_dict(),
        }
        # Preserve the exact default-disabled serialized report contract.
        if self.telemetry is not None:
            payload["telemetry"] = self.telemetry.to_dict()
        return payload
