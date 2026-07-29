# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
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


def _validate_finite(value: Any, path: str = "") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"canonical replay rejects non-finite number at {path or '/'}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}/{index}")


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
class ReplayReport:
    summary: dict[str, Any]
    per_request: list[dict[str, Any]] | None
    coverage: dict[str, Any]
    planner: PlannerReplayDetails | None
    _native: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": _to_primitive(self.summary),
            "per_request": _to_primitive(self.per_request),
            "coverage": _to_primitive(self.coverage),
            "planner": None if self.planner is None else self.planner.to_dict(),
        }

    def to_canonical_dict(self) -> dict[str, Any]:
        if self._native is None:
            raise ValueError(
                "canonical serialization requires a replay run with canonical_capture=True"
            )
        planner = None if self.planner is None else self.planner.to_dict()
        _validate_finite(planner, "/planner")
        return self._native.canonical_dict(planner)

    def _canonical_json_line(self) -> bytes:
        if self._native is None:
            raise ValueError(
                "canonical serialization requires a replay run with canonical_capture=True"
            )
        planner = None if self.planner is None else self.planner.to_dict()
        _validate_finite(planner, "/planner")
        return bytes(self._native.canonical_json_line(planner))
