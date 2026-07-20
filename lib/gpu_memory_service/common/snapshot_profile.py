# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in structured timing for GMS snapshot experiments.

Set ``GMS_SNAPSHOT_PROFILE=1`` to emit compact ``GMS_SNAPSHOT_PROFILE`` JSON
records at INFO level. Disabled profiles do not read clocks or acquire locks.
Wall-clock endpoints align records across processes; durations use monotonic
clocks and aggregate CPU durations use per-thread CPU clocks.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from uuid import uuid4

SNAPSHOT_PROFILE_ENV = "GMS_SNAPSHOT_PROFILE"
_EVENT_NAME = "gms_snapshot_profile"
_SCALAR_TYPES = (str, int, float, bool, type(None))
_MANAGED_FIELDS = {
    "bytes",
    "component",
    "count",
    "cpu_duration_ns",
    "duration_ns",
    "event",
    "phase",
    "wall_end_ns",
    "wall_start_ns",
}


def snapshot_profile_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether detailed snapshot profiling is explicitly enabled."""
    source = os.environ if environ is None else environ
    return source.get(SNAPSHOT_PROFILE_ENV) == "1"


@dataclass
class _Aggregate:
    wall_start_ns: int
    wall_end_ns: int
    duration_ns: int = 0
    cpu_duration_ns: int = 0
    count: int = 0
    bytes: int = 0


class _DisabledSpan(AbstractContextManager):
    def __enter__(self) -> "_DisabledSpan":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


_DISABLED_SPAN = _DisabledSpan()


class _ProfileSpan(AbstractContextManager):
    def __init__(
        self,
        profile: "SnapshotProfile",
        phase: str,
        fields: Mapping[str, Any],
        *,
        aggregate: bool,
        count: int,
        byte_count: int,
    ) -> None:
        self._profile = profile
        self._phase = phase
        self._fields = profile.validate_fields(fields)
        self._aggregate = aggregate
        self._count = count
        self._byte_count = byte_count
        self._wall_start_ns = time.time_ns()
        self._monotonic_start_ns = time.monotonic_ns()
        self._cpu_start_ns = time.thread_time_ns()

    def __enter__(self) -> "_ProfileSpan":
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        wall_end_ns = time.time_ns()
        duration_ns = time.monotonic_ns() - self._monotonic_start_ns
        cpu_duration_ns = time.thread_time_ns() - self._cpu_start_ns
        fields = dict(self._fields)
        if exc_type is not None:
            fields["status"] = "error"
            fields["error_type"] = exc_type.__name__
        if self._aggregate:
            self._profile.add_aggregate(
                self._phase,
                wall_start_ns=self._wall_start_ns,
                wall_end_ns=wall_end_ns,
                duration_ns=duration_ns,
                cpu_duration_ns=cpu_duration_ns,
                count=self._count,
                byte_count=self._byte_count,
                **fields,
            )
        else:
            self._profile.emit(
                self._phase,
                wall_start_ns=self._wall_start_ns,
                wall_end_ns=wall_end_ns,
                duration_ns=duration_ns,
                cpu_duration_ns=cpu_duration_ns,
                count=self._count,
                bytes=self._byte_count,
                kind="phase",
                **fields,
            )


class SnapshotProfile:
    """Thread-safe phase and aggregate recorder for one snapshot component."""

    def __init__(
        self,
        component: str,
        *,
        logger: Optional[logging.Logger] = None,
        enabled: Optional[bool] = None,
        profile_session_id: Optional[str] = None,
        **identity: Any,
    ) -> None:
        self.component = component
        self.enabled = snapshot_profile_enabled() if enabled is None else bool(enabled)
        self._logger = logger or logging.getLogger(__name__)
        self.profile_session_id = profile_session_id if self.enabled else None
        if self.profile_session_id is not None:
            identity["session"] = self.profile_session_id
        self._identity = self.validate_fields(identity) if self.enabled else {}
        self._aggregates: dict[tuple[str, tuple[tuple[str, Any], ...]], _Aggregate] = {}
        self._lock = threading.Lock() if self.enabled else None

    def ensure_profile_session_id(self) -> Optional[str]:
        """Return the client-generated correlation ID, creating it if needed."""
        if not self.enabled:
            return None
        assert self._lock is not None
        with self._lock:
            if self.profile_session_id is None:
                self.profile_session_id = str(uuid4())
                self._identity["session"] = self.profile_session_id
            return self.profile_session_id

    @staticmethod
    def validate_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
        """Return JSON-safe scalar fields suitable for aggregate keys."""
        validated = dict(fields)
        for name, value in validated.items():
            if not isinstance(name, str):
                raise TypeError("snapshot profile field names must be strings")
            if name in _MANAGED_FIELDS:
                raise ValueError(
                    f"snapshot profile field {name!r} is managed by the recorder"
                )
            if not isinstance(value, _SCALAR_TYPES):
                raise TypeError(
                    f"snapshot profile field {name!r} must be a JSON scalar, "
                    f"got {type(value).__name__}"
                )
        return validated

    def phase(
        self,
        phase: str,
        *,
        count: int = 0,
        byte_count: int = 0,
        **fields: Any,
    ) -> AbstractContextManager:
        """Time and emit one bounded phase."""
        if not self.enabled:
            return _DISABLED_SPAN
        return _ProfileSpan(
            self,
            phase,
            fields,
            aggregate=False,
            count=count,
            byte_count=byte_count,
        )

    def aggregate(
        self,
        phase: str,
        *,
        count: int = 1,
        byte_count: int = 0,
        **fields: Any,
    ) -> AbstractContextManager:
        """Measure one operation into a later aggregate summary."""
        if not self.enabled:
            return _DISABLED_SPAN
        return _ProfileSpan(
            self,
            phase,
            fields,
            aggregate=True,
            count=count,
            byte_count=byte_count,
        )

    def add_aggregate(
        self,
        phase: str,
        *,
        wall_start_ns: int,
        wall_end_ns: int,
        duration_ns: int,
        cpu_duration_ns: int = 0,
        count: int = 1,
        byte_count: int = 0,
        **fields: Any,
    ) -> None:
        """Add an externally measured duration, such as CUDA-event time."""
        if not self.enabled:
            return
        key_fields = tuple(sorted(self.validate_fields(fields).items()))
        key = (phase, key_fields)
        assert self._lock is not None
        with self._lock:
            aggregate = self._aggregates.get(key)
            if aggregate is None:
                aggregate = _Aggregate(
                    wall_start_ns=wall_start_ns,
                    wall_end_ns=wall_end_ns,
                )
                self._aggregates[key] = aggregate
            else:
                aggregate.wall_start_ns = min(aggregate.wall_start_ns, wall_start_ns)
                aggregate.wall_end_ns = max(aggregate.wall_end_ns, wall_end_ns)
            aggregate.duration_ns += int(duration_ns)
            aggregate.cpu_duration_ns += int(cpu_duration_ns)
            aggregate.count += int(count)
            aggregate.bytes += int(byte_count)

    def emit(
        self,
        phase: str,
        *,
        wall_start_ns: int,
        wall_end_ns: int,
        duration_ns: int,
        cpu_duration_ns: int = 0,
        count: int = 0,
        bytes: int = 0,
        **fields: Any,
    ) -> None:
        """Emit one machine-parseable timing event."""
        if not self.enabled:
            return
        fields = self.validate_fields(fields)
        record = {
            "event": _EVENT_NAME,
            "component": self.component,
            "phase": phase,
            **self._identity,
            **fields,
            "wall_start_ns": int(wall_start_ns),
            "wall_end_ns": int(wall_end_ns),
            "duration_ns": int(duration_ns),
            "cpu_duration_ns": int(cpu_duration_ns),
            "count": int(count),
            "bytes": int(bytes),
        }
        self._logger.info(
            "GMS_SNAPSHOT_PROFILE %s",
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )

    def emit_aggregates(self, **matching_fields: Any) -> None:
        """Emit and clear aggregate records matching the supplied identity."""
        if not self.enabled:
            return
        assert self._lock is not None
        selected: list[tuple[str, tuple[tuple[str, Any], ...], _Aggregate]] = []
        with self._lock:
            for key, aggregate in list(self._aggregates.items()):
                phase, key_fields = key
                fields = dict(key_fields)
                if any(
                    fields.get(name) != value for name, value in matching_fields.items()
                ):
                    continue
                selected.append((phase, key_fields, aggregate))
                del self._aggregates[key]
        for phase, key_fields, aggregate in sorted(
            selected, key=lambda item: (item[0], repr(item[1]))
        ):
            self.emit(
                phase,
                wall_start_ns=aggregate.wall_start_ns,
                wall_end_ns=aggregate.wall_end_ns,
                duration_ns=aggregate.duration_ns,
                cpu_duration_ns=aggregate.cpu_duration_ns,
                count=aggregate.count,
                bytes=aggregate.bytes,
                kind="aggregate",
                duration_semantics="cumulative",
                **dict(key_fields),
            )
