# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-run driver for the Autoscaling Arena.

Builds the DynoSim substrate (mocker engine + AIS perf model + router) and
drives one autoscaler-under-test through Dynamo's unified Rust replay loop.
The substrate construction deliberately reuses Dynamo's engine-argument and
capability helpers so *every* autoscaler runs on identical hardware, model,
router, and cold-start modeling. Only the decision-maker varies.

Rival adapters carry no FPM regression, so the Planner's AIS
regression-bootstrap is skipped for them; the mocker still simulates request
latencies from the AIS perf model in the engine args.

This module runs ONE (autoscaler × config × workload) match. The leaderboard
sweep and scorecard live one layer up (Phases 3–4).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.engine_protocol import EngineProtocol
from dynamo.planner.core.types import TrafficObservation, WorkerCapabilities
from dynamo.planner.offline.replay_adapter import ReplayPlannerAdapter
from dynamo.planner.offline.trace_data import extract_traffic_observations_from_trace
from dynamo.replay import planner as _replay_planner
from dynamo.replay.config import load_engine_args as _load_engine_args
from dynamo.replay.report import PlannerReplayDetails, ReplayReport
from dynamo.replay.reporting import write_report_json

if TYPE_CHECKING:
    from dynamo.mocker import MockEngineArgs

EngineFactory = Callable[[PlannerConfig, WorkerCapabilities], EngineProtocol]

TELEMETRY_CONTRACT = "dynamo.replay.telemetry.v1"

logger = logging.getLogger(__name__)

# Arena must construct arbitrary autoscaler engines before wrapping them in the
# common replay adapter, so it cannot call Dynamo's builtin adapter factory as a
# whole. Keep the substrate-sensitive derivations bound to Dynamo's current
# Planner preparation implementation instead of copying them here.
_engine_caps = _replay_planner._engine_caps
_generate_ais_decode_fpms = _replay_planner._generate_ais_decode_fpms
_generate_ais_prefill_fpms = _replay_planner._generate_ais_prefill_fpms


@dataclass(frozen=True)
class ArenaReplayResult:
    """Arena timeline composed with Dynamo's canonical offline report."""

    replay_report: ReplayReport
    timeline: list[dict[str, Any]]
    telemetry_artifact: Optional[dict[str, Any]] = None

    @property
    def trace_report(self) -> dict[str, Any]:
        """Backward-compatible view of the canonical replay summary."""

        return self.replay_report.summary

    @property
    def per_request(self) -> Optional[list[dict[str, Any]]]:
        return self.replay_report.per_request

    @property
    def coverage(self) -> dict[str, Any]:
        return self.replay_report.coverage

    @property
    def planner(self) -> Optional[PlannerReplayDetails]:
        return self.replay_report.planner

    @property
    def scaling_events(self) -> list[Any]:
        return self.planner.scaling_events if self.planner is not None else []

    @property
    def total_ticks(self) -> int:
        return self.planner.total_ticks if self.planner is not None else 0

    @property
    def html_report_path(self) -> Optional[str]:
        return self.planner.html_report_path if self.planner is not None else None

    @property
    def gpu_hours(self) -> float:
        """Exact provisioned GPU-hours integrated by the Rust runtime."""

        return float(self.trace_report.get("gpu_hours", 0.0) or 0.0)


class _TelemetryReductions:
    """Pure reductions for replay-owned scheduler telemetry."""

    @staticmethod
    def _scheduler_rows(value: Any) -> Optional[list[dict[str, Any]]]:
        """Copy raw scheduler rows while preserving payload availability.

        ``None`` means that the telemetry payload omitted scheduler metrics.
        An empty list is a valid observation for a pool with no reported
        ranks. Keeping that distinction prevents absent data from silently
        turning into measured zeroes when rendered later.
        """

        if not isinstance(value, list):
            return None
        return [dict(row) for row in value if isinstance(row, Mapping)]

    @staticmethod
    def _active_scheduler_rows(
        rows: Optional[list[dict[str, Any]]], active_worker_ids: Any
    ) -> Optional[list[dict[str, Any]]]:
        """Return rank rows belonging to every currently active worker.

        Starting and draining workers can still appear in the raw telemetry
        rows, but their cache capacity is not available to the autoscaler's
        pre-decision fleet view.  Require at least one row for every active
        worker so a partial sample cannot under-report queue or KV pressure.
        """

        if rows is None or not isinstance(active_worker_ids, list):
            return None
        active = {str(worker_id) for worker_id in active_worker_ids}
        selected = [
            row
            for row in rows
            if row.get("worker_id") is not None and str(row["worker_id"]) in active
        ]
        covered = {
            str(row["worker_id"])
            for row in selected
            if row.get("worker_id") is not None
        }
        return selected if active.issubset(covered) else None

    @staticmethod
    def _non_negative_int(value: Any) -> Optional[int]:
        if isinstance(value, bool) or value is None:
            return None
        try:
            rendered = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return rendered if rendered >= 0 else None

    @classmethod
    def _sum_row_field(
        cls, rows: Optional[list[dict[str, Any]]], field: str
    ) -> Optional[int]:
        if rows is None:
            return None
        values = [cls._non_negative_int(row.get(field)) for row in rows]
        return (
            sum(value for value in values if value is not None)
            if all(value is not None for value in values)
            else None
        )

    @classmethod
    def _scheduler_aggregate(
        cls, rows: Optional[list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Reduce active rank rows with capacity/token-weighted ratios."""

        active_blocks = cls._sum_row_field(rows, "active_blocks")
        inactive_blocks = cls._sum_row_field(rows, "inactive_blocks")
        total_blocks = cls._sum_row_field(rows, "total_blocks")
        running_requests = cls._sum_row_field(rows, "running_requests")
        waiting_requests = cls._sum_row_field(rows, "waiting_requests")

        valid_active_capacity = (
            active_blocks is not None
            and total_blocks is not None
            and total_blocks > 0
            and active_blocks <= total_blocks
        )
        valid_physical_capacity = (
            valid_active_capacity
            and inactive_blocks is not None
            and active_blocks + inactive_blocks <= total_blocks
        )
        return {
            "active_blocks": active_blocks,
            "inactive_blocks": inactive_blocks,
            "total_blocks": total_blocks,
            "active_kv_cache_utilization": (
                active_blocks / total_blocks if valid_active_capacity else None
            ),
            "physical_kv_cache_utilization": (
                (active_blocks + inactive_blocks) / total_blocks
                if valid_physical_capacity
                else None
            ),
            "running_requests": running_requests,
            "waiting_requests": waiting_requests,
        }

    @staticmethod
    def _combine_scheduler_aggregates(*aggregates: Mapping[str, Any]) -> dict[str, Any]:
        """Combine pool totals without averaging already-computed ratios."""

        def total(field: str) -> Optional[int]:
            values = [aggregate.get(field) for aggregate in aggregates]
            return (
                sum(int(value) for value in values)
                if all(value is not None for value in values)
                else None
            )

        active_blocks = total("active_blocks")
        inactive_blocks = total("inactive_blocks")
        total_blocks = total("total_blocks")
        active_ratio = (
            active_blocks / total_blocks
            if active_blocks is not None
            and total_blocks is not None
            and total_blocks > 0
            and active_blocks <= total_blocks
            else None
        )
        physical_ratio = (
            (active_blocks + inactive_blocks) / total_blocks
            if active_blocks is not None
            and inactive_blocks is not None
            and total_blocks is not None
            and total_blocks > 0
            and active_blocks + inactive_blocks <= total_blocks
            else None
        )
        return {
            "active_blocks": active_blocks,
            "inactive_blocks": inactive_blocks,
            "total_blocks": total_blocks,
            "active_kv_cache_utilization": active_ratio,
            "physical_kv_cache_utilization": physical_ratio,
            "running_requests": total("running_requests"),
            "waiting_requests": total("waiting_requests"),
        }


class _ReplayTelemetryTimeline:
    """Project top-level replay telemetry into the Arena chart schema.

    The samples are simulation-owned observations. Planner details are only
    used for exact controller-decision markers; scheduler telemetry never
    enters or leaves through the Planner callback.
    """

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _sequence(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    @staticmethod
    def _ids(value: Any) -> Optional[list[Any]]:
        if not isinstance(value, (list, tuple)):
            return None
        return list(value)

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        if isinstance(value, bool) or value is None:
            return None
        try:
            rendered = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return rendered if math.isfinite(rendered) else None

    @staticmethod
    def _ratio(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
        if (
            numerator is None
            or denominator is None
            or denominator <= 0
            or numerator > denominator
        ):
            return None
        return numerator / denominator

    @classmethod
    def _interval_metrics(cls, sample: Any, role: str) -> dict[str, Optional[int]]:
        interval = cls._mapping(cls._field(sample, f"{role}_interval_metrics"))
        to_int = _TelemetryReductions._non_negative_int
        return {
            "cache_hit_tokens": to_int(interval.get("cache_hit_tokens")),
            "cache_total_tokens": to_int(interval.get("cache_total_tokens")),
            "preemptions": to_int(interval.get("preemptions")),
        }

    @classmethod
    def _normalize_sample(cls, raw: Any) -> Optional[dict[str, Any]]:
        sampled_at_ms = cls._float(cls._field(raw, "sampled_at_ms"))
        if sampled_at_ms is None or sampled_at_ms < 0:
            return None
        interval_start_ms = cls._float(cls._field(raw, "interval_start_ms"))
        if interval_start_ms is None or interval_start_ms < 0:
            interval_start_ms = sampled_at_ms
        interval_start_ms = min(interval_start_ms, sampled_at_ms)
        traffic = cls._mapping(cls._field(raw, "traffic"))
        to_int = _TelemetryReductions._non_negative_int

        prefill_rows = _TelemetryReductions._scheduler_rows(
            cls._field(raw, "prefill_scheduler_metrics")
        )
        decode_rows = _TelemetryReductions._scheduler_rows(
            cls._field(raw, "decode_scheduler_metrics")
        )
        active_prefill_ids = cls._ids(cls._field(raw, "active_prefill_ids"))
        active_decode_ids = cls._ids(cls._field(raw, "active_decode_ids"))
        starting_prefill_ids = cls._ids(cls._field(raw, "starting_prefill_ids"))
        starting_decode_ids = cls._ids(cls._field(raw, "starting_decode_ids"))
        draining_prefill_ids = cls._ids(cls._field(raw, "draining_prefill_ids"))
        draining_decode_ids = cls._ids(cls._field(raw, "draining_decode_ids"))
        topology_available = all(
            ids is not None
            for ids in (
                active_prefill_ids,
                active_decode_ids,
                starting_prefill_ids,
                starting_decode_ids,
                draining_prefill_ids,
                draining_decode_ids,
            )
        )
        active_prefill_ids = active_prefill_ids or []
        active_decode_ids = active_decode_ids or []
        starting_prefill_ids = starting_prefill_ids or []
        starting_decode_ids = starting_decode_ids or []
        draining_prefill_ids = draining_prefill_ids or []
        draining_decode_ids = draining_decode_ids or []

        active_prefill_rows = _TelemetryReductions._active_scheduler_rows(
            prefill_rows, active_prefill_ids if topology_available else None
        )
        active_decode_rows = _TelemetryReductions._active_scheduler_rows(
            decode_rows, active_decode_ids if topology_available else None
        )
        prefill_active = _TelemetryReductions._scheduler_aggregate(active_prefill_rows)
        decode_active = _TelemetryReductions._scheduler_aggregate(active_decode_rows)
        fleet_active = _TelemetryReductions._combine_scheduler_aggregates(
            prefill_active, decode_active
        )
        scheduler_payload_available = (
            prefill_rows is not None and decode_rows is not None
        )
        scheduler_metrics_available = (
            scheduler_payload_available
            and active_prefill_rows is not None
            and active_decode_rows is not None
        )
        prefill_live = _TelemetryReductions._scheduler_aggregate(
            prefill_rows if scheduler_metrics_available else None
        )
        decode_live = _TelemetryReductions._scheduler_aggregate(
            decode_rows if scheduler_metrics_available else None
        )
        fleet_live = _TelemetryReductions._combine_scheduler_aggregates(
            prefill_live, decode_live
        )

        scheduler_waiting_prefill = prefill_live["waiting_requests"]
        scheduler_waiting_decode = decode_live["waiting_requests"]
        router_pending_prefill = to_int(
            cls._field(raw, "router_pending_prefill_requests")
        )
        router_pending_decode = to_int(
            cls._field(raw, "router_pending_decode_requests")
        )
        queue_available = (
            scheduler_metrics_available
            and scheduler_waiting_prefill is not None
            and scheduler_waiting_decode is not None
            and router_pending_prefill is not None
            and router_pending_decode is not None
        )
        queued_prefill = (
            scheduler_waiting_prefill + router_pending_prefill
            if queue_available
            else None
        )
        queued_decode = (
            scheduler_waiting_decode + router_pending_decode
            if queue_available
            else None
        )

        prefill_interval = cls._interval_metrics(raw, "prefill")
        decode_interval = cls._interval_metrics(raw, "decode")

        def role_sum(field: str) -> Optional[int]:
            left = prefill_interval[field]
            right = decode_interval[field]
            return left + right if left is not None and right is not None else None

        cache_hit_tokens = role_sum("cache_hit_tokens")
        cache_total_tokens = role_sum("cache_total_tokens")
        preemptions = role_sum("preemptions")
        arriving = to_int(traffic.get("arriving_requests"))
        completed = to_int(traffic.get("completed_requests"))
        ttft_count = to_int(traffic.get("ttft_count"))
        itl_count = to_int(traffic.get("itl_count"))
        router_hit_count = to_int(traffic.get("router_kv_hit_rate_count"))
        accept_length_forward_count = to_int(traffic.get("accept_length_forward_count"))
        kind = cls._field(raw, "kind", "periodic")
        kind = getattr(kind, "value", kind)
        kind_name = str(kind).lower()
        provisioned_prefill = (
            len(active_prefill_ids)
            + len(starting_prefill_ids)
            + len(draining_prefill_ids)
        )
        provisioned_decode = (
            len(active_decode_ids) + len(starting_decode_ids) + len(draining_decode_ids)
        )
        return {
            "sample_ordinal": to_int(cls._field(raw, "sample_ordinal")),
            "sample_kind": kind_name,
            "is_final": kind_name == "final",
            "timestamp_s": sampled_at_ms / 1000.0,
            "window_start_s": interval_start_ms / 1000.0,
            # Keep the simulator-owned traffic payload so future charts can be
            # derived from saved match results without rerunning the trace.
            "traffic": dict(traffic),
            "interval_duration_s": cls._float(traffic.get("duration_s")),
            "arriving_requests": arriving,
            "completed_requests": completed,
            "mean_isl": cls._float(traffic.get("avg_isl")),
            "mean_osl": cls._float(traffic.get("avg_osl")),
            "ttft_sample_count": ttft_count,
            "itl_sample_count": itl_count,
            "mean_ttft_ms": (
                cls._float(traffic.get("avg_ttft_ms"))
                if ttft_count is not None and ttft_count > 0
                else None
            ),
            "mean_tpot_ms": (
                cls._float(traffic.get("avg_itl_ms"))
                if itl_count is not None and itl_count > 0
                else None
            ),
            "mean_router_kv_hit_rate": (
                cls._float(traffic.get("avg_router_kv_hit_rate"))
                if router_hit_count is not None and router_hit_count > 0
                else None
            ),
            "router_kv_hit_sample_count": router_hit_count,
            "mean_accept_length": (
                cls._float(traffic.get("avg_accept_length"))
                if accept_length_forward_count is not None
                and accept_length_forward_count > 0
                else None
            ),
            "accept_length_forward_count": accept_length_forward_count,
            "prefill_scheduler_metrics": prefill_rows,
            "decode_scheduler_metrics": decode_rows,
            "scheduler_metrics_payload_available": scheduler_payload_available,
            "scheduler_metrics_available": scheduler_metrics_available,
            "active_scheduler_metrics_available": scheduler_metrics_available,
            "active_kv_blocks": fleet_active["active_blocks"],
            "inactive_kv_blocks": fleet_active["inactive_blocks"],
            "total_kv_blocks": fleet_active["total_blocks"],
            "active_kv_cache_utilization": fleet_active["active_kv_cache_utilization"],
            "physical_kv_cache_utilization": fleet_active[
                "physical_kv_cache_utilization"
            ],
            "prefill_active_kv_cache_utilization": prefill_active[
                "active_kv_cache_utilization"
            ],
            "decode_active_kv_cache_utilization": decode_active[
                "active_kv_cache_utilization"
            ],
            "prefill_physical_kv_cache_utilization": prefill_active[
                "physical_kv_cache_utilization"
            ],
            "decode_physical_kv_cache_utilization": decode_active[
                "physical_kv_cache_utilization"
            ],
            "scheduler_cache_hit_tokens": cache_hit_tokens,
            "scheduler_cache_total_tokens": cache_total_tokens,
            "scheduler_cache_reuse": cls._ratio(cache_hit_tokens, cache_total_tokens),
            "prefill_scheduler_cache_reuse": cls._ratio(
                prefill_interval["cache_hit_tokens"],
                prefill_interval["cache_total_tokens"],
            ),
            "decode_scheduler_cache_reuse": cls._ratio(
                decode_interval["cache_hit_tokens"],
                decode_interval["cache_total_tokens"],
            ),
            "running_prefill_requests": prefill_live["running_requests"],
            "running_decode_requests": decode_live["running_requests"],
            "total_running_requests": fleet_live["running_requests"],
            "prefill_preemptions": prefill_interval["preemptions"],
            "decode_preemptions": decode_interval["preemptions"],
            "preemptions": preemptions,
            "scheduler_waiting_prefill_requests": scheduler_waiting_prefill,
            "scheduler_waiting_decode_requests": scheduler_waiting_decode,
            "scheduler_waiting_requests": (
                scheduler_waiting_prefill + scheduler_waiting_decode
                if scheduler_waiting_prefill is not None
                and scheduler_waiting_decode is not None
                else None
            ),
            "router_pending_prefill_requests": router_pending_prefill,
            "router_pending_decode_requests": router_pending_decode,
            "router_pending_requests": (
                router_pending_prefill + router_pending_decode
                if router_pending_prefill is not None
                and router_pending_decode is not None
                else None
            ),
            "queue_telemetry_semantics": (
                "scheduler_waiting_plus_router_pending" if queue_available else None
            ),
            "queued_prefill_requests": queued_prefill,
            "queued_decode_requests": queued_decode,
            "total_queued_requests": (
                queued_prefill + queued_decode
                if queued_prefill is not None and queued_decode is not None
                else None
            ),
            "active_prefill_ids": active_prefill_ids,
            "active_decode_ids": active_decode_ids,
            "starting_prefill_ids": starting_prefill_ids,
            "starting_decode_ids": starting_decode_ids,
            "draining_prefill_ids": draining_prefill_ids,
            "draining_decode_ids": draining_decode_ids,
            "active_prefill_replicas": len(active_prefill_ids),
            "active_decode_replicas": len(active_decode_ids),
            "provisioned_prefill_replicas": provisioned_prefill,
            "provisioned_decode_replicas": provisioned_decode,
            "scaling_decision": False,
            "requested_prefill_replicas": None,
            "requested_decode_replicas": None,
        }

    @classmethod
    def _decisions(cls, planner: Any) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        for raw_tick in cls._sequence(cls._field(planner, "ticks")):
            tick = cls._mapping(raw_tick)
            runtime_decision = cls._mapping(tick.get("runtime_decision"))
            to_int = _TelemetryReductions._non_negative_int
            target_prefill = to_int(runtime_decision.get("target_prefill"))
            target_decode = to_int(runtime_decision.get("target_decode"))
            if target_prefill is None and target_decode is None:
                continue
            at_ms = cls._float(tick.get("at_ms"))
            if at_ms is None or at_ms < 0:
                continue
            topology = cls._mapping(tick.get("topology"))
            prefill = cls._mapping(topology.get("prefill"))
            decode = cls._mapping(topology.get("decode"))

            def ids(role: Mapping[str, Any], state: str) -> list[Any]:
                return cls._sequence(role.get(state))

            active_prefill = len(ids(prefill, "active"))
            active_decode = len(ids(decode, "active"))
            starting_prefill = len(ids(prefill, "starting"))
            starting_decode = len(ids(decode, "starting"))
            provisioned_prefill = (
                active_prefill + starting_prefill + len(ids(prefill, "draining"))
            )
            provisioned_decode = (
                active_decode + starting_decode + len(ids(decode, "draining"))
            )
            decisions.append(
                {
                    "timestamp_s": at_ms / 1000.0,
                    "window_start_s": at_ms / 1000.0,
                    "active_prefill_replicas": active_prefill,
                    "active_decode_replicas": active_decode,
                    "provisioned_prefill_replicas": provisioned_prefill,
                    "provisioned_decode_replicas": provisioned_decode,
                    "scaling_decision": True,
                    "requested_prefill_replicas": (
                        target_prefill
                        if target_prefill is not None
                        else active_prefill + starting_prefill
                    ),
                    "requested_decode_replicas": (
                        target_decode
                        if target_decode is not None
                        else active_decode + starting_decode
                    ),
                    "decision_only": True,
                }
            )
        return decisions

    @classmethod
    def build(cls, telemetry_samples: Any, planner: Any) -> list[dict[str, Any]]:
        samples = [
            sample
            for raw in cls._sequence(telemetry_samples)
            if (sample := cls._normalize_sample(raw)) is not None
        ]
        return cls.add_decisions(samples, planner)

    @classmethod
    def add_decisions(
        cls, samples: list[dict[str, Any]], planner: Any
    ) -> list[dict[str, Any]]:
        """Merge exact Planner decisions into already-projected samples."""

        if not samples:
            return []
        by_timestamp = {sample["timestamp_s"]: sample for sample in samples}
        for decision in cls._decisions(planner):
            existing = by_timestamp.get(decision["timestamp_s"])
            if existing is None:
                samples.append(decision)
                by_timestamp[decision["timestamp_s"]] = decision
            else:
                existing["scaling_decision"] = True
                existing["requested_prefill_replicas"] = decision[
                    "requested_prefill_replicas"
                ]
                existing["requested_decode_replicas"] = decision[
                    "requested_decode_replicas"
                ]
        samples.sort(key=lambda sample: sample["timestamp_s"])
        return samples


@dataclass(frozen=True)
class _TelemetryJsonlIngestion:
    timeline: list[dict[str, Any]]
    sample_count: int
    sha256: str


class _TelemetryContractValidator:
    """Validate the minimum forward-compatible telemetry v1 contract."""

    _top_level_keys = (
        "sample_ordinal",
        "kind",
        "interval_start_ms",
        "sampled_at_ms",
        "traffic",
        "prefill_scheduler_metrics",
        "decode_scheduler_metrics",
        "prefill_interval_metrics",
        "decode_interval_metrics",
        "router_pending_prefill_requests",
        "router_pending_decode_requests",
        "active_prefill_ids",
        "active_decode_ids",
        "starting_prefill_ids",
        "starting_decode_ids",
        "draining_prefill_ids",
        "draining_decode_ids",
    )
    _traffic_numbers = (
        "duration_s",
        "avg_isl",
        "avg_osl",
        "avg_ttft_ms",
        "avg_itl_ms",
        "avg_router_kv_hit_rate",
    )
    _traffic_integers = (
        "arriving_requests",
        "completed_requests",
        "ttft_count",
        "itl_count",
        "router_kv_hit_rate_count",
        "accept_length_forward_count",
    )
    _interval_keys = ("cache_hit_tokens", "cache_total_tokens", "preemptions")
    _scheduler_integers = (
        "worker_id",
        "dp_rank",
        "active_blocks",
        "inactive_blocks",
        "total_blocks",
        "running_requests",
        "waiting_requests",
    )
    _scheduler_numbers = ("active_cache_usage", "physical_cache_usage")
    _id_lists = (
        "active_prefill_ids",
        "active_decode_ids",
        "starting_prefill_ids",
        "starting_decode_ids",
        "draining_prefill_ids",
        "draining_decode_ids",
    )

    def __init__(self, path: Path) -> None:
        self._path = path
        self._expected_ordinal = 0
        self._previous_timestamp: Optional[float] = None
        self._saw_final = False

    def _fail(self, line_number: int, field: str, message: str) -> None:
        location = f"{self._path}:{line_number}"
        if field:
            location = f"{location} ({field})"
        raise ValueError(
            f"invalid {TELEMETRY_CONTRACT} telemetry JSONL at " f"{location}: {message}"
        )

    def _required(
        self,
        value: Mapping[str, Any],
        key: str,
        *,
        line_number: int,
        parent: str = "",
    ) -> Any:
        if key not in value:
            field = f"{parent}.{key}" if parent else key
            self._fail(line_number, field, "required field is missing")
        return value[key]

    def _mapping(
        self, value: Any, *, line_number: int, field: str
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            self._fail(line_number, field, "must be an object")
        return value

    def _integer(self, value: Any, *, line_number: int, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self._fail(line_number, field, "must be a non-negative integer")
        return value

    def _number(
        self,
        value: Any,
        *,
        line_number: int,
        field: str,
        maximum: Optional[float] = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._fail(line_number, field, "must be a finite non-negative number")
        rendered = float(value)
        if not math.isfinite(rendered) or rendered < 0:
            self._fail(line_number, field, "must be a finite non-negative number")
        if maximum is not None and rendered > maximum:
            self._fail(line_number, field, f"must be at most {maximum:g}")
        return rendered

    def _validate_traffic(self, value: Any, *, line_number: int) -> None:
        traffic = self._mapping(value, line_number=line_number, field="traffic")
        for key in self._traffic_numbers:
            maximum = 1.0 if key == "avg_router_kv_hit_rate" else None
            self._number(
                self._required(traffic, key, line_number=line_number, parent="traffic"),
                line_number=line_number,
                field=f"traffic.{key}",
                maximum=maximum,
            )
        for key in self._traffic_integers:
            self._integer(
                self._required(traffic, key, line_number=line_number, parent="traffic"),
                line_number=line_number,
                field=f"traffic.{key}",
            )
        avg_accept_length = self._required(
            traffic,
            "avg_accept_length",
            line_number=line_number,
            parent="traffic",
        )
        if avg_accept_length is not None:
            self._number(
                avg_accept_length,
                line_number=line_number,
                field="traffic.avg_accept_length",
            )

    def _validate_interval(self, value: Any, *, line_number: int, field: str) -> None:
        interval = self._mapping(value, line_number=line_number, field=field)
        for key in self._interval_keys:
            self._integer(
                self._required(interval, key, line_number=line_number, parent=field),
                line_number=line_number,
                field=f"{field}.{key}",
            )

    def _validate_scheduler_rows(
        self, value: Any, *, line_number: int, field: str
    ) -> None:
        if not isinstance(value, list):
            self._fail(line_number, field, "must be an array")
        for index, raw_row in enumerate(value):
            row_field = f"{field}[{index}]"
            row = self._mapping(raw_row, line_number=line_number, field=row_field)
            for key in self._scheduler_integers:
                self._integer(
                    self._required(row, key, line_number=line_number, parent=row_field),
                    line_number=line_number,
                    field=f"{row_field}.{key}",
                )
            for key in self._scheduler_numbers:
                self._number(
                    self._required(row, key, line_number=line_number, parent=row_field),
                    line_number=line_number,
                    field=f"{row_field}.{key}",
                    maximum=1.0,
                )

    def _validate_id_list(self, value: Any, *, line_number: int, field: str) -> None:
        if not isinstance(value, list):
            self._fail(line_number, field, "must be an array")
        for index, worker_id in enumerate(value):
            self._integer(
                worker_id,
                line_number=line_number,
                field=f"{field}[{index}]",
            )

    def validate(self, sample: Any, *, line_number: int) -> Mapping[str, Any]:
        if not isinstance(sample, Mapping):
            self._fail(line_number, "", "sample must be an object")
        if self._saw_final:
            self._fail(line_number, "kind", "final sample must be the last sample")
        for key in self._top_level_keys:
            self._required(sample, key, line_number=line_number)

        ordinal = self._integer(
            sample["sample_ordinal"],
            line_number=line_number,
            field="sample_ordinal",
        )
        if ordinal != self._expected_ordinal:
            self._fail(
                line_number,
                "sample_ordinal",
                f"expected contiguous ordinal {self._expected_ordinal}, got {ordinal}",
            )

        kind = sample["kind"]
        if not isinstance(kind, str) or kind not in {
            "baseline",
            "periodic",
            "final",
        }:
            self._fail(
                line_number,
                "kind",
                "must be one of baseline, periodic, or final",
            )
        if ordinal == 0 and kind != "baseline":
            self._fail(line_number, "kind", "the first sample must be baseline")
        if ordinal > 0 and kind == "baseline":
            self._fail(line_number, "kind", "baseline is only legal for ordinal 0")

        interval_start_ms = self._number(
            sample["interval_start_ms"],
            line_number=line_number,
            field="interval_start_ms",
        )
        sampled_at_ms = self._number(
            sample["sampled_at_ms"],
            line_number=line_number,
            field="sampled_at_ms",
        )
        if interval_start_ms > sampled_at_ms:
            self._fail(
                line_number,
                "interval_start_ms",
                "must be less than or equal to sampled_at_ms",
            )
        if (
            self._previous_timestamp is not None
            and sampled_at_ms < self._previous_timestamp
        ):
            self._fail(
                line_number,
                "sampled_at_ms",
                "must be monotonically non-decreasing",
            )

        self._validate_traffic(sample["traffic"], line_number=line_number)
        for field in (
            "prefill_interval_metrics",
            "decode_interval_metrics",
        ):
            self._validate_interval(sample[field], line_number=line_number, field=field)
        for field in (
            "prefill_scheduler_metrics",
            "decode_scheduler_metrics",
        ):
            self._validate_scheduler_rows(
                sample[field], line_number=line_number, field=field
            )
        for field in (
            "router_pending_prefill_requests",
            "router_pending_decode_requests",
        ):
            self._integer(sample[field], line_number=line_number, field=field)
        for field in self._id_lists:
            self._validate_id_list(sample[field], line_number=line_number, field=field)

        self._expected_ordinal += 1
        self._previous_timestamp = sampled_at_ms
        self._saw_final = kind == "final"
        return sample


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _ingest_telemetry_jsonl(path: Path, planner: Any) -> _TelemetryJsonlIngestion:
    """Validate and project Dynamo telemetry without retaining raw samples.

    JSONL is the integration contract between replay and Arena reporting. Each
    line must be one v1 snapshot object. Validation and timeline projection
    happen together, so memory contains the report rows but never a second list
    of the complete raw snapshot objects.
    """

    validator = _TelemetryContractValidator(path)
    digest = hashlib.sha256()
    timeline: list[dict[str, Any]] = []
    sample_count = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"invalid {TELEMETRY_CONTRACT} telemetry JSONL at "
                    f"{path}:{line_number}: line is not valid UTF-8"
                ) from exc
            if not line.strip():
                raise ValueError(
                    f"invalid {TELEMETRY_CONTRACT} telemetry JSONL at "
                    f"{path}:{line_number}: blank lines are not allowed"
                )
            try:
                sample = json.loads(
                    line, parse_constant=_reject_nonstandard_json_constant
                )
            except (json.JSONDecodeError, ValueError) as exc:
                message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise ValueError(
                    f"invalid {TELEMETRY_CONTRACT} telemetry JSONL at "
                    f"{path}:{line_number}: {message}"
                ) from exc
            validated = validator.validate(sample, line_number=line_number)
            projected = _ReplayTelemetryTimeline._normalize_sample(validated)
            if projected is None:  # guarded by contract validation above
                raise AssertionError("validated telemetry sample did not project")
            timeline.append(projected)
            sample_count += 1
    if sample_count == 0:
        raise ValueError(
            f"invalid {TELEMETRY_CONTRACT} telemetry JSONL at {path}: " "file is empty"
        )
    return _TelemetryJsonlIngestion(
        timeline=_ReplayTelemetryTimeline.add_decisions(timeline, planner),
        sample_count=sample_count,
        sha256=digest.hexdigest(),
    )


def _paths_refer_to_same_file(left: Path, right: Path) -> bool:
    """Compare existing hard links and unresolved output aliases safely."""

    try:
        if left.samefile(right):
            return True
    except OSError:
        pass
    return left.resolve(strict=False) == right.resolve(strict=False)


def _validate_telemetry_destination(
    telemetry_path: Path,
    *,
    trace_paths: Sequence[str],
    report_json: Optional[str],
) -> None:
    for trace_path in trace_paths:
        candidate = Path(trace_path).expanduser()
        if _paths_refer_to_same_file(telemetry_path, candidate):
            raise ValueError(
                "telemetry_jsonl_path must not refer to a replay trace input: "
                f"{candidate}"
            )
    if report_json is not None:
        report_path = Path(report_json).expanduser()
        if _paths_refer_to_same_file(telemetry_path, report_path):
            raise ValueError(
                "telemetry_jsonl_path and report_json must refer to different files"
            )


def _as_config(substrate_config: Union[PlannerConfig, str, dict]) -> PlannerConfig:
    if isinstance(substrate_config, PlannerConfig):
        return substrate_config
    if isinstance(substrate_config, dict):
        return PlannerConfig.from_config_arg(json.dumps(substrate_config))
    return PlannerConfig.from_config_arg(substrate_config)


def _normalize_engine_args_role(
    raw_args: Optional[str],
    *,
    expected: Literal["aggregated", "prefill", "decode"],
    argument_name: str,
) -> Optional[str]:
    """Make an Arena engine-argument slot's worker role explicit.

    Arena callers commonly reuse one role-neutral substrate JSON object for
    both disaggregated pools. Dynamo's unified replay validates the role on
    each ``MockEngineArgs``, so infer it from the slot unless the caller
    already supplied ``worker_type`` or the legacy role flags.
    """

    if raw_args is None:
        return None
    values = json.loads(raw_args)
    if not isinstance(values, dict):
        raise ValueError(f"{argument_name} must contain a JSON object")

    declared = values.get("worker_type")
    if declared is not None and declared != expected:
        raise ValueError(
            f"{argument_name}.worker_type must be {expected!r}, got {declared!r}"
        )
    if declared is None and "is_prefill" not in values and "is_decode" not in values:
        values["worker_type"] = expected
    perf_config = values.get("ais_perf_config")
    if perf_config is not None:
        if not isinstance(perf_config, dict):
            raise ValueError(f"{argument_name}.ais_perf_config must be a mapping")
        perf_role = perf_config.get("worker_type")
        if perf_role is not None and perf_role != expected:
            raise ValueError(
                f"{argument_name}.ais_perf_config.worker_type must be {expected!r}"
            )
        perf_config["worker_type"] = expected
    return json.dumps(values)


def _supports_planner_bootstrap(engine: EngineProtocol) -> bool:
    """Whether an Arena engine consumes Dynamo Planner bootstrap state."""

    return hasattr(engine, "install_regressions_from_fpms") and getattr(
        engine, "supports_ais_bootstrap", True
    )


def _planner_warmup_observations(
    config: PlannerConfig, engine: EngineProtocol
) -> Optional[list[TrafficObservation]]:
    """Load the builtin Planner's predictor history exactly as Dynamo does."""

    if not _supports_planner_bootstrap(engine):
        return None
    warmup_trace = getattr(config, "load_predictor_warmup_trace", None)
    if warmup_trace is None:
        return None
    return extract_traffic_observations_from_trace(
        warmup_trace,
        config.throughput_adjustment_interval_seconds,
    )


def _resolved_performance_model_metadata(
    metadata: Optional[dict[str, Any]],
    *,
    mode: str,
    extra_engine_args: Optional[MockEngineArgs],
    prefill_engine_args: Optional[MockEngineArgs],
    decode_engine_args: Optional[MockEngineArgs],
) -> Optional[dict[str, Any]]:
    """Backfill each AIS role's version from its own lowered engine arguments.

    Match Config permits an omitted AIS backend version. Dynamo resolves that
    version while lowering ``MockEngineArgs``; keeping the role-specific
    metadata and filling it here avoids the canonical no-metadata fallback,
    which intentionally chooses one reference identity for both roles.
    Caller-owned metadata is never mutated and an explicit version wins.

    Only ``backend_version`` is filled, and only when the metadata and engine
    argument name the same backend. Other incomplete third-party metadata
    retains Dynamo's normal validation/fallback semantics instead of silently
    acquiring fields from the engine arguments.
    """

    if metadata is None:
        return None
    resolved: dict[str, Any] = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in metadata.items()
    }
    if mode == "agg":
        role_args = [("aggregated", "agg", extra_engine_args)]
    else:
        role_args = [
            ("prefill", None, prefill_engine_args),
            ("decode", None, decode_engine_args),
        ]

    for primary, alias, engine_args in role_args:
        if engine_args is None:
            continue
        key = primary if primary in resolved else alias
        if key is None or key not in resolved:
            continue
        raw = resolved[key]
        if not isinstance(raw, Mapping) or raw.get("provider") != "ais":
            continue
        raw_config = raw.get("config")
        if not isinstance(raw_config, Mapping):
            continue
        config = dict(raw_config)
        canonical = engine_args.ais_perf_config or {}
        if (
            "backend_version" not in config
            and config.get("backend") == canonical.get("backend")
            and canonical.get("backend_version") is not None
        ):
            config["backend_version"] = canonical["backend_version"]
        resolved[key] = {**dict(raw), "config": config}
    return resolved


def _bootstrap_ais_regressions(
    adapter: ReplayPlannerAdapter,
    engine: EngineProtocol,
    config: PlannerConfig,
    *,
    extra_engine_args: Optional[MockEngineArgs],
    prefill_engine_args: Optional[MockEngineArgs],
    decode_engine_args: Optional[MockEngineArgs],
    performance_model_metadata: Optional[dict[str, Any]],
    benchmark_granularity: int,
) -> None:
    """Mirror Dynamo's current AIS bootstrap for an Arena-created Planner.

    Arena cannot call ``prepare_planner_replay`` wholesale because its common
    factory seam must also wrap rival autoscaler engines. Keep the selection,
    session sharing, failure states, FPM installation, and provenance aligned
    with that Dynamo helper while leaving rival engines untouched.
    """
    if not _supports_planner_bootstrap(engine):
        return

    adapter.set_bootstrap_metadata({"status": "not_required"})
    if config.optimization_target != "sla":
        return

    perf_configs = _replay_planner._ais_performance_model_configs(
        performance_model_metadata, config.mode
    )
    if perf_configs:
        ref_args = extra_engine_args or decode_engine_args or prefill_engine_args
    else:
        ref_args = (
            extra_engine_args
            or (
                decode_engine_args
                if decode_engine_args is not None
                and decode_engine_args.ais_perf_config is not None
                else None
            )
            or prefill_engine_args
            or decode_engine_args
        )
    if ref_args is None:
        raise ValueError("Planner AIS bootstrap requires engine arguments")
    p_args = (
        extra_engine_args if config.mode == "agg" else prefill_engine_args
    ) or ref_args
    d_args = (
        extra_engine_args if config.mode == "agg" else decode_engine_args
    ) or ref_args
    if perf_configs:
        prefill_session_kwargs = _replay_planner._ais_session_kwargs(
            perf_configs["prefill"], p_args
        )
        decode_session_kwargs = _replay_planner._ais_session_kwargs(
            perf_configs["decode"], d_args
        )
    else:
        prefill_session_kwargs = _replay_planner._ais_session_kwargs(None, p_args)
        decode_session_kwargs = _replay_planner._ais_session_kwargs(None, d_args)
    if prefill_session_kwargs is None or decode_session_kwargs is None:
        adapter.set_bootstrap_metadata(
            {
                "status": "not_configured_load_only",
                "benchmark_granularity": benchmark_granularity,
            }
        )
        logger.warning(
            "throughput-based scaling regression requires AIS perf model; "
            "falling back to load-based scaling only"
        )
        return

    try:
        prefill_session = _replay_planner.create_session(**prefill_session_kwargs)
        decode_session = (
            prefill_session
            if decode_session_kwargs == prefill_session_kwargs
            else _replay_planner.create_session(**decode_session_kwargs)
        )
    except (
        ImportError,
        RuntimeError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        logger.warning(
            "AIS session creation failed (%s); throughput regression will not "
            "be bootstrapped",
            exc,
        )
        adapter.set_bootstrap_metadata(
            {
                "status": "session_failed_load_only",
                "benchmark_granularity": benchmark_granularity,
            }
        )
        return

    try:
        prefill_fpms = _generate_ais_prefill_fpms(
            prefill_session, p_args, benchmark_granularity
        )
        decode_fpms = _generate_ais_decode_fpms(
            decode_session, d_args, benchmark_granularity
        )
    except (RuntimeError, ValueError, KeyError, ArithmeticError) as exc:
        logger.warning(
            "AIS benchmark generation failed (%s); throughput regression will "
            "not be bootstrapped",
            exc,
        )
        prefill_fpms, decode_fpms = [], []

    bootstrap_metadata = {
        "status": "installed",
        "benchmark_granularity": benchmark_granularity,
        "prefill_fpm_count": len(prefill_fpms),
        "decode_fpm_count": len(decode_fpms),
        "fpm_sha256": _replay_planner._ais_fpm_digest(prefill_fpms, decode_fpms),
    }
    if config.mode == "agg":
        agg_fpms = prefill_fpms + decode_fpms
        if agg_fpms:
            adapter.install_benchmark_fpms(agg_fpms=agg_fpms)
        else:
            bootstrap_metadata["status"] = "empty"
            logger.warning("AIS produced no agg benchmark FPMs")
    elif prefill_fpms and decode_fpms:
        adapter.install_benchmark_fpms(
            prefill_fpms=prefill_fpms,
            decode_fpms=decode_fpms,
        )
    else:
        bootstrap_metadata["status"] = "empty"
        logger.warning(
            "AIS produced empty benchmark FPMs (prefill=%d, decode=%d)",
            len(prefill_fpms),
            len(decode_fpms),
        )
    adapter.set_bootstrap_metadata(bootstrap_metadata)


def run_arena_replay(
    *,
    trace_file: Optional[str] = None,
    trace_files: Optional[Sequence[str]] = None,
    trace_format: Literal["mooncake", "dynamo"] = "mooncake",
    autoscaler: EngineFactory,
    substrate_config: Union[PlannerConfig, str, dict],
    prefill_engine_args: Optional[str] = None,
    decode_engine_args: Optional[str] = None,
    extra_engine_args: Optional[str] = None,
    num_prefill_workers: int = 1,
    num_decode_workers: int = 1,
    num_workers: int = 1,
    arrival_speedup_ratio: float = 1.0,
    trace_block_size: Optional[int] = 512,
    router_mode: str = "round_robin",
    router_config: Optional[Any] = None,
    replay_concurrency: Optional[int] = None,
    model_name: Optional[str] = None,
    sla_ttft_ms: Optional[float] = None,
    sla_itl_ms: Optional[float] = None,
    sla_e2e_ms: Optional[float] = None,
    capture_per_request: bool = True,
    telemetry_sample_interval_s: Optional[float] = 5.0,
    telemetry_jsonl_path: Optional[str] = None,
    ais_bootstrap: bool = True,
    performance_model_metadata: Optional[dict[str, Any]] = None,
    benchmark_granularity: int = 8,
    report_json: Optional[str] = None,
) -> ArenaReplayResult:
    """Drive one autoscaler through one DynoSim match.

    Args:
        trace_file: single replay trace. This preserves the original Mooncake
            caller interface and is mutually exclusive with ``trace_files``.
        trace_files: one or more replay trace shards. Multiple files are only
            supported for native Dynamo request traces.
        trace_format: ``"mooncake"`` for a single Mooncake JSONL trace or
            ``"dynamo"`` for one or more native Dynamo request-trace shards.
        autoscaler: factory ``(PlannerConfig, WorkerCapabilities) -> EngineProtocol``
            building the autoscaler-under-test (e.g. ``StaticAutoscaler``).
        substrate_config: deployment substrate descriptor (mode, per-engine GPU
            counts, SLA targets) — NOT autoscaler-algorithm config. A
            ``PlannerConfig``, JSON string, or dict.
        prefill_engine_args / decode_engine_args: disagg engine-arg JSON. The
            corresponding worker role is inferred when omitted.
        extra_engine_args: agg engine-arg JSON; the aggregated role is inferred
            when omitted.
        num_prefill_workers / num_decode_workers: disagg start counts.
        num_workers: agg start count.
        trace_block_size: expected KV-cache block size. ``None`` uses the block
            size embedded in native Dynamo traces (and Dynamo's default of 512
            for Mooncake traces).
        router_config: kv-router knobs (``KvRouterConfig``), or None for round_robin.
        replay_concurrency: closed-loop in-flight cap; None = open-loop (arrival
            timestamps).
        sla_ttft_ms / sla_itl_ms / sla_e2e_ms: **goodput** SLA passed to the
            collector so the trace_report carries ``goodput_*`` (SLA-gated
            output tok/s) for ANY autoscaler. Independent of the planner's own
            scaling SLA in ``substrate_config``.
        capture_per_request: retain terminal request records in the canonical
            replay report so the scorecard can evaluate multiple SLO profiles.
        telemetry_sample_interval_s: simulated-time cadence for persisted
            replay telemetry. ``None`` disables telemetry capture.
        telemetry_jsonl_path: optional durable JSONL destination for the replay
            telemetry stream. Match Config runs always provide one. Other
            callers use a temporary JSONL artifact when omitted.
        performance_model_metadata: optional runner-neutral AIS identities by
            engine role. The builtin Planner uses these before falling back to
            identities embedded in the engine arguments; rivals ignore them.
        benchmark_granularity: number of intervals used when generating the
            builtin Planner's AIS bootstrap samples.
        report_json: optional path to write the AIPerf trace report.

    Returns:
        An :class:`ArenaReplayResult` combining Dynamo's planner report with
        the Arena timeline and optional per-request records.
    """
    if trace_file is not None and trace_files is not None:
        raise ValueError("trace_file and trace_files are mutually exclusive")
    if trace_file is None and trace_files is None:
        raise ValueError("one of trace_file or trace_files is required")
    if trace_format not in ("mooncake", "dynamo"):
        raise ValueError(
            f"trace_format must be 'mooncake' or 'dynamo', got {trace_format!r}"
        )
    if telemetry_sample_interval_s is not None and (
        not math.isfinite(telemetry_sample_interval_s)
        or telemetry_sample_interval_s <= 0
    ):
        raise ValueError("telemetry_sample_interval_s must be positive and finite")
    if (
        isinstance(benchmark_granularity, bool)
        or not isinstance(benchmark_granularity, int)
        or benchmark_granularity <= 0
    ):
        raise ValueError("benchmark_granularity must be a positive integer")

    replay_trace_files = (
        [trace_file] if trace_file is not None else list(trace_files or ())
    )
    if not replay_trace_files:
        raise ValueError("trace_files must contain at least one trace file")
    if trace_format != "dynamo" and len(replay_trace_files) != 1:
        raise ValueError(
            f"trace_format={trace_format!r} requires exactly one trace file"
        )

    config = _as_config(substrate_config)
    # Rivals run in advisory mode like the planner replay (decisions are applied
    # by the harness, not by a live orchestrator).
    config.advisory = True

    p_args = _load_engine_args(
        _normalize_engine_args_role(
            prefill_engine_args,
            expected="prefill",
            argument_name="prefill_engine_args",
        )
    )
    d_args = _load_engine_args(
        _normalize_engine_args_role(
            decode_engine_args,
            expected="decode",
            argument_name="decode_engine_args",
        )
    )
    agg_args = _load_engine_args(
        _normalize_engine_args_role(
            extra_engine_args,
            expected="aggregated",
            argument_name="extra_engine_args",
        )
    )

    if config.mode == "disagg":
        if p_args is None or d_args is None:
            raise ValueError(
                "disagg arena replay requires prefill_engine_args and decode_engine_args"
            )
        capabilities = WorkerCapabilities(
            prefill=_engine_caps(p_args), decode=_engine_caps(d_args)
        )
    elif config.mode == "agg":
        if agg_args is None:
            agg_args = _load_engine_args("{}")
        capabilities = WorkerCapabilities(decode=_engine_caps(agg_args))
    else:
        raise ValueError(
            f"arena replay supports mode 'agg'/'disagg', got '{config.mode}'"
        )

    resolved_performance_model_metadata = _resolved_performance_model_metadata(
        performance_model_metadata,
        mode=config.mode,
        extra_engine_args=agg_args,
        prefill_engine_args=p_args,
        decode_engine_args=d_args,
    )

    telemetry_path = (
        Path(telemetry_jsonl_path).expanduser()
        if telemetry_sample_interval_s is not None and telemetry_jsonl_path is not None
        else None
    )
    if telemetry_path is not None:
        _validate_telemetry_destination(
            telemetry_path,
            trace_paths=replay_trace_files,
            report_json=report_json,
        )

    engine = autoscaler(config, capabilities)
    warmup_observations = _planner_warmup_observations(config, engine)
    adapter = ReplayPlannerAdapter(
        planner_config=config,
        engine=engine,
        capabilities=capabilities,
        warmup_observations=warmup_observations,
        benchmark_granularity=benchmark_granularity,
    )

    temporary_telemetry_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        with adapter:
            if telemetry_sample_interval_s is not None and telemetry_path is None:
                temporary_telemetry_dir = tempfile.TemporaryDirectory(
                    prefix="autoscaling-arena-replay-telemetry-"
                )
                telemetry_path = Path(temporary_telemetry_dir.name) / "telemetry.jsonl"
            # Faithful Planner: install AIS perf-model regressions before the run,
            # as the production planner does. No-op for rival adapters.
            if ais_bootstrap:
                _bootstrap_ais_regressions(
                    adapter,
                    engine,
                    config,
                    extra_engine_args=agg_args,
                    prefill_engine_args=p_args,
                    decode_engine_args=d_args,
                    performance_model_metadata=resolved_performance_model_metadata,
                    benchmark_granularity=benchmark_granularity,
                )
            elif _supports_planner_bootstrap(engine):
                adapter.set_bootstrap_metadata(
                    {
                        "status": "disabled",
                        "benchmark_granularity": benchmark_granularity,
                    }
                )

            native = _run_mocker_trace_replay(
                replay_trace_files,
                extra_engine_args=agg_args,
                prefill_engine_args=p_args,
                decode_engine_args=d_args,
                router_config=router_config,
                num_workers=num_workers,
                num_prefill_workers=num_prefill_workers,
                num_decode_workers=num_decode_workers,
                replay_concurrency=replay_concurrency,
                replay_mode="offline",
                router_mode=router_mode,
                arrival_speedup_ratio=arrival_speedup_ratio,
                trace_block_size=trace_block_size,
                trace_format=trace_format,
                model_name=model_name,
                sla_ttft_ms=sla_ttft_ms,
                sla_itl_ms=sla_itl_ms,
                sla_e2e_ms=sla_e2e_ms,
                capture_per_request=capture_per_request,
                capture_telemetry=False,
                telemetry_sample_interval_ms=(
                    telemetry_sample_interval_s * 1000.0
                    if telemetry_sample_interval_s is not None
                    else 5_000.0
                ),
                telemetry_jsonl_path=(
                    str(telemetry_path) if telemetry_path is not None else None
                ),
                scaling_policy=adapter,
            )
            planner = adapter.finalize(native.lifecycle_operations)
            replay_report = ReplayReport(
                summary=native.summary,
                per_request=native.per_request,
                coverage=native.coverage,
                planner=planner,
            )

        telemetry_ingestion = (
            _ingest_telemetry_jsonl(telemetry_path, replay_report.planner)
            if telemetry_path is not None
            else None
        )
        report = ArenaReplayResult(
            replay_report=replay_report,
            timeline=(
                telemetry_ingestion.timeline if telemetry_ingestion is not None else []
            ),
            telemetry_artifact=(
                {
                    "contract": TELEMETRY_CONTRACT,
                    "sample_count": telemetry_ingestion.sample_count,
                    "sha256": telemetry_ingestion.sha256,
                }
                if telemetry_ingestion is not None
                else None
            ),
        )
    finally:
        if temporary_telemetry_dir is not None:
            temporary_telemetry_dir.cleanup()

    if report_json is not None:
        write_report_json(report.trace_report, report_json)

    return report
