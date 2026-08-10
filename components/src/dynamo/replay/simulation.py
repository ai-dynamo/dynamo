# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-backed implementation of the Sweeper replay-runner contract.

This is a compatibility composition over the current Dynamo replay API.  Sweeper
passes only a serializable ``ReplaySpec``; this module resolves Dynamo runtime
hooks and translates the neutral spec to the existing replay entry points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any

from aisimulate.aic import materialize_aic_num_gpu_blocks
from aisimulate.sweeper.provider import JSONValue, RuntimeHookSpec
from aisimulate.sweeper.replay import (
    HookCapability,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RunnerCapabilities,
)
from dynamo.llm import KvRouterConfig
from dynamo.mocker import MockEngineArgs
from dynamo.replay.api import run_synthetic_trace_replay, run_trace_replay

_PLANNER_HOOK = HookCapability(
    provider="dynamo.planner",
    kind="scaling_policy",
    api_version=1,
)
_ROUTER_HOOK = HookCapability(
    provider="dynamo.router",
    kind="placement_policy",
    api_version=1,
)
_REPLAY_SPEC_API_VERSION = 1
_SUPPORTED_BACKEND_TOPOLOGIES = (
    ("vllm", "agg"),
    ("vllm", "disagg"),
    ("sglang", "agg"),
    ("sglang", "disagg"),
    ("trtllm", "agg"),
)


@dataclass(frozen=True)
class DynamoReplayRunnerFactory:
    """Serializable factory for the current Dynamo offline replay composition."""

    trace_block_size: int = 512
    benchmark_granularity: int = 8

    def capabilities(self) -> RunnerCapabilities:
        """Advertise the backend/topology and Dynamo hook support."""

        return RunnerCapabilities(
            # Runner-owned constant: do not inherit the consumer package's default,
            # otherwise an old Dynamo wheel can self-certify against a newer spec.
            replay_spec_api_version=_REPLAY_SPEC_API_VERSION,
            supported_backend_topologies=_SUPPORTED_BACKEND_TOPOLOGIES,
            supported_hooks=(_PLANNER_HOOK, _ROUTER_HOOK),
        )

    def create(self, worker_id: int) -> DynamoReplayRunner:
        """Create one reusable worker-local runner."""

        return DynamoReplayRunner(
            worker_id=worker_id,
            capabilities=self.capabilities(),
            trace_block_size=self.trace_block_size,
            benchmark_granularity=self.benchmark_granularity,
        )


@dataclass
class DynamoReplayRunner:
    """Translate neutral replay specifications to the current Dynamo replay API."""

    worker_id: int
    capabilities: RunnerCapabilities
    trace_block_size: int = 512
    benchmark_granularity: int = 8

    def run(
        self,
        spec: ReplaySpec,
        *,
        output_requirements: ReplayOutputRequirements | None = None,
    ) -> ReplayReport:
        """Execute one trace or synthetic replay with requested Dynamo hooks."""

        output_requirements = output_requirements or ReplayOutputRequirements()
        self.capabilities.require_compatible(spec)
        planner_config, router_mode, router_config = self._resolve_hooks(
            spec.runtime_hooks
        )
        common: dict[str, Any] = {
            "router_mode": router_mode,
            "router_config": router_config,
            "arrival_speedup_ratio": self._arrival_speedup_ratio(spec),
            "replay_concurrency": self._effective_in_flight_cap(spec),
            "planner_config": planner_config,
            "benchmark_granularity": self.benchmark_granularity,
            # Sweeper uses the lightweight default. CLI and other callers can
            # explicitly request detailed output through the Runner contract.
            "capture_per_request": output_requirements.capture_per_request,
            "capture_planner_details": False,
            **self._goodput_sla_kwargs(spec),
        }

        if self._is_trace(spec):
            common["trace_block_size"] = self.trace_block_size
            report = self._run_trace(spec, common)
        else:
            common.update(self._synthetic_kwargs(spec))
            report = self._run_synthetic(spec, common)

        metrics, metadata = self._normalize_report(report, output_requirements)
        self._require_goodput_metric(metrics, spec)
        return ReplayReport(metrics=metrics, metadata=metadata)

    def close(self) -> None:
        """Release runner-local resources.

        The current replay entry points are call-scoped and retain no resources
        between candidates.
        """

    @staticmethod
    def _resolve_hooks(
        hooks: tuple[RuntimeHookSpec, ...],
    ) -> tuple[dict[str, JSONValue] | None, str, KvRouterConfig | None]:
        planner_config: dict[str, JSONValue] | None = None
        router_mode = "round_robin"
        router_config: KvRouterConfig | None = None
        planner_seen = False
        router_seen = False
        for hook in hooks:
            if _PLANNER_HOOK.supports(hook):
                if planner_seen:
                    raise ValueError(
                        "ReplaySpec contains multiple Planner scaling hooks"
                    )
                planner_seen = True
                raw_config = hook.config.get("planner_config")
                if not isinstance(raw_config, dict):
                    raise TypeError(
                        "Dynamo Planner hook config requires a planner_config mapping"
                    )
                planner_config = raw_config
                continue
            if _ROUTER_HOOK.supports(hook):
                if router_seen:
                    raise ValueError(
                        "ReplaySpec contains multiple Router placement hooks"
                    )
                router_seen = True
                router_mode = str(hook.config.get("router_mode", "kv_router"))
                raw_config = hook.config.get("router_config")
                if not isinstance(raw_config, dict):
                    raise TypeError(
                        "Dynamo Router hook config requires a router_config mapping"
                    )
                router_config = KvRouterConfig.from_json(json.dumps(raw_config))
                continue
            raise ValueError(
                f"unsupported Dynamo runtime hook "
                f"{hook.provider}:{hook.kind}@{hook.api_version}"
            )
        return planner_config, router_mode, router_config

    @staticmethod
    def _is_trace(spec: ReplaySpec) -> bool:
        return spec.workload.get("trace_path") is not None

    @staticmethod
    def _arrival_speedup_ratio(spec: ReplaySpec) -> float:
        if DynamoReplayRunner._is_trace(spec):
            return _float_value(
                spec.workload.get("arrival_speedup_ratio", 1.0),
                "arrival_speedup_ratio",
            )
        return 1.0

    @staticmethod
    def _effective_in_flight_cap(spec: ReplaySpec) -> int | None:
        if DynamoReplayRunner._is_trace(spec):
            value = spec.workload.get("replay_concurrency")
            return (
                _int_value(value, "replay_concurrency") if value is not None else None
            )
        if spec.concurrency is not None:
            return spec.concurrency
        value = spec.workload.get("concurrency")
        return _int_value(value, "concurrency") if value is not None else None

    @staticmethod
    def _goodput_sla_kwargs(spec: ReplaySpec) -> dict[str, float | None]:
        raw_sla = spec.goal.get("sla")
        if not isinstance(raw_sla, dict):
            return {}
        return {
            "sla_ttft_ms": _optional_float(raw_sla.get("ttft_ms"), "sla.ttft_ms"),
            "sla_itl_ms": _optional_float(raw_sla.get("itl_ms"), "sla.itl_ms"),
            "sla_e2e_ms": _optional_float(raw_sla.get("e2e_ms"), "sla.e2e_ms"),
        }

    @staticmethod
    def _engine_args(payload: dict[str, JSONValue] | None) -> MockEngineArgs:
        if payload is None:
            raise ValueError("ReplaySpec is missing required engine arguments")
        return MockEngineArgs.from_json(
            json.dumps(materialize_aic_num_gpu_blocks(payload))
        )

    def _run_trace(self, spec: ReplaySpec, common: dict[str, Any]):
        deployment = spec.backend_deployment
        trace_path = spec.workload["trace_path"]
        if not isinstance(trace_path, str):
            raise TypeError("trace workload requires a string trace_path")
        trace_format = spec.workload.get("trace_format", "mooncake")
        if not isinstance(trace_format, str):
            raise TypeError("trace workload requires a string trace_format")
        if deployment.deployment_mode == "agg":
            return run_trace_replay(
                trace_files=trace_path,
                trace_format=trace_format,
                extra_engine_args=self._engine_args(deployment.agg_engine_args),
                num_workers=deployment.num_workers,
                **common,
            )
        return run_trace_replay(
            trace_files=trace_path,
            trace_format=trace_format,
            prefill_engine_args=self._engine_args(deployment.prefill_engine_args),
            decode_engine_args=self._engine_args(deployment.decode_engine_args),
            num_prefill_workers=deployment.num_prefill_workers,
            num_decode_workers=deployment.num_decode_workers,
            **common,
        )

    def _run_synthetic(self, spec: ReplaySpec, common: dict[str, Any]):
        deployment = spec.backend_deployment
        if deployment.deployment_mode == "agg":
            return run_synthetic_trace_replay(
                extra_engine_args=self._engine_args(deployment.agg_engine_args),
                num_workers=deployment.num_workers,
                **common,
            )
        return run_synthetic_trace_replay(
            prefill_engine_args=self._engine_args(deployment.prefill_engine_args),
            decode_engine_args=self._engine_args(deployment.decode_engine_args),
            num_prefill_workers=deployment.num_prefill_workers,
            num_decode_workers=deployment.num_decode_workers,
            **common,
        )

    @staticmethod
    def _synthetic_kwargs(spec: ReplaySpec) -> dict[str, JSONValue]:
        workload = spec.workload
        request_rate = workload.get("request_rate")
        rate: float | None = None
        arrival_interval_ms: float | None = None
        if request_rate is not None:
            rate = _float_value(request_rate, "request_rate")
            if rate <= 0.0:
                raise ValueError(
                    f"synthetic replay requires a positive request_rate, got {rate}"
                )
            arrival_interval_ms = 1000.0 / rate
        load: float
        if spec.concurrency is not None:
            load = float(spec.concurrency)
        elif workload.get("concurrency") is not None:
            load = _float_value(workload["concurrency"], "concurrency")
        elif rate is not None:
            load = rate
        else:
            raise ValueError(
                "synthetic replay requires concrete concurrency or request_rate"
            )
        ratio = _float_value(workload["num_request_ratio"], "num_request_ratio")
        return {
            "input_tokens": _int_value(workload["isl"], "isl"),
            "output_tokens": _int_value(workload["osl"], "osl"),
            "request_count": max(1, round(ratio * load)),
            "arrival_interval_ms": arrival_interval_ms,
            "turns_per_session": _int_value(
                workload.get("turns_per_session", 1), "turns_per_session"
            ),
            "shared_prefix_ratio": _float_value(
                workload.get("shared_prefix_ratio", 0.0), "shared_prefix_ratio"
            ),
            "num_prefix_groups": _int_value(
                workload.get("num_prefix_groups", 0), "num_prefix_groups"
            ),
            "inter_turn_delay_ms": _float_value(
                workload.get("inter_turn_delay_ms", 0.0), "inter_turn_delay_ms"
            ),
        }

    @staticmethod
    def _normalize_report(
        report: Any,
        output_requirements: ReplayOutputRequirements,
    ) -> tuple[dict[str, float], dict[str, JSONValue]]:
        metadata: dict[str, JSONValue] = {}
        if hasattr(report, "summary"):
            trace_report = dict(report.summary)
            planner = getattr(report, "planner", None)
            if planner is not None:
                trace_report["planner_total_ticks"] = float(planner.total_ticks)
                metadata["planner_total_ticks"] = int(planner.total_ticks)
        elif hasattr(report, "trace_report"):
            trace_report = dict(report.trace_report)
            if hasattr(report, "total_ticks"):
                trace_report["planner_total_ticks"] = float(report.total_ticks)
                metadata["planner_total_ticks"] = int(report.total_ticks)
        else:
            trace_report = dict(report)

        metrics: dict[str, float] = {}
        for name, value in trace_report.items():
            if isinstance(value, Real):
                metrics[str(name)] = float(value)
        if (
            output_requirements.include_raw_report
            or output_requirements.capture_per_request
        ):
            if hasattr(report, "to_dict"):
                native_report = report.to_dict()
            elif hasattr(report, "trace_report"):
                native_report = {"trace_report": dict(report.trace_report)}
            else:
                native_report = dict(report)
            metadata["native_report"] = native_report
        return metrics, metadata

    @staticmethod
    def _require_goodput_metric(metrics: dict[str, float], spec: ReplaySpec) -> None:
        if "goodput_output_throughput_tok_s" in metrics:
            return
        target = str(_plain(spec.goal.get("target", "throughput")))
        raw_objectives = spec.goal.get("pareto_objectives")
        objectives = (
            {str(_plain(item)) for item in raw_objectives}
            if target == "pareto" and isinstance(raw_objectives, list)
            else {target}
        )
        if not objectives.intersection({"goodput", "goodput_per_gpu"}):
            return
        raise RuntimeError(
            "Dynamo replay did not emit goodput_output_throughput_tok_s for a "
            "goodput objective; install a replay version with per-request SLA "
            "accounting"
        )


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _numeric_value(value: JSONValue, name: str) -> str | int | float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"replay field {name} must be numeric")
    return value


def _float_value(value: JSONValue, name: str) -> float:
    return float(_numeric_value(value, name))


def _int_value(value: JSONValue, name: str) -> int:
    return int(_numeric_value(value, name))


def _optional_float(value: JSONValue, name: str) -> float | None:
    return _float_value(value, name) if value is not None else None
