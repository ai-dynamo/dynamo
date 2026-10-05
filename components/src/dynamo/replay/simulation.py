# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-backed implementation of the Sweeper replay-runner contract.

This is a compatibility composition over the current Dynamo replay API.  Sweeper
passes only a serializable ``ReplaySpec``; this module resolves Dynamo runtime
hooks and translates the neutral spec to the existing replay entry points.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from aisimulate.runner import (
    EngineReplayRunner,
    EngineReplayRunnerFactory,
    _execution_target_model,
)
from aisimulate.sweeper.provider import JSONValue, RuntimeHookSpec
from aisimulate.sweeper.replay import (
    HookCapability,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RunnerCapabilities,
)

from dynamo.llm import AisPerfConfig, KvRouterConfig
from dynamo.mocker import MockEngineArgs
from dynamo.replay.api import (
    TelemetryOptions,
    _add_agentic_model_projection,
    _planner_config_arg,
    _planner_replay_adapter,
    run_trace_replay,
)
from dynamo.replay.config import lower_upstream_engine_args
from dynamo.router.simulation.config import ConversationAffinityConfig

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


@dataclass(frozen=True)
class DynamoReplayRunnerFactory:
    """Serializable factory for the current Dynamo offline replay composition."""

    trace_block_size: int = 512
    benchmark_granularity: int = 8

    def capabilities(self) -> RunnerCapabilities:
        """Advertise the backend/topology and Dynamo hook support."""

        engine_capabilities = EngineReplayRunnerFactory().capabilities()
        return RunnerCapabilities(
            # Runner-owned constant: do not inherit the consumer package's default,
            # otherwise an old Dynamo wheel can self-certify against a newer spec.
            replay_spec_api_version=_REPLAY_SPEC_API_VERSION,
            supported_backend_topologies=tuple(
                (backend, topology)
                for backend, topology in engine_capabilities.supported_backend_topologies
                if backend in ("vllm", "sglang", "trtllm")
                and topology in ("agg", "disagg")
            ),
            supported_hooks=(_PLANNER_HOOK, _ROUTER_HOOK),
            supports_disaggregated_attention_dp=(
                engine_capabilities.supports_disaggregated_attention_dp
            ),
            supported_execution_modes=("offline",),
            supported_trace_formats=(
                "mooncake",
                "mooncake-delta",
                "agentic_mooncake",
                "applied_compute_agentic",
                "dynamo",
                "weka",
            ),
            supports_agentic_lanes=engine_capabilities.supports_agentic_lanes,
            # Full AgentX runtime conformance remains a separate checkpoint.
            # Keep the public runner honest about the narrower integration here.
            supported_agentic_topologies=tuple(
                topology
                for topology in engine_capabilities.supported_agentic_topologies
                if topology in ("agg", "disagg")
            ),
            supported_agentic_backends=engine_capabilities.supported_agentic_backends,
            supports_state_cache=True,
            supports_agentic_snapshots=True,
            supports_agentic_warmup=True,
            supports_agentic_profile=True,
            supports_agentic_host_offload=False,
            supports_agentic_speculative_decoding=False,
            agentic_qualification="functional_only",
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
        """Run every supported offline workload through the shared executor."""

        output = output_requirements or ReplayOutputRequirements()
        spec = _normalize_legacy_spec(spec)
        self.capabilities.require_compatible(spec)
        (
            planner,
            router_mode,
            router_config,
            ais_perf_config,
            affinity,
        ) = self._resolve_hooks(spec.runtime_hooks)
        profile = spec.workload.get("agentic_profile")
        if planner is not None and (affinity is not None or profile is not None):
            raise ValueError(
                "conversation affinity and AgentX profiles require static worker pools without a Planner"
            )
        if affinity is not None and router_mode != "kv_router":
            raise ValueError("conversation affinity requires router.policy='kv_router'")
        if (
            affinity is not None or profile is not None
        ) and spec.backend_deployment.backend not in {"vllm", "sglang"}:
            raise ValueError("Dynamo conversation replay supports vllm or sglang")
        runtime = _DynamoReplayRuntime(
            spec,
            router_mode,
            router_config,
            ais_perf_config,
            affinity,
            planner,
            self.benchmark_granularity,
            output,
        )
        runner = EngineReplayRunner(
            worker_id=self.worker_id,
            capabilities=self.capabilities,
            trace_block_size=self.trace_block_size,
            runtime=runtime,
        )
        try:
            # The native observer owns telemetry. The shared runner still owns
            # input materialization and report normalization for every policy.
            report = runner.run(
                # Dynamo hooks have been resolved into this runtime's composition;
                # they are not a second request for the engine's local adapters.
                replace(spec, adapters={}),
                output_requirements=replace(
                    output,
                    include_raw_report=True,
                    capture_telemetry=False,
                ),
            )
        finally:
            runner.close()
        self._require_goodput_metric(report.metrics, spec)
        metrics = dict(report.metrics)
        metadata = dict(report.metadata)
        native = metadata.pop("native_report")
        for name in ("agentic_graph", "agentic_model_projection", "routing_policy"):
            if name in native:
                metadata[name] = native[name]
        planner_details = native.get("planner")
        if planner_details is not None:
            ticks = int(planner_details["total_ticks"])
            metrics["planner_total_ticks"] = float(ticks)
            metadata["planner_total_ticks"] = ticks
        if output.capture_telemetry:
            metadata["telemetry"] = native["telemetry"]
        if (
            output.include_raw_report
            or output.capture_per_request
            or output.capture_memory_diagnostics
            or output.capture_performance_diagnostics
        ):
            metadata["native_report"] = {**native, "summary": metrics}
        return ReplayReport(metrics=metrics, metadata=metadata)

    def close(self) -> None:
        """Release runner-local resources.

        The current replay entry points are call-scoped and retain no resources
        between candidates.
        """

    @staticmethod
    def _resolve_hooks(
        hooks: tuple[RuntimeHookSpec, ...],
    ) -> tuple[
        dict[str, JSONValue] | None,
        str,
        KvRouterConfig | None,
        AisPerfConfig | None,
        dict[str, JSONValue] | None,
    ]:
        planner_config: dict[str, JSONValue] | None = None
        router_mode = "round_robin"
        router_config: KvRouterConfig | None = None
        ais_perf_config: AisPerfConfig | None = None
        affinity: dict[str, JSONValue] | None = None
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
                raw_affinity = hook.config.get("affinity")
                if raw_affinity is not None:
                    affinity = ConversationAffinityConfig.model_validate(
                        raw_affinity
                    ).model_dump(mode="json")
                raw_ais = hook.config.get("ais_perf_config")
                if raw_ais is not None:
                    if not isinstance(raw_ais, dict):
                        raise TypeError(
                            "Dynamo Router AIS config must be a mapping or null"
                        )
                    ais_perf_config = AisPerfConfig(config=raw_ais)
                continue
            raise ValueError(
                f"unsupported Dynamo runtime hook "
                f"{hook.provider}:{hook.kind}@{hook.api_version}"
            )
        return planner_config, router_mode, router_config, ais_perf_config, affinity

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


def _normalize_legacy_spec(spec: ReplaySpec) -> ReplaySpec:
    """Translate legacy input spelling without constructing engines or traffic."""
    deployment = spec.backend_deployment
    roles = {}
    for role in ("agg_engine_args", "prefill_engine_args", "decode_engine_args"):
        args = getattr(deployment, role)
        if args is not None and args.get("ais_perf_config") is not None:
            args = dict(args)
            if args.get("timing_model") is not None:
                raise ValueError("ais_perf_config cannot be combined with timing_model")
            args["timing_model"] = {
                "type": "external",
                "provider": "aic",
                "config": args.pop("ais_perf_config"),
            }
            roles[role] = args
    if roles:
        spec = replace(spec, backend_deployment=replace(deployment, **roles))
    workload = dict(spec.workload)
    if workload.get("source_type") is not None:
        return spec
    if (
        workload.get("trace_path") is not None
        or workload.get("trace_paths") is not None
    ):
        deployment = spec.backend_deployment
        if deployment.deployment_mode == "disagg" and workload.get("trace_format") in {
            "weka",
            "agentic_mooncake",
            "dynamo",
        }:
            # The legacy runner used decode's model as the shared trace
            # projection. Fill only an absent prefill identity; the shared
            # runner still rejects explicitly different models.
            prefill_model = _execution_target_model(
                deployment, "prefill", deployment.prefill_engine_args or {}
            )
            decode_model = _execution_target_model(
                deployment, "decode", deployment.decode_engine_args or {}
            )
            if prefill_model is None and decode_model is not None:
                metadata = dict(deployment.performance_model_metadata)
                prefill = dict(metadata.get("prefill", {}))
                prefill["config"] = {
                    **prefill.get("config", {}),
                    "model": decode_model,
                }
                metadata["prefill"] = prefill
                spec = replace(
                    spec,
                    backend_deployment=replace(
                        deployment, performance_model_metadata=metadata
                    ),
                )
        workload["source_type"] = "trace"
        default_load = (
            "concurrency"
            if workload.get("replay_concurrency") is not None
            else "trace_timestamps"
        )
    else:
        workload["source_type"] = (
            "synthetic-session"
            if workload.get("turns_per_session", 1) != 1
            else "synthetic"
        )
        if spec.concurrency is not None:
            workload["concurrency"] = spec.concurrency
        if workload.get("concurrency") is not None:
            default_load = "concurrency"
        elif workload.get("kv_load_ratio") is not None:
            default_load = "kv_capacity_fraction"
        else:
            # Legacy request_rate is evenly spaced unless Poisson is explicit.
            default_load = "constant_rate"
    if workload.get("load_type") is None:
        workload["load_type"] = default_load
    return replace(spec, workload=workload)


@dataclass(frozen=True)
class _DynamoReplayRuntime:
    """Supply Dynamo policies and optional observers to the shared executor."""

    spec: ReplaySpec
    router_mode: str
    router_config: KvRouterConfig | None
    ais_perf_config: AisPerfConfig | None
    affinity: dict[str, JSONValue] | None
    planner_config: dict[str, JSONValue] | None
    benchmark_granularity: int
    output: ReplayOutputRequirements

    def run_replay_json(self, execution_spec_json: str) -> str:
        scope = nullcontext(None)
        if self.planner_config is not None:
            deployment = self.spec.backend_deployment

            def args(payload):
                return (
                    None
                    if payload is None
                    else MockEngineArgs.from_json(
                        json.dumps(lower_upstream_engine_args(payload))
                    )
                )

            scope = _planner_replay_adapter()(
                extra_engine_args=args(deployment.agg_engine_args),
                prefill_engine_args=args(deployment.prefill_engine_args),
                decode_engine_args=args(deployment.decode_engine_args),
                planner_config_arg=_planner_config_arg(self.planner_config),
                performance_model_metadata=deployment.performance_model_metadata,
                benchmark_granularity=self.benchmark_granularity,
                capture_details=self.output.include_raw_report,
            )
        with scope as planner:
            report = json.loads(
                run_trace_replay(
                    [],
                    router_mode=self.router_mode,
                    router_config=self.router_config,
                    ais_perf_config=self.ais_perf_config,
                    replay_spec_json=execution_spec_json,
                    affinity=self.affinity,
                    scaling_policy=planner,
                    capture_planner_details=self.output.include_raw_report,
                    telemetry_options=(
                        TelemetryOptions(
                            sample_interval_ms=self.output.telemetry_sample_interval_ms,
                        )
                        if self.output.capture_telemetry
                        else None
                    ),
                )
            )
            if isinstance(report.get("agentic_graph"), dict):
                traffic = json.loads(execution_spec_json).get("traffic", {})
                _add_agentic_model_projection(report, traffic.get("execution_model"))
            report["planner"] = (
                planner.finalize(report["lifecycle_operations"]).to_dict()
                if planner is not None
                else None
            )
            return json.dumps(report, allow_nan=False)


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value
