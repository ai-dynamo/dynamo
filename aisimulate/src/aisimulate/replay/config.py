# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-neutral schema and lowering for one CLI replay invocation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..sweeper.provider import JSONValue
from ..sweeper.replay import BackendDeploymentSpec, ReplayOutputRequirements, ReplaySpec


@dataclass(frozen=True)
class ReplayOutputConfig:
    """Local presentation destinations, excluded from replay semantics."""

    report_json: Path | None = None
    per_request_jsonl: Path | None = None


@dataclass(frozen=True)
class ReplayCliConfig:
    """Canonical base configuration shared by AISimulate and Dynamo CLIs."""

    trace_files: tuple[str, ...]
    extra_engine_args: dict[str, JSONValue] | None
    prefill_engine_args: dict[str, JSONValue] | None
    decode_engine_args: dict[str, JSONValue] | None
    num_workers: int
    num_prefill_workers: int
    num_decode_workers: int
    replay_mode: str
    workload: dict[str, JSONValue]
    goal: dict[str, JSONValue]
    output: ReplayOutputConfig

    @property
    def uses_trace(self) -> bool:
        return bool(self.trace_files)

    @property
    def output_requirements(self) -> ReplayOutputRequirements:
        return ReplayOutputRequirements(
            include_raw_report=True,
            capture_per_request=self.output.per_request_jsonl is not None,
        )

    def to_replay_spec(self) -> ReplaySpec:
        deployment = self._deployment_spec()
        workload = dict(self.workload)
        if self.uses_trace:
            if len(self.trace_files) != 1:
                raise ValueError(
                    "AISimulate Runner replay requires exactly one trace file"
                )
            workload["trace_path"] = self.trace_files[0]
        concurrency = workload.get("replay_concurrency")
        return ReplaySpec(
            backend_deployment=deployment,
            workload=workload,
            goal=dict(self.goal),
            concurrency=(
                _positive_int(concurrency, "replay_concurrency")
                if not self.uses_trace and concurrency is not None
                else None
            ),
        )

    def _deployment_spec(self) -> BackendDeploymentSpec:
        if self.extra_engine_args is not None:
            if (
                self.prefill_engine_args is not None
                or self.decode_engine_args is not None
            ):
                raise ValueError(
                    "--extra-engine-args cannot be combined with prefill/decode engine arguments"
                )
            backend = _engine_backend(self.extra_engine_args)
            return BackendDeploymentSpec(
                deployment_mode="agg",
                backend=backend,
                backend_version=_engine_backend_version(self.extra_engine_args),
                agg_engine_args=dict(self.extra_engine_args),
                num_workers=self.num_workers,
            )

        if self.prefill_engine_args is None or self.decode_engine_args is None:
            raise ValueError(
                "engine replay requires --extra-engine-args or both "
                "--prefill-engine-args and --decode-engine-args"
            )
        prefill_backend = _engine_backend(self.prefill_engine_args)
        decode_backend = _engine_backend(self.decode_engine_args)
        if prefill_backend != decode_backend:
            raise ValueError(
                "prefill and decode engine backends must match: "
                f"{prefill_backend!r} != {decode_backend!r}"
            )
        prefill_version = _engine_backend_version(self.prefill_engine_args)
        decode_version = _engine_backend_version(self.decode_engine_args)
        if prefill_version and decode_version and prefill_version != decode_version:
            raise ValueError(
                "prefill and decode backend versions must match: "
                f"{prefill_version!r} != {decode_version!r}"
            )
        return BackendDeploymentSpec(
            deployment_mode="disagg",
            backend=prefill_backend,
            backend_version=prefill_version or decode_version,
            prefill_engine_args=dict(self.prefill_engine_args),
            decode_engine_args=dict(self.decode_engine_args),
            num_prefill_workers=self.num_prefill_workers,
            num_decode_workers=self.num_decode_workers,
        )


def parse_base_replay_config(args: argparse.Namespace) -> ReplayCliConfig:
    """Validate shared arguments and lower them into the canonical CLI schema."""

    trace_files = tuple(args.trace_files)
    synthetic_args = (args.input_tokens, args.output_tokens, args.request_count)
    uses_synthetic = any(value is not None for value in synthetic_args) or any(
        (
            args.turns_per_session != 1,
            args.shared_prefix_ratio != 0.0,
            args.num_prefix_groups != 0,
            args.inter_turn_delay_ms != 0.0,
        )
    )
    if bool(trace_files) == uses_synthetic:
        raise ValueError(
            "provide either trace_file or all of "
            "--input-tokens/--output-tokens/--request-count"
        )
    if uses_synthetic and not all(value is not None for value in synthetic_args):
        raise ValueError(
            "synthetic replay requires --input-tokens, --output-tokens, and --request-count"
        )
    if args.trace_format == "dynamo" and not trace_files:
        raise ValueError("--trace-format=dynamo requires at least one trace file")
    if args.trace_format != "dynamo" and len(trace_files) > 1:
        raise ValueError(
            f"--trace-format={args.trace_format} requires exactly one trace file"
        )

    if uses_synthetic:
        controllers = (
            args.replay_concurrency,
            args.request_rate,
            args.arrival_interval_ms,
        )
        if sum(value is not None for value in controllers) != 1:
            raise ValueError(
                "synthetic replay requires exactly one of --replay-concurrency, "
                "--request-rate, or --arrival-interval-ms"
            )
    elif args.request_rate is not None or args.arrival_interval_ms is not None:
        raise ValueError(
            "--request-rate and --arrival-interval-ms only apply to synthetic replay"
        )
    if (
        trace_files
        and args.trace_format == "applied_compute_agentic"
        and args.replay_concurrency is None
    ):
        raise ValueError(
            "--trace-format=applied_compute_agentic requires --replay-concurrency "
            "because the source traces do not include first-turn timestamps"
        )
    if args.max_sim_time_seconds is not None:
        _nonnegative_number(args.max_sim_time_seconds, "max_sim_time_seconds")
        if not trace_files:
            raise ValueError(
                "--max-sim-time-seconds currently only supports trace-file replay"
            )
    if args.trace_block_size is not None:
        _positive_int(args.trace_block_size, "trace_block_size")
    if args.agentic_lanes is not None:
        _positive_int(args.agentic_lanes, "agentic_lanes")
        if args.trace_format not in {"agentic_mooncake", "weka", "dynamo"}:
            raise ValueError(
                "--agentic-lanes requires --trace-format=agentic_mooncake, weka, or dynamo"
            )
        if args.replay_concurrency is not None:
            raise ValueError(
                "--agentic-lanes cannot be combined with --replay-concurrency"
            )

    workload: dict[str, JSONValue]
    if trace_files:
        workload = {
            "trace_format": args.trace_format,
            "trace_block_size": args.trace_block_size,
            "arrival_speedup_ratio": args.arrival_speedup_ratio,
            "trace_shared_prefix_ratio": args.trace_shared_prefix_ratio,
            "trace_num_prefix_groups": args.trace_num_prefix_groups,
        }
        if args.replay_concurrency is not None:
            workload["replay_concurrency"] = args.replay_concurrency
        if args.max_sim_time_seconds is not None:
            workload["max_sim_time_ms"] = args.max_sim_time_seconds * 1_000.0
        if args.agentic_lanes is not None:
            workload["agentic_lanes"] = args.agentic_lanes
    else:
        workload = {
            "isl": args.input_tokens,
            "osl": args.output_tokens,
            "request_count": args.request_count,
            "arrival_seed": args.arrival_seed,
            "turns_per_session": args.turns_per_session,
            "shared_prefix_ratio": args.shared_prefix_ratio,
            "num_prefix_groups": args.num_prefix_groups,
            "inter_turn_delay_ms": args.inter_turn_delay_ms,
        }
        if args.replay_concurrency is not None:
            workload["concurrency"] = args.replay_concurrency
            workload["replay_concurrency"] = args.replay_concurrency
        if args.request_rate is not None:
            workload["request_rate"] = args.request_rate
        if args.arrival_interval_ms is not None:
            workload["arrival_interval_ms"] = args.arrival_interval_ms

    sla = {
        name: value
        for name, value in (
            ("ttft_ms", args.sla_ttft_ms),
            ("itl_ms", args.sla_itl_ms),
            ("e2e_ms", args.sla_e2e_ms),
        )
        if value is not None
    }
    return ReplayCliConfig(
        trace_files=trace_files,
        extra_engine_args=_json_object(args.extra_engine_args, "--extra-engine-args"),
        prefill_engine_args=_json_object(
            args.prefill_engine_args, "--prefill-engine-args"
        ),
        decode_engine_args=_json_object(
            args.decode_engine_args, "--decode-engine-args"
        ),
        num_workers=_positive_int(args.num_workers, "num_workers"),
        num_prefill_workers=_positive_int(
            args.num_prefill_workers, "num_prefill_workers"
        ),
        num_decode_workers=_positive_int(args.num_decode_workers, "num_decode_workers"),
        replay_mode=args.replay_mode,
        workload=workload,
        goal={"target": "throughput", "sla": sla or None},
        output=ReplayOutputConfig(
            report_json=Path(args.report_json) if args.report_json else None,
            per_request_jsonl=(
                Path(args.per_request_jsonl) if args.per_request_jsonl else None
            ),
        ),
    )


def _json_object(value: str | None, flag: str) -> dict[str, JSONValue] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag} must contain valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag} must contain a JSON object")
    return parsed


def _engine_backend(config: dict[str, JSONValue]) -> str:
    nested = config.get("rank")
    nested_backend = nested.get("backend") if isinstance(nested, dict) else None
    values = [
        value
        for value in (
            config.get("engine_type"),
            config.get("aic_backend"),
            nested_backend,
        )
        if value is not None
    ]
    if any(not isinstance(value, str) for value in values):
        raise ValueError("engine backend fields must be strings")
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError(f"engine backend aliases conflict: {values}")
    backend = values[0] if values else "vllm"
    if backend not in {"vllm", "sglang", "trtllm"}:
        raise ValueError(f"unsupported engine backend {backend!r}")
    return backend


def _engine_backend_version(config: dict[str, JSONValue]) -> str:
    value = config.get("aic_backend_version") or config.get("backend_version")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("engine backend version must be a string")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_number(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)
