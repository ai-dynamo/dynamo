# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed command-line configuration for router mocker tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MockerConfig:
    speedup_ratio: float | None = None
    block_size: int | None = None
    num_gpu_blocks: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    preemption_mode: str | None = None
    dp_size: int | None = None
    planner_profile_data: str | Path | None = None
    aic_perf_model: bool = False
    aic_system: str | None = None
    aic_backend_version: str | None = None
    aic_tp_size: int | None = None
    bootstrap_ports: str | None = None
    zmq_kv_events_ports: str | None = None
    zmq_replay_ports: str | None = None
    response_replay_trace_path: str | Path | None = None
    router_mode: str | None = None
    router_session_affinity_ttl_secs: int | None = None

    def __post_init__(self) -> None:
        if self.dp_size is not None and self.dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {self.dp_size}")

    @classmethod
    def from_value(cls, value: MockerConfig | Mapping[str, Any] | None) -> MockerConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value

        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - field_names)
        if unknown:
            raise ValueError(f"Unknown mocker config field(s): {', '.join(unknown)}")
        return cls(**value)

    def to_cli_args(self) -> list[str]:
        args: list[str] = []
        scalar_flags = (
            ("--speedup-ratio", self.speedup_ratio),
            ("--block-size", self.block_size),
            ("--num-gpu-blocks-override", self.num_gpu_blocks),
            ("--max-num-seqs", self.max_num_seqs),
            ("--max-num-batched-tokens", self.max_num_batched_tokens),
            ("--preemption-mode", self.preemption_mode),
            ("--data-parallel-size", self.dp_size),
            ("--planner-profile-data", self.planner_profile_data),
            ("--aic-system", self.aic_system),
            ("--aic-backend-version", self.aic_backend_version),
            ("--aic-tp-size", self.aic_tp_size),
            ("--bootstrap-ports", self.bootstrap_ports),
            ("--zmq-kv-events-ports", self.zmq_kv_events_ports),
            ("--zmq-replay-ports", self.zmq_replay_ports),
            ("--response-replay-trace-path", self.response_replay_trace_path),
            ("--router-mode", self.router_mode),
            (
                "--router-session-affinity-ttl-secs",
                self.router_session_affinity_ttl_secs,
            ),
        )
        for flag, value in scalar_flags:
            if value is not None:
                args.extend([flag, str(value)])

        boolean_flags = (
            (
                self.enable_prefix_caching,
                "--enable-prefix-caching",
                "--no-enable-prefix-caching",
            ),
            (
                self.enable_chunked_prefill,
                "--enable-chunked-prefill",
                "--no-enable-chunked-prefill",
            ),
        )
        for enabled, positive_flag, negative_flag in boolean_flags:
            if enabled is not None:
                args.append(positive_flag if enabled else negative_flag)

        if self.aic_perf_model:
            args.append("--aic-perf-model")
        return args
