# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, intentionally small on-demand profiling contract."""

from typing import Literal

import msgspec


class Target(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    target_id: str
    process_epoch: str


class Worker(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    target_id: str
    process_epoch: str
    backend: Literal["vllm", "sglang"]
    control_url: str
    # Paths/options are operator-owned, never supplied by the requesting client.
    start_options: dict = msgspec.field(default_factory=dict)
    artifact_location: str | None = None
    # Require the lightweight worker agent to attest the configured epoch.
    # Off preserves compatibility with images that only expose start/stop.
    verify_runtime: bool = False
    expected_rank_ids: list[str] = msgspec.field(default_factory=list)
    # Unique report prefix of this replica's prepared Nsight launch.
    nsys_output_prefix: str | None = None


class Config(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    socket_path: str
    output_dir: str
    workers: list[Worker] = msgspec.field(default_factory=list)
    max_duration_ms: int = 60000
    start_timeout_s: float = 10.0
    finalize_timeout_s: float = 1200.0
    max_sessions: int = 64
    # A reference to an existing backend; no assumption of a local OTLP archive.
    lifecycle_evidence_location: str | None = None
    # Temporary operator override for environments that deny perf attachment.
    frontend_perf_enabled: bool = True
    # Enables in-process NVTX capture; requires ai-dynamo[profiling] and a
    # prepared CPU-only Nsight launch with automatic SQLite export.
    nsys_frontend_output_prefix: str | None = None
    clock_probe_timeout_s: float = 1.0
    clock_probe_samples: int = 3
    artifact_validation_timeout_s: float = 120.0
    max_trace_bytes: int = 8 * 1024 * 1024 * 1024


class RuntimeSnapshot(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """Frontend-owned runtime information passed explicitly to the coordinator."""

    frontend_pid: int
    frontend_epoch: str
    hostname: str
    workers: list[Worker] = msgspec.field(default_factory=list)
    context: dict = msgspec.field(default_factory=dict)


class Plan(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    client_request_id: str
    targets: list[Target]
    # Best-effort lease from acceptance, including clock probes and startup.
    # Native recording duration and export/validation timeouts are separate.
    duration_ms: int
    mode: Literal["torch_profile", "nsys"] = "torch_profile"
    schema_version: Literal[1] = 1
    trace_ids: list[str] = msgspec.field(default_factory=list)
    request_ids: list[str] = msgspec.field(default_factory=list)
    reason: str = "client_request"
