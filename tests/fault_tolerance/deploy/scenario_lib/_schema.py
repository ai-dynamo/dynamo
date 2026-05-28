# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dataclass schema for scenario YAML files. Kept independent of pydantic
# (no extra deps) — validation lives in _loader.py.
#
# Schema is intentionally minimal: every field a scenario can express
# maps to one dataclass field. The loader is strict — unknown YAML keys
# fail to parse so typos surface at collection time.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Deployment:
    """Where + how the deployment lands on k8s."""

    backend: str  # vllm | mocker | trtllm | sglang
    topology: str = "disagg"  # disagg | agg
    units: int = 2  # prod-unit replica multiplier
    image: Optional[str] = None  # full image tag; None → backend default
    template: Optional[str] = None  # DGD template basename (without .yaml)
    storage_class: Optional[str] = None  # falls back to cluster default
    # Per-service replica overrides. Maps service-name → replica count.
    # When set, these EXACT counts are used (units is ignored for those
    # services). Useful for asymmetric topologies like ``{Frontend: 2,
    # decode: 4}`` where the FE-to-worker ratio is part of the test.
    replicas: Optional[dict[str, int]] = None
    # PVC reuse knobs — None = framework default, "" = disable
    model_cache_pvc: Optional[str] = None
    log_pvc: Optional[str] = None


@dataclass
class Router:
    """FE-side router configuration. ``mode`` sets DYN_ROUTER_MODE;
    ``knobs`` is a key→value dict applied to the Frontend service env
    (each entry becomes one set_env_var call on spec["Frontend"])."""

    mode: str = "kv"  # DYN_ROUTER_MODE value
    knobs: dict[str, str] = field(default_factory=dict)


@dataclass
class Admission:
    """Admission-control knobs — applied to worker services
    (VllmPrefillWorker + VllmDecodeWorker). Same shape as Router.knobs;
    typical keys: DYN_TCP_WORKER_POOL_SIZE, DYN_TCP_WORK_QUEUE_SIZE,
    DYN_VLLM_REJECT_QUEUE_THRESHOLD."""

    knobs: dict[str, str] = field(default_factory=dict)


@dataclass
class Shape:
    """Workload shape (prefix-sharing pattern, ISL distribution). ``type``
    selects the canonical shape; the remaining fields are knobs
    overridable per-scenario. The loader validates which fields are
    meaningful per type."""

    type: str  # no_prefix | same_prefix | partial_prefix | long_isl | high_qps | custom
    # Prefix-cache shaping (mutually exclusive groups)
    num_prefix_prompts: Optional[int] = None
    prefix_prompt_length: Optional[int] = None
    shared_system_prompt_length: Optional[int] = None
    # ISL/OSL distribution (either seq_dist OR mean+stddev)
    seq_dist: Optional[str] = None
    input_tokens_mean: Optional[int] = None
    input_tokens_stddev: Optional[int] = None
    output_tokens_mean: Optional[int] = None
    output_tokens_stddev: Optional[int] = None


@dataclass
class LoadCommon:
    """Common LoadConfig fields applied to every rung unless the rung
    overrides them."""

    request_timeout_seconds: float = 30.0
    streaming: bool = True
    ignore_eos: bool = True
    connection_reuse_strategy: str = "never"
    warmup_requests: int = 0
    # Optional per-LoadConfig knobs that show up in the cliff/admission
    # scenarios — kept here so the schema covers them.
    request_cancellation_rate: Optional[float] = None
    goodput: Optional[list[str]] = None
    # aiperf install spec — pip-install target inside the load Job pod.
    # None → LoadConfig.aiperf_ref default (pinned working commit). Set
    # per-scenario to test newer / older revisions without rebuilding
    # any image. Format: "aiperf @ git+https://...@<commit-or-branch>".
    aiperf_ref: Optional[str] = None


@dataclass
class Rung:
    """One AIPerf job within the scenario's rung sequence."""

    name: str
    concurrency: int
    duration_minutes: float
    # Optional per-rung shape override. If None, the scenario's
    # top-level shape applies.
    shape: Optional[Shape] = None


@dataclass
class Load:
    """Workload definition. Either:
    - rungs: list of (name, concurrency, duration) — the runner
      generates StartLoad/WaitForLoadCompletion event pairs in order
    - OR an explicit events: list (set on Scenario) that drives the
      sequence manually. If both are set, events: wins.
    """

    shape: Shape
    rungs: list[Rung] = field(default_factory=list)
    common: LoadCommon = field(default_factory=LoadCommon)


@dataclass
class EventSpec:
    """Reference to a framework Event class (by name) + its kwargs.

    ``kind`` matches a key in the event registry (see _events.py:
    EVENT_REGISTRY). ``params`` is passed to the Event's constructor
    as kwargs. Special-case: a StartLoad whose params contains
    ``load_ref: <rung_name>`` is materialised against the matching
    rung from the scenario's load.rungs list.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSpec:
    """Reference to a framework Report class + kwargs (see
    _reports.py: REPORT_REGISTRY). Same shape as EventSpec."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckSpec:
    """Reference to a framework Check class + kwargs (see _checks.py:
    CHECK_REGISTRY). Each check produces a pass/fail row in the
    scenario report."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedRange:
    """Documentation for an expected outcome. Not asserted by default;
    the test framework can opt-in to range-checking via a per-kind
    convention (see test_router_modes for the slope-vs-history check)."""

    expected_range: Optional[list[float]] = None  # [low, high]
    observed_history: list[dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class Scenario:
    """Top-level scenario object. ``kind`` must match the parent
    directory name (validated by the loader)."""

    kind: str  # router_memory | admission_control | endurance | ad_hoc
    name: str
    description: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    deployment: Deployment = field(default_factory=lambda: Deployment(backend="vllm"))
    router: Router = field(default_factory=Router)
    admission: Optional[Admission] = None
    load: Optional[Load] = None

    # Optional explicit events sequence. If empty AND load is set, the
    # runner generates a default sequence from load.rungs.
    events: list[EventSpec] = field(default_factory=list)

    reports: list[ReportSpec] = field(default_factory=list)
    checks: list[CheckSpec] = field(default_factory=list)

    # Documentation. Keys are metric names (e.g. "fe_memory_growth_mb_per_min").
    expectations: dict[str, ExpectedRange] = field(default_factory=dict)

    # Set by the loader — the on-disk path the scenario came from.
    path: Optional[str] = None
