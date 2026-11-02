# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Types shared by the live retune modules.

Every module in ``dynamo.planner.retune.live`` depends on this one and on
nothing else in the package except through the protocols defined here, so
each can be built and tested with fakes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

# Router knobs a DGDR v1beta2 Request can search, AISimulate public name ->
# live name. ``dynamo.router.simulation.config.ROUTER_PUBLIC_TO_LIVE`` is the
# source of truth; this mirrors it for modules that must not import the
# router simulation stack.
DGDR_ROUTER_KNOBS: dict[str, str] = {
    "overlap_score_credit": "overlap_score_credit",
    "prefill_load_scale": "prefill_load_scale",
    "temperature": "router_temperature",
}

LABEL_PREFIX = "nvidia.com/planner-retune-"
LABEL_DGD = LABEL_PREFIX + "dgd"
LABEL_DGD_UID = LABEL_PREFIX + "dgd-uid"
LABEL_WINDOW = LABEL_PREFIX + "window"
LABEL_SHARD = LABEL_PREFIX + "shard"
LABEL_SPEC = LABEL_PREFIX + "spec"
LABEL_ATTEMPT = LABEL_PREFIX + "attempt"

# Pod labels the operator sets (deploy/operator/internal/consts/consts.go).
POD_LABEL_DGD_NAME = "nvidia.com/dynamo-graph-deployment-name"
POD_LABEL_COMPONENT_TYPE = "nvidia.com/dynamo-component-type"

DGDR_GROUP = "nvidia.com"
DGDR_VERSION = "v1beta2"
DGDR_REQUESTS = "dynamographdeploymentrequests"
DGDR_RUNS = "dynamographdeploymentruns"
DGDR_CANDIDATES = "dynamographdeploymentcandidates"
RUN_UID_LABEL = "nvidia.com/dgdr-run-uid"

# SweepResult.failure_reason values that mean "this process stopped working
# on the window", not "the window failed": no decision is recorded and the
# controller rewinds so the window can be relaunched or adopted by the next
# leader.
NON_DECISION_FAILURES = frozenset({"shutdown", "lost_leadership"})


# ------------------------------------------------------------ kubernetes


class KubeError(Exception):
    """A Kubernetes call failed (network, timeout, 5xx, unexpected status)."""


class KubeNotFound(KubeError):
    pass


class KubeConflict(KubeError):
    """409 on update (stale resourceVersion)."""


class KubeAlreadyExists(KubeError):
    """409 on create (the name is taken)."""


class KubeInvalid(KubeError):
    """422 / 400: the API server rejected the object (e.g. Strict pruning)."""


@dataclass(frozen=True)
class DgdRef:
    """The DynamoGraphDeployment this planner serves."""

    name: str
    namespace: str  # Kubernetes namespace
    uid: str
    dynamo_namespace: str  # the routers' ``namespace`` on GET /router_policy


@dataclass(frozen=True)
class PodInfo:
    name: str
    uid: str
    ip: Optional[str]
    ready: bool
    terminating: bool
    labels: Mapping[str, str] = field(default_factory=dict)
    # Container port named ``http`` of the frontend container, if any.
    http_port: Optional[int] = None
    # ``status.phase`` (Pending, Running, Succeeded, Failed, Unknown).
    phase: Optional[str] = None


class RetuneKube(Protocol):
    """Synchronous Kubernetes access for live retune.

    Every method takes an optional ``timeout_s`` (connect+read, applied as
    the kubernetes client's ``_request_timeout`` tuple of ints) and raises
    the ``Kube*`` errors above instead of client exceptions. Callers on the
    event loop wrap calls in ``asyncio.to_thread``.
    """

    namespace: str

    def list_pods(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[PodInfo]:
        ...

    def create_request(
        self,
        body: dict,
        *,
        dry_run: bool = False,
        timeout_s: float = 10.0,
    ) -> dict:
        """Create with ``field_validation='Strict'`` (and ``dry_run='All'``)."""
        ...

    def get_request(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        ...

    def list_requests(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[dict]:
        ...

    def delete_request(
        self,
        name: str,
        *,
        propagation: Literal["Background", "Foreground"] = "Background",
        timeout_s: float = 10.0,
    ) -> None:
        """Missing objects are not an error."""
        ...

    def get_run(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        ...

    def list_candidates(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[dict]:
        ...


# ------------------------------------------------------- frontend config


@dataclass(frozen=True)
class FrontendStartup:
    """The frontend router configuration parsed from the DGD spec.

    ``router_config`` is the startup ``KvRouterConfig`` JSON (only keys the
    frontend sets); ``startup_maps`` is ``policy_parameters(wt)`` for each
    worker type of the planner mode. ``fingerprint_source`` is a canonical
    JSON string of everything parsed that the state fingerprint covers.
    """

    router_mode: str  # "kv_router" or "round_robin"
    router_config: dict[str, Any]
    startup_maps: dict[str, dict[str, float]]
    frontend_component: str  # DGD component name of the frontend
    replicas: int
    replica_sync: bool
    fingerprint_source: str


@dataclass(frozen=True)
class PremiseReport:
    """Spec-class premise evaluation (P4, P7-P10; P14 is the actuator's).

    ``p4_ok`` alone decides whether a controller can be built; ``spec_ok``
    is the conjunction of every spec premise. ``reasons`` explain each
    failed premise as ``"P<n>: ..."``; ``warnings`` carry warn-only ones.
    """

    p4_ok: bool
    spec_ok: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    startup: Optional[FrontendStartup] = None


# ----------------------------------------------------------- reconciler


class SweepRegistry(Protocol):
    """Thread-safe link between a running sweep and the reconciler.

    Protection is reference-counted per window: every ``protect`` call
    takes one reference and needs one matching ``release``. A window, with
    the union of every Request name given to it, stays protected while any
    reference is held, so two sweeps of one window (a rebuilt controller
    relaunching it while the old sweep winds down) cannot drop each
    other's protection.
    """

    def protect(self, window_end_ms: int, request_names: Iterable[str]) -> None:
        """Mark a window file and Requests as in use by this process (one
        reference)."""
        ...

    def release(self, window_end_ms: int) -> None:
        """Drop one reference taken by ``protect``."""
        ...

    def enqueue_delete(self, request_names: Iterable[str]) -> None:
        ...


# ---------------------------------------------------------------- window


@dataclass(frozen=True)
class WindowCutResult:
    path: Optional[str]  # None for an empty window (no file written)
    num_requests: int
    trace_block_size: Optional[int]
    row_counts: dict[str, int]
    dropped_members: int = 0
    dropped_bytes: int = 0
    reused: bool = False  # the final file already existed (EEXIST)


class SupportsIsSet(Protocol):
    """A cancellation flag: a ``threading.Event`` or any object with ``is_set()``."""

    def is_set(self) -> bool:
        ...


class WindowCutter(Protocol):
    def __call__(
        self,
        segments_root: str,
        start_ms: int,
        end_ms: int,
        as_of_ms: int,
        out_path: str,
        *,
        max_rows: int,
        deadline_s: float,
        cancel: Optional[SupportsIsSet] = None,
    ) -> WindowCutResult:
        ...


# --------------------------------------------------------------- metrics


class RetuneMetricsSink(Protocol):
    """``dynamo_planner_retune_*`` metrics; every method is cheap and safe."""

    def decision(self, outcome: str) -> None:
        ...

    def failure(self, reason: str) -> None:
        ...

    def skipped(self, reason: str) -> None:
        ...

    def supervisor_state(
        self,
        *,
        leader: bool,
        ready: bool,
        initialized: bool,
        rollback_pending: bool,
        generation: int,
    ) -> None:
        ...

    def scores(
        self,
        *,
        baseline: Optional[float],
        best: Optional[float],
        sweep_seconds: Optional[float],
        missing_candidates: int,
    ) -> None:
        ...

    def router_converged(self, converged: bool) -> None:
        ...

    def patch_error(self, code: str) -> None:
        ...

    def trace_sink_missing(self, pod: str, missing: bool) -> None:
        ...

    def dropped_members(self, count: int) -> None:
        ...
