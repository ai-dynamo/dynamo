# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Types shared by the live retune modules.

Every module in ``dynamo.planner.retune.live`` depends on this one and on
nothing else in the package except through the protocols defined here, so
each can be built and tested with fakes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

LABEL_PREFIX = "nvidia.com/planner-retune-"
LABEL_DGD = LABEL_PREFIX + "dgd"
LABEL_DGD_UID = LABEL_PREFIX + "dgd-uid"
LABEL_WINDOW = LABEL_PREFIX + "window"


# ------------------------------------------------------------ kubernetes


class KubeError(Exception):
    """A Kubernetes call failed (network, timeout, 5xx, unexpected status)."""


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

    def get_dgd(self, name: str, *, timeout_s: float = 10.0) -> dict:
        ...

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

    def get_lease(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        ...

    def create_lease(self, body: dict, *, timeout_s: float = 10.0) -> dict:
        ...

    def replace_lease(self, name: str, body: dict, *, timeout_s: float = 10.0) -> dict:
        """Update with ``body['metadata']['resourceVersion']``; 409 -> KubeConflict."""
        ...

    def delete_lease(self, name: str, *, timeout_s: float = 10.0) -> None:
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
