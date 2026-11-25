# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Types shared by the live retune modules.

Every module in ``dynamo.planner.retune.live`` depends on this one and on
nothing else in the package except through the protocols defined here, so
each can be built and tested with fakes.
"""

from __future__ import annotations

import math
import posixpath
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

# The eight builtin default-policy parameters a frontend router exposes on
# GET/PATCH /router_policy, by their live (PATCH) names.
POLICY_PARAMETER_KEYS: tuple[str, ...] = (
    "decode_active_request_weight",
    "disk_cache_hit_weight",
    "host_cache_hit_weight",
    "overlap_score_credit",
    "overlap_score_credit_decay",
    "prefill_load_scale",
    "router_temperature",
    "shared_cache_multiplier",
)

# Router knobs a DGDR v1beta2 Request can search, AISimulate public name ->
# live name. ``dynamo.router.simulation.config.ROUTER_PUBLIC_TO_LIVE`` is the
# source of truth; this mirrors it for modules that must not import the
# router simulation stack.
DGDR_ROUTER_KNOBS: dict[str, str] = {
    "overlap_score_credit": "overlap_score_credit",
    "prefill_load_scale": "prefill_load_scale",
    "temperature": "router_temperature",
}

# Immutable binding rendered into the Planner Pod by the operator.
ENV_RETUNE_CAPABILITY = "DYN_PLANNER_RETUNE_CAPABILITY"

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

GateReason = Literal["not_leader", "not_ready", "already_decided"]


# --------------------------------------------------------------- controller


@dataclass(frozen=True)
class GateResult:
    """What ``launch_gate`` tells the controller about a due window.

    ``reason`` is ``None`` to let the window launch (or record ``busy``).
    ``not_leader``/``not_ready`` keep ``_last_k`` so the window is retried on
    a later tick; ``already_decided`` advances it without a decision.
    ``spec_replicas`` (worker type -> DGD spec replicas) and ``abort`` (set
    by the supervisor when this process stops leading) are only meaningful
    when ``reason`` is ``None``.
    """

    reason: Optional[GateReason] = None
    spec_replicas: Optional[dict[str, int]] = None
    abort: Optional[threading.Event] = None


# (k, (window_start_s, window_end_s, as_of_s)) -> GateResult
LaunchGate = Callable[[int, tuple[float, float, float]], GateResult]
# controller generation of the sweep's controller -> may it be harvested?
HarvestAllowed = Callable[[int], bool]


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
class RetuneCapability:
    """Immutable operator launch contract, checkpointed for router cleanup."""

    frontend_component_name: str
    claim_name: str
    mount_path: str
    segments_dir: str
    windows_dir: str
    admin_port: int
    config_hash: str
    profile: str
    overlap_score_credit: float
    prefill_load_scale: float
    temperature: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "configHash": self.config_hash,
            "frontendComponentName": self.frontend_component_name,
            "traceStorage": {
                "claimName": self.claim_name,
                "mountPath": self.mount_path,
                "segmentsDir": self.segments_dir,
                "windowsDir": self.windows_dir,
            },
            "adminPort": self.admin_port,
            "router": {
                "profile": self.profile,
                "overlapScoreCredit": self.overlap_score_credit,
                "prefillLoadScale": self.prefill_load_scale,
                "temperature": self.temperature,
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> RetuneCapability:
        if not isinstance(value, Mapping):
            raise ValueError("the Retune capability is unavailable")
        if type(value.get("version")) is not int or value["version"] != 1:
            raise ValueError("unsupported Retune launch contract version")
        config_hash = value.get("configHash")
        if not isinstance(config_hash, str) or not config_hash:
            raise ValueError("Retune configHash must be a non-empty string")
        router = value.get("router")
        if not isinstance(router, Mapping) or router.get("profile") != "kv-v1":
            raise ValueError("unsupported Retune router profile")
        parameters = []
        for name in ("overlapScoreCredit", "prefillLoadScale", "temperature"):
            parameter = router.get(name)
            if isinstance(parameter, bool) or not isinstance(parameter, (int, float)):
                raise ValueError(
                    f"Retune router.{name} must be finite and non-negative"
                )
            try:
                number = float(parameter)
            except OverflowError as error:
                raise ValueError(
                    f"Retune router.{name} must be finite and non-negative"
                ) from error
            if not math.isfinite(number) or number < 0:
                raise ValueError(
                    f"Retune router.{name} must be finite and non-negative"
                )
            parameters.append(number)
        storage = value.get("traceStorage")
        if not isinstance(storage, Mapping):
            raise ValueError("Retune traceStorage is unavailable")
        frontend = value.get("frontendComponentName")
        claim = storage.get("claimName")
        mount = storage.get("mountPath")
        segments = storage.get("segmentsDir")
        windows = storage.get("windowsDir")
        if (
            not isinstance(frontend, str)
            or not frontend
            or not isinstance(claim, str)
            or not claim
            or not isinstance(mount, str)
            or not mount
            or not isinstance(segments, str)
            or not segments
            or not isinstance(windows, str)
            or not windows
        ):
            raise ValueError(
                "Retune capability names and paths must be non-empty strings"
            )
        if not mount.startswith("/") or posixpath.normpath(mount) != mount:
            raise ValueError("Retune mountPath must be an absolute normalized path")
        for name, path in (("segmentsDir", segments), ("windowsDir", windows)):
            if path.startswith("/") or any(
                part in ("", ".", "..") for part in path.split("/")
            ):
                raise ValueError(
                    f"Retune {name} must be a normalized claim-relative directory"
                )
        if (
            segments == windows
            or segments.startswith(windows + "/")
            or windows.startswith(segments + "/")
        ):
            raise ValueError("Retune trace directories must not overlap")
        port = value.get("adminPort")
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise ValueError("Retune adminPort must be between 1 and 65535")
        return cls(
            frontend,
            claim,
            mount,
            segments,
            windows,
            port,
            config_hash,
            "kv-v1",
            *parameters,
        )


@dataclass(frozen=True)
class PodInfo:
    name: str
    uid: str
    ip: Optional[str]
    ready: bool
    terminating: bool
    labels: Mapping[str, str] = field(default_factory=dict)
    # ``status.phase`` (Pending, Running, Succeeded, Failed, Unknown).
    phase: Optional[str] = None
    # ``restartCount`` of the ``main`` container, if reported. A restarted
    # frontend registers new routers (new router ids).
    restarts: Optional[int] = None


# A frontend process is replaced by a new Pod UID or main-container restart.
PodInstance = tuple[str, Optional[int]]


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

    def delete_lease(
        self,
        name: str,
        *,
        resource_version: Optional[str] = None,
        timeout_s: float = 10.0,
    ) -> None:
        """Missing Leases are not an error. With ``resource_version`` the
        delete is conditional on it; a changed Lease -> KubeConflict."""
        ...


# ------------------------------------------------------- frontend config


@dataclass(frozen=True)
class FrontendStartup:
    """The frontend startup configuration resolved from the launch profile.

    ``router_config`` is the complete startup ``KvRouterConfig`` JSON;
    ``startup_maps`` is ``policy_parameters(wt)`` for each
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


# ---------------------------------------------------------- router policy


@dataclass(frozen=True)
class RouterInfo:
    """One router on GET /router_policy."""

    router_id: str
    namespace: str
    worker_type: str
    revision: int
    # The 8 PolicyParameters; None for a custom policy without runtime config.
    configuration: Optional[dict[str, float]]
    model: Optional[str] = None


@dataclass(frozen=True)
class PolicyCallResult:
    """Outcome of one GET or PATCH. ``status`` is the HTTP status (0 on a
    network error or timeout)."""

    status: int
    routers: tuple[RouterInfo, ...] = ()  # GET
    router: Optional[RouterInfo] = None  # PATCH/reset response, when 200
    error: Optional[str] = None
    retune_config_hash: Optional[str] = None  # Validated runtime contract (GET).


class RouterPolicyClient(Protocol):
    async def get(self, pod: PodInfo) -> PolicyCallResult:
        ...

    async def patch(
        self,
        pod: PodInfo,
        router_id: str,
        expected_revision: int,
        parameters: Mapping[str, float],
    ) -> PolicyCallResult:
        ...

    async def reset(
        self, pod: PodInfo, router_id: str, expected_revision: int
    ) -> PolicyCallResult:
        """PATCH ``{router_id, expected_revision, reset: true}``."""
        ...


ActuationMode = Literal["apply", "rollback", "observe"]


@dataclass(frozen=True)
class ActuationTarget:
    """What one actuator round converges to.

    ``apply``: PATCH explicit values so every router's 8 parameters equal
    ``desired_by_wt[worker_type]``. ``rollback``: reset every router whose
    configuration differs from its latched initial (never by revision).
    ``observe``: no writes; latch initials and check parity only.
    ``include_not_ready`` also targets NotReady, non-terminating pods.
    ``learn_unknown`` lets the actuator reset a router whose initial was
    never latched (revision > 0) once to learn it; that is active mode only.
    With False (advisory) such a router is never written: rollback
    leaves it alone and does not wait for it to converge.
    """

    mode: ActuationMode
    desired_by_wt: Optional[dict[str, dict[str, float]]] = None
    include_not_ready: bool = False
    learn_unknown: bool = True


@dataclass(frozen=True)
class ActuationReport:
    converged: bool
    # router_id -> configuration latched this round (revision 0 or reset).
    new_initials: dict[str, dict[str, float]] = field(default_factory=dict)
    # router_ids seen this round, or known and not provably gone; the
    # initials of every other router are pruned.
    seen_router_ids: frozenset[str] = frozenset()
    # P14: every known initial equals startup_maps[worker_type].
    parity_ok: Optional[bool] = None
    parity_mismatches: tuple[str, ...] = ()
    admin_api_disabled: bool = False
    errors: dict[str, int] = field(default_factory=dict)  # by status code / kind
    # P12: enough Ready frontends, each with a configured router for every
    # worker type, whatever their values. Gates launches; convergence does not.
    routers_present: Optional[bool] = None
    router_hosts: dict[str, tuple[PodInstance, ...]] = field(default_factory=dict)


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


class NullMetrics:
    """A ``RetuneMetricsSink`` that records nothing (tests, disabled metrics)."""

    def decision(self, outcome: str) -> None:
        pass

    def failure(self, reason: str) -> None:
        pass

    def skipped(self, reason: str) -> None:
        pass

    def supervisor_state(self, **_: Any) -> None:
        pass

    def scores(self, **_: Any) -> None:
        pass

    def router_converged(self, converged: bool) -> None:
        pass

    def patch_error(self, code: str) -> None:
        pass

    def trace_sink_missing(self, pod: str, missing: bool) -> None:
        pass

    def dropped_members(self, count: int) -> None:
        pass
