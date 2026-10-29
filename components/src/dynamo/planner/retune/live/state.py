# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persisted live retune state and the leadership Lease (design §4.7.1-§4.7.2).

Two ``coordination.k8s.io/v1`` Leases per DGD, both owned by the DGD so the
garbage collector removes them with it:

- ``<dgd>-planner-retune``: leadership. ``LeaseLeadership`` renews it with
  get -> check holder -> update(resourceVersion), takes over an empty holder
  at once and a foreign holder only after observing it unchanged (holder and
  resourceVersion) for a whole lease duration on the local monotonic clock.
- ``<dgd>-planner-retune-state``: the state snapshot (``RetuneState``) as
  JSON in two annotations. Every write replaces the snapshot and is
  conditional on the resourceVersion read; the supervisor additionally
  requires ``state.leader_term`` to equal its own term for every write but
  the claim.

The write-rule functions below return a new ``RetuneState`` and never touch
Kubernetes; ``StateStore`` reads and writes the snapshot. Everything here is
synchronous: callers on the event loop run it in a thread.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Optional

from dynamo.planner.retune.live.contracts import (
    LABEL_DGD,
    LABEL_DGD_UID,
    DgdRef,
    FrontendStartup,
    KubeAlreadyExists,
    KubeConflict,
    KubeError,
    RetuneKube,
)

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig

logger = logging.getLogger(__name__)

STATE_ANNOTATION = "nvidia.com/planner-retune-state"
HISTORY_ANNOTATION = "nvidia.com/planner-retune-history"
HISTORY_LIMIT = 50
STATE_FORMAT_VERSION = 1

DGD_API_VERSION = "nvidia.com/v1beta1"
DGD_KIND = "DynamoGraphDeployment"
LEASE_API_VERSION = "coordination.k8s.io/v1"

RetuneMode = Literal["active", "advisory"]
_MODES = ("active", "advisory")
_REL_TOL = 1e-9
_ABS_TOL = 1e-12


def leadership_lease_name(dgd_name: str) -> str:
    return f"{dgd_name}-planner-retune"


def state_lease_name(dgd_name: str) -> str:
    return f"{dgd_name}-planner-retune-state"


def owner_reference(dgd: DgdRef) -> dict[str, Any]:
    """ownerReference to the DGD; the GC deletes the Leases with it."""

    return {
        "apiVersion": DGD_API_VERSION,
        "kind": DGD_KIND,
        "name": dgd.name,
        "uid": dgd.uid,
        "blockOwnerDeletion": False,
    }


def leader_term(lease_transitions: int, identity: str, term_seq: int) -> str:
    """``<leaseTransitions>/<holderIdentity>/<term_seq>``.

    ``term_seq`` is bumped on every entry into initialization, so a write of
    an earlier term (another process, or this one before re-initializing)
    always fails the ``leader_term`` precondition.
    """

    return f"{int(lease_transitions)}/{identity}/{int(term_seq)}"


def micro_time(epoch_s: float) -> str:
    """A Kubernetes ``MicroTime`` (RFC 3339 with microseconds, UTC)."""

    stamp = datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def maps_equal(
    left: Optional[Mapping[str, Any]], right: Optional[Mapping[str, Any]]
) -> bool:
    """Deep equality of parameter maps, floats within a tiny tolerance."""

    if left is None or right is None:
        return left is None and right is None
    if set(left) != set(right):
        return False
    for key, value in left.items():
        other = right[key]
        if isinstance(value, Mapping) or isinstance(other, Mapping):
            if not (
                isinstance(value, Mapping)
                and isinstance(other, Mapping)
                and maps_equal(value, other)
            ):
                return False
        elif isinstance(value, (int, float)) and isinstance(other, (int, float)):
            if not math.isclose(
                float(value), float(other), rel_tol=_REL_TOL, abs_tol=_ABS_TOL
            ):
                return False
        elif value != other:
            return False
    return True


def fingerprint(config: PlannerConfig, startup: Optional[FrontendStartup]) -> str:
    """What the deployed point was chosen under; a change resets it.

    Covers the router search domain as configured (its expansion is a pure
    function of it and the startup), the parsed frontend startup, the pinned
    DGDR engine section with its model, SKU and backend, the planner mode,
    the objective and the scoring SLA. ``advisory`` is not part of it: the
    state's ``mode`` records it.
    """

    retune = config.retune
    if retune is None:
        raise ValueError("fingerprint requires config.retune")
    dgdr = retune.dgdr
    sla_ttft_ms, sla_itl_ms = retune.scoring_sla(config.ttft_ms, config.itl_ms)
    payload = {
        "version": STATE_FORMAT_VERSION,
        "router": retune.router,
        "planner": retune.planner,
        "objective": retune.objective,
        "sla": [float(sla_ttft_ms), float(sla_itl_ms)],
        "mode": config.mode,
        "backend": config.backend,
        "engine": None if dgdr is None else dgdr.engine,
        "model": None if dgdr is None else dgdr.model,
        "sku": None if dgdr is None else dgdr.sku,
        "startup": None if startup is None else startup.fingerprint_source,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:16]


# ------------------------------------------------------------------ state


class StateFormatError(ValueError):
    """The state annotations do not hold a readable snapshot."""


def _float_map(value: Any, what: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise StateFormatError(f"{what} must be an object")
    out: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise StateFormatError(f"{what}.{key} must be a number")
        out[str(key)] = float(item)
    return out


def _nested_map(value: Any, what: str) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        raise StateFormatError(f"{what} must be an object")
    return {str(key): _float_map(item, f"{what}.{key}") for key, item in value.items()}


@dataclass(frozen=True)
class RetuneState:
    """The state snapshot of design §4.7.2.

    ``mode is None`` is "no state": only the claim (``dgd_uid`` and
    ``leader_term``) has been written. ``by_wt`` is the deployed per worker
    type 8-parameter map, ``flat`` the controller's router point (live
    names), ``initials`` the latched initial configuration of each router.
    ``history`` is newest last, at most ``HISTORY_LIMIT`` entries.
    """

    dgd_uid: str
    leader_term: str = ""
    fingerprint: Optional[str] = None
    mode: Optional[RetuneMode] = None
    rollback_pending: bool = False
    deploy_seq: int = 0
    flat: dict[str, float] = field(default_factory=dict)
    by_wt: dict[str, dict[str, float]] = field(default_factory=dict)
    initials: dict[str, dict[str, float]] = field(default_factory=dict)
    last_window_end_ms: Optional[int] = None
    updated_ms: int = 0
    history: tuple[dict[str, Any], ...] = ()

    @property
    def has_deployment(self) -> bool:
        """Whether an initialization has recorded what is deployed."""

        return self.mode is not None

    def to_annotations(self) -> dict[str, str]:
        body = {
            "version": STATE_FORMAT_VERSION,
            "dgd_uid": self.dgd_uid,
            "leader_term": self.leader_term,
            "fingerprint": self.fingerprint,
            "mode": self.mode,
            "rollback_pending": self.rollback_pending,
            "deploy_seq": self.deploy_seq,
            "flat": self.flat,
            "by_wt": self.by_wt,
            "initials": self.initials,
            "last_window_end_ms": self.last_window_end_ms,
            "updated_ms": self.updated_ms,
        }
        return {
            STATE_ANNOTATION: canonical_json(body),
            HISTORY_ANNOTATION: canonical_json(list(self.history)),
        }

    @classmethod
    def from_annotations(
        cls, annotations: Optional[Mapping[str, str]]
    ) -> Optional[RetuneState]:
        """Parse the snapshot; ``None`` when the lease holds none.

        Raises ``StateFormatError`` for a snapshot that cannot be read.
        """

        raw = (annotations or {}).get(STATE_ANNOTATION)
        if raw is None:
            return None
        try:
            body = json.loads(raw)
        except ValueError as error:
            raise StateFormatError(f"state annotation is not JSON: {error}") from None
        if not isinstance(body, Mapping):
            raise StateFormatError("state annotation must be a JSON object")
        dgd_uid = body.get("dgd_uid")
        if not isinstance(dgd_uid, str) or not dgd_uid:
            raise StateFormatError("state.dgd_uid must be a non-empty string")
        mode = body.get("mode")
        if mode is not None and mode not in _MODES:
            raise StateFormatError(f"state.mode must be one of {_MODES} or null")
        deploy_seq = body.get("deploy_seq", 0)
        if isinstance(deploy_seq, bool) or not isinstance(deploy_seq, int):
            raise StateFormatError("state.deploy_seq must be an integer")
        last = body.get("last_window_end_ms")
        if last is not None and (
            isinstance(last, bool) or not isinstance(last, (int, float))
        ):
            raise StateFormatError("state.last_window_end_ms must be a number")
        history_raw = (annotations or {}).get(HISTORY_ANNOTATION)
        history: list[dict[str, Any]] = []
        if history_raw:
            try:
                parsed = json.loads(history_raw)
            except ValueError:
                # History is an audit trail; a damaged one never blocks.
                logger.warning("retune state: unreadable history annotation dropped")
                parsed = []
            if isinstance(parsed, list):
                history = [dict(item) for item in parsed if isinstance(item, Mapping)]
        return cls(
            dgd_uid=dgd_uid,
            leader_term=str(body.get("leader_term") or ""),
            fingerprint=body.get("fingerprint"),
            mode=mode,
            rollback_pending=bool(body.get("rollback_pending", False)),
            deploy_seq=int(deploy_seq),
            flat=_float_map(body.get("flat") or {}, "state.flat"),
            by_wt=_nested_map(body.get("by_wt") or {}, "state.by_wt"),
            initials=_nested_map(body.get("initials") or {}, "state.initials"),
            last_window_end_ms=None if last is None else int(last),
            updated_ms=int(body.get("updated_ms") or 0),
            history=tuple(history[-HISTORY_LIMIT:]),
        )


def deployed_off(
    state: Optional[RetuneState],
    startup_by_wt: Optional[Mapping[str, Mapping[str, float]]],
) -> bool:
    """Whether the routers may be off their startup values (§4.7.3 step 4).

    ``rollback_pending ∨ by_wt ≠ startup``, false without a state, with one
    refinement: only an ``active`` record can have deployed anything. An
    ``advisory`` record without ``rollback_pending`` was written by a planner
    that only observed (or finished rolling back), so its ``by_wt`` is the
    startup it saw, not a deployment; after a frontend startup change it
    must not make an advisory planner reset routers it never wrote. With an
    unknown startup (the frontend could not be parsed) an active record
    counts as deployed: only the routers' latched initials can tell.
    """

    if state is None or not state.has_deployment:
        return False
    if state.rollback_pending:
        return True
    if state.mode != "active":
        return False
    if startup_by_wt is None:
        return True
    return not maps_equal(state.by_wt, startup_by_wt)


# -------------------------------------------------------- write rules (§4.7.2)


def claim(state: Optional[RetuneState], *, dgd_uid: str, term: str) -> RetuneState:
    """Claim write: only ``leader_term`` changes (a new record without state)."""

    if state is None:
        return RetuneState(dgd_uid=dgd_uid, leader_term=term)
    return dataclasses.replace(state, leader_term=term)


def _replace_deployment(
    state: RetuneState,
    *,
    fingerprint: str,
    mode: RetuneMode,
    flat: Mapping[str, float],
    by_wt: Mapping[str, Mapping[str, float]],
    rollback_pending: bool,
) -> RetuneState:
    return dataclasses.replace(
        state,
        fingerprint=fingerprint,
        mode=mode,
        flat={key: float(value) for key, value in flat.items()},
        by_wt={wt: dict(params) for wt, params in by_wt.items()},
        rollback_pending=bool(rollback_pending),
        deploy_seq=state.deploy_seq + 1,
    )


def deploy(
    state: RetuneState,
    *,
    fingerprint: str,
    mode: RetuneMode,
    flat: Mapping[str, float],
    by_wt: Mapping[str, Mapping[str, float]],
) -> RetuneState:
    """applied / promote: replace the deployment, clear rollback_pending."""

    return _replace_deployment(
        state,
        fingerprint=fingerprint,
        mode=mode,
        flat=flat,
        by_wt=by_wt,
        rollback_pending=False,
    )


def rollback_start(
    state: RetuneState,
    *,
    fingerprint: str,
    mode: RetuneMode,
    flat: Mapping[str, float],
    by_wt: Mapping[str, Mapping[str, float]],
) -> RetuneState:
    """Rollback start (advisory, or spec_ok false): startup, pending set."""

    return _replace_deployment(
        state,
        fingerprint=fingerprint,
        mode=mode,
        flat=flat,
        by_wt=by_wt,
        rollback_pending=True,
    )


def reset(
    state: RetuneState,
    *,
    fingerprint: str,
    mode: RetuneMode,
    flat: Mapping[str, float],
    by_wt: Mapping[str, Mapping[str, float]],
    rollback_pending: bool,
) -> RetuneState:
    """Reset (fingerprint change, no state, mode switch): startup,
    ``rollback_pending = deployed_off``."""

    return _replace_deployment(
        state,
        fingerprint=fingerprint,
        mode=mode,
        flat=flat,
        by_wt=by_wt,
        rollback_pending=rollback_pending,
    )


def rollback_complete(
    state: RetuneState,
    *,
    expected_deploy_seq: int,
    entry: Optional[Mapping[str, Any]] = None,
) -> Optional[RetuneState]:
    """Flip only ``rollback_pending`` to false.

    ``None`` (discard, never retry) when the deployment changed since the
    rollback started (``deploy_seq``) or the rollback is already complete.
    ``entry`` (a ``rolled_back`` record) is appended to the history.
    """

    if not state.rollback_pending or state.deploy_seq != expected_deploy_seq:
        return None
    new = dataclasses.replace(state, rollback_pending=False)
    if entry is not None:
        new = append_history(new, entry)
    return new


def merge_initials(
    state: RetuneState,
    latched: Mapping[str, Mapping[str, float]],
    *,
    keep: Optional[Iterable[str]] = None,
) -> RetuneState:
    """Merge newly latched initials; drop router ids not in ``keep``."""

    initials = {rid: dict(cfg) for rid, cfg in state.initials.items()}
    for router_id, configuration in latched.items():
        initials[str(router_id)] = {k: float(v) for k, v in configuration.items()}
    if keep is not None:
        kept = set(keep)
        initials = {rid: cfg for rid, cfg in initials.items() if rid in kept}
    return dataclasses.replace(state, initials=initials)


def append_history(state: RetuneState, entry: Mapping[str, Any]) -> RetuneState:
    """Append one entry, deduplicated by ``decision_id``; keep the newest 50."""

    decision_id = entry.get("decision_id")
    history = [
        item
        for item in state.history
        if decision_id is None or item.get("decision_id") != decision_id
    ]
    history.append(dict(entry))
    return dataclasses.replace(state, history=tuple(history[-HISTORY_LIMIT:]))


def has_decision(state: RetuneState, decision_id: str) -> bool:
    return any(item.get("decision_id") == decision_id for item in state.history)


def record_decision(
    state: RetuneState, entry: Mapping[str, Any], window_end_ms: Optional[int]
) -> RetuneState:
    """Every decision: history entry, ``last_window_end_ms = max(old, end)``."""

    new = append_history(state, entry)
    if window_end_ms is None:
        return new
    last = state.last_window_end_ms
    return dataclasses.replace(
        new,
        last_window_end_ms=int(window_end_ms)
        if last is None
        else max(last, int(window_end_ms)),
    )


# ----------------------------------------------------------------- store


@dataclass(frozen=True)
class StateSnapshot:
    """One read of the state Lease. ``lease is None``: it does not exist."""

    state: Optional[RetuneState]
    resource_version: Optional[str]
    lease: Optional[dict[str, Any]]
    # The annotations held a snapshot that could not be parsed.
    corrupt: bool = False


class StateStore:
    """Reads and conditionally writes ``<dgd>-planner-retune-state``."""

    def __init__(
        self,
        kube: RetuneKube,
        dgd: DgdRef,
        *,
        timeout_s: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._kube = kube
        self._dgd = dgd
        self._timeout_s = float(timeout_s)
        self._clock = clock
        self.name = state_lease_name(dgd.name)

    @property
    def dgd(self) -> DgdRef:
        return self._dgd

    def read(self) -> StateSnapshot:
        lease = self._kube.get_lease(self.name, timeout_s=self._timeout_s)
        if lease is None:
            return StateSnapshot(state=None, resource_version=None, lease=None)
        metadata = lease.get("metadata") or {}
        rv = metadata.get("resourceVersion")
        try:
            state = RetuneState.from_annotations(metadata.get("annotations"))
        except StateFormatError as error:
            logger.error("retune state lease %s is unreadable: %s", self.name, error)
            return StateSnapshot(
                state=None, resource_version=rv, lease=lease, corrupt=True
            )
        return StateSnapshot(state=state, resource_version=rv, lease=lease)

    def write(self, state: RetuneState, snapshot: StateSnapshot) -> StateSnapshot:
        """Replace the snapshot read as ``snapshot`` (create it if absent).

        A changed resourceVersion, or a lease created by someone else in the
        meantime, raises ``KubeConflict``; other failures raise ``KubeError``
        (the write may still have landed: re-read before deciding).
        """

        state = dataclasses.replace(state, updated_ms=int(round(self._clock() * 1000)))
        annotations = state.to_annotations()
        if snapshot.lease is None:
            body = self._new_body(annotations)
            try:
                created = self._kube.create_lease(body, timeout_s=self._timeout_s)
            except KubeAlreadyExists as error:
                raise KubeConflict(str(error)) from error
            return StateSnapshot(
                state=state,
                resource_version=(created.get("metadata") or {}).get("resourceVersion"),
                lease=created,
            )
        body = json.loads(json.dumps(snapshot.lease))
        metadata = body.setdefault("metadata", {})
        metadata["resourceVersion"] = snapshot.resource_version
        merged = dict(metadata.get("annotations") or {})
        merged.update(annotations)
        metadata["annotations"] = merged
        metadata["ownerReferences"] = [owner_reference(self._dgd)]
        labels = dict(metadata.get("labels") or {})
        labels.update({LABEL_DGD: self._dgd.name, LABEL_DGD_UID: self._dgd.uid})
        metadata["labels"] = labels
        replaced = self._kube.replace_lease(self.name, body, timeout_s=self._timeout_s)
        return StateSnapshot(
            state=state,
            resource_version=(replaced.get("metadata") or {}).get("resourceVersion"),
            lease=replaced,
        )

    def delete(self) -> None:
        self._kube.delete_lease(self.name, timeout_s=self._timeout_s)

    def _new_body(self, annotations: Mapping[str, str]) -> dict[str, Any]:
        return {
            "apiVersion": LEASE_API_VERSION,
            "kind": "Lease",
            "metadata": {
                "name": self.name,
                "namespace": self._kube.namespace,
                "labels": {LABEL_DGD: self._dgd.name, LABEL_DGD_UID: self._dgd.uid},
                "annotations": dict(annotations),
                "ownerReferences": [owner_reference(self._dgd)],
            },
            "spec": {},
        }


# ------------------------------------------------------------ leadership


LeaseStep = Literal["acquired", "renewed", "standby", "lost", "error"]


class LeaseLeadership:
    """Leadership over ``<dgd>-planner-retune`` (design §4.7.1).

    ``step`` runs one renewal/acquisition attempt and returns what happened:

    - ``acquired``: this process now leads (a new lease, an empty holder, an
      expired foreign holder, or its own lease after it had lost track).
    - ``renewed``: it led and still does.
    - ``lost``: it led and another holder now owns the lease.
    - ``standby``: another holder leads.
    - ``error``: the API call failed; leadership, if any, lasts until
      ``leader_until``.

    ``is_leader`` holds from a successful write until ``leader_until`` =
    send time + lease duration - ``margin_s``, all on the local monotonic
    clock: a contender takes over only after watching the holder unchanged
    for a whole lease duration, which starts after the send.
    """

    def __init__(
        self,
        kube: RetuneKube,
        dgd: DgdRef,
        *,
        identity: str,
        lease_duration_s: float = 60.0,
        margin_s: Optional[float] = None,
        timeout_s: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not identity:
            raise ValueError("identity must be non-empty")
        if not (math.isfinite(lease_duration_s) and lease_duration_s > 0):
            raise ValueError("lease_duration_s must be positive and finite")
        self._kube = kube
        self._dgd = dgd
        self.identity = identity
        self.name = leadership_lease_name(dgd.name)
        self.lease_duration_s = float(lease_duration_s)
        self._duration_int = max(1, int(math.ceil(lease_duration_s)))
        if margin_s is None:
            margin_s = 3.0 * timeout_s
        # Never leave less than a third of the duration to lead in.
        self.margin_s = min(float(margin_s), self.lease_duration_s * 2.0 / 3.0)
        self._timeout_s = float(timeout_s)
        self._monotonic = monotonic
        self._clock = clock
        self._lock = threading.Lock()
        self._held = False
        self._leader_until = -math.inf
        self.transitions = 0
        # (holderIdentity, resourceVersion) of a foreign lease, and when this
        # process first saw it unchanged.
        self._observed: Optional[tuple[str, Optional[str]]] = None
        self._observed_since = 0.0

    # ------------------------------------------------------------- queries

    def is_leader(self) -> bool:
        with self._lock:
            return self._held and self._monotonic() < self._leader_until

    @property
    def held(self) -> bool:
        with self._lock:
            return self._held

    @property
    def leader_until(self) -> float:
        with self._lock:
            return self._leader_until

    def set_owner(self, dgd: DgdRef) -> None:
        """The DGD the Lease's ownerReference names from now on."""

        self._dgd = dgd

    def drop(self) -> None:
        """Forget leadership locally (no API call)."""

        with self._lock:
            self._held = False
            self._leader_until = -math.inf

    # ---------------------------------------------------------------- step

    def step(self) -> LeaseStep:
        t_send = self._monotonic()
        try:
            lease = self._kube.get_lease(self.name, timeout_s=self._timeout_s)
        except KubeError as error:
            logger.warning(
                "retune leadership: reading lease %s failed: %s", self.name, error
            )
            return "error"
        if lease is None:
            return self._create(t_send)

        spec = lease.get("spec") or {}
        metadata = lease.get("metadata") or {}
        holder = spec.get("holderIdentity") or ""
        rv = metadata.get("resourceVersion")
        transitions = _int(spec.get("leaseTransitions"))

        if holder == self.identity:
            # Expired locally counts as lost: renewing it starts a new term.
            was_held = self.is_leader()
            body = self._body(lease, transitions=transitions, acquire=not was_held)
            try:
                updated = self._kube.replace_lease(
                    self.name, body, timeout_s=self._timeout_s
                )
            except KubeConflict:
                # Changed since the read: re-read on the next step.
                logger.info("retune leadership: renewal of %s conflicted", self.name)
                return "error"
            except KubeError as error:
                logger.warning(
                    "retune leadership: renewing %s failed: %s", self.name, error
                )
                return "error"
            new_holder = (updated.get("spec") or {}).get("holderIdentity") or ""
            if new_holder != self.identity:
                return self._lose()
            self._lead(t_send, transitions)
            return "renewed" if was_held else "acquired"

        lost = self.held
        if lost:
            self.drop()
            logger.warning(
                "retune leadership: lease %s is now held by %r",
                self.name,
                holder or None,
            )
        if not holder:
            # Released by its holder: take it over at once.
            return self._take_over(lease, t_send, transitions, lost)
        observed = (holder, rv)
        if self._observed != observed:
            self._observed = observed
            self._observed_since = t_send
            return "lost" if lost else "standby"
        duration = _positive(spec.get("leaseDurationSeconds")) or self.lease_duration_s
        if t_send - self._observed_since < duration:
            return "lost" if lost else "standby"
        logger.info(
            "retune leadership: holder %r of %s unchanged for %.0fs; taking over",
            holder,
            self.name,
            t_send - self._observed_since,
        )
        return self._take_over(lease, t_send, transitions, lost)

    def release(self) -> bool:
        """Clear the holder if it is still this process (conditional update)."""

        held = self.held
        self.drop()
        if not held:
            return False
        try:
            lease = self._kube.get_lease(self.name, timeout_s=self._timeout_s)
            if lease is None:
                return False
            spec = lease.get("spec") or {}
            if (spec.get("holderIdentity") or "") != self.identity:
                return False
            body = json.loads(json.dumps(lease))
            body_spec = body.setdefault("spec", {})
            body_spec["holderIdentity"] = None
            body_spec["renewTime"] = micro_time(self._clock())
            self._kube.replace_lease(self.name, body, timeout_s=self._timeout_s)
            logger.info("retune leadership: released lease %s", self.name)
            return True
        except KubeError as error:
            logger.warning(
                "retune leadership: releasing %s failed: %s", self.name, error
            )
            return False

    # ------------------------------------------------------------ helpers

    def _lead(self, t_send: float, transitions: int) -> None:
        with self._lock:
            self._held = True
            self._leader_until = t_send + self.lease_duration_s - self.margin_s
        self.transitions = transitions
        self._observed = None

    def _lose(self) -> LeaseStep:
        lost = self.held
        self.drop()
        return "lost" if lost else "standby"

    def _create(self, t_send: float) -> LeaseStep:
        now = micro_time(self._clock())
        body = {
            "apiVersion": LEASE_API_VERSION,
            "kind": "Lease",
            "metadata": {
                "name": self.name,
                "namespace": self._kube.namespace,
                "labels": {LABEL_DGD: self._dgd.name, LABEL_DGD_UID: self._dgd.uid},
                "ownerReferences": [owner_reference(self._dgd)],
            },
            "spec": {
                "holderIdentity": self.identity,
                "leaseDurationSeconds": self._duration_int,
                "acquireTime": now,
                "renewTime": now,
                "leaseTransitions": 0,
            },
        }
        try:
            self._kube.create_lease(body, timeout_s=self._timeout_s)
        except (KubeAlreadyExists, KubeConflict):
            return self._lose()
        except KubeError as error:
            logger.warning(
                "retune leadership: creating %s failed: %s", self.name, error
            )
            return "error"
        self._lead(t_send, 0)
        logger.info(
            "retune leadership: acquired new lease %s as %s", self.name, self.identity
        )
        return "acquired"

    def _take_over(
        self, lease: dict[str, Any], t_send: float, transitions: int, lost: bool
    ) -> LeaseStep:
        body = self._body(lease, transitions=transitions + 1, acquire=True)
        try:
            self._kube.replace_lease(self.name, body, timeout_s=self._timeout_s)
        except KubeConflict:
            return "lost" if lost else "standby"
        except KubeError as error:
            logger.warning(
                "retune leadership: taking over %s failed: %s", self.name, error
            )
            return "error"
        self._lead(t_send, transitions + 1)
        logger.info(
            "retune leadership: acquired lease %s as %s (transitions=%d)",
            self.name,
            self.identity,
            transitions + 1,
        )
        return "acquired"

    def _body(
        self, lease: dict[str, Any], *, transitions: int, acquire: bool
    ) -> dict[str, Any]:
        body = json.loads(json.dumps(lease))
        metadata = body.setdefault("metadata", {})
        metadata["ownerReferences"] = [owner_reference(self._dgd)]
        spec = body.setdefault("spec", {})
        now = micro_time(self._clock())
        spec["holderIdentity"] = self.identity
        spec["leaseDurationSeconds"] = self._duration_int
        spec["renewTime"] = now
        spec["leaseTransitions"] = int(transitions)
        if acquire or not spec.get("acquireTime"):
            spec["acquireTime"] = now
        return body


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _positive(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None
