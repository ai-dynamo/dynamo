# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the retune router parameters to every frontend router (design §4.8).

``RouterActuator.reconcile`` runs one round: list the DGD's frontend pods,
``GET /router_policy`` on each, keep the routers of the DGD's dynamo
namespace, latch each router's initial configuration, check it against the
parsed startup maps (P14), write what the target needs and report whether
every router has converged.

Wire format (``lib/llm/src/http/service/router_policy.rs``)::

    GET   /router_policy -> {"routers": [{router_id, model, namespace,
                              component, endpoint, worker_type,
                              configuration: {revision, parameters} | null}]}
    PATCH /router_policy  {router_id, expected_revision, parameters: {...}}
                          {router_id, expected_revision, reset: true}
                          -> one router object as above

A PATCH replaces only the keys it names; ``reset: true`` restores every
parameter to the router's startup values. Both bump the router's revision
(``lib/kv-router/src/plugins/worker_selection/runtime.rs``), and GET never
returns the startup values, so the actuator latches them itself: the
configuration of a router seen at revision 0, or returned by a reset, is that
router's initial.

Modes (``ActuationTarget.mode``):

- ``apply``: PATCH explicit values for every one of the 8 policy parameters
  that differs from ``desired_by_wt[worker_type]``. A router whose initial is
  unknown (revision > 0, never latched) is reset once first to learn it; a
  router whose initial differs from the startup map (P14) is never written,
  so planner-parsed values cannot leak into keys the sweep did not search.
- ``rollback``: reset only routers whose configuration differs from their
  latched initial. Never by revision: a reset itself bumps the revision.
- ``observe``: no writes; latch initials and check parity only.

``ActuationTarget.learn_unknown=False`` (advisory) never resets a router to
learn its initial: such a router is left alone, and rollback does not wait
for it.

Targeted pods: Ready frontends (plus NotReady ones with ``include_not_ready``)
that have an IP, are not terminating and are not in phase Succeeded/Failed
(an evicted pod can keep its IP but serves nothing).

Convergence: at least ``max(1, expected_frontends)`` Ready frontends, every
Ready frontend answered GET and has a router with a runtime configuration
for every worker type of the planner mode, and every router seen has its 8
parameters equal to the target (``apply``: the desired map;
``rollback``/``observe``: the router's latched initial). A NotReady pod
that answers GET is checked like the others; one that does not answer is
ignored (a restarted frontend comes back at revision 0, its initial). No
routers at all is not converged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp

from dynamo.planner.retune.live.contracts import (
    POD_LABEL_COMPONENT_TYPE,
    POD_LABEL_DGD_NAME,
    POLICY_PARAMETER_KEYS,
    ActuationReport,
    ActuationTarget,
    DgdRef,
    PodInfo,
    PolicyCallResult,
    RetuneKube,
    RetuneMetricsSink,
    RouterInfo,
    RouterPolicyClient,
)

logger = logging.getLogger(__name__)

ROUTER_POLICY_PATH = "/router_policy"
# Container port of the frontend when the pod has no port named ``http``
# (deploy/operator/internal/consts/consts.go DynamoServicePort).
DEFAULT_FRONTEND_PORT = 8000
# Consecutive rounds of GET 404 on a frontend after which its admin API is
# reported disabled (P12).
ADMIN_API_DISABLED_ROUNDS = 3
# Largest error text kept from a failed call.
_MAX_ERROR_CHARS = 512
# Relative/absolute tolerance of parameter comparisons. Values round-trip
# exactly through serde_json and Python floats; the tolerance only absorbs
# formatting noise from hand-written targets.
_REL_TOL = 1e-9
_ABS_TOL = 1e-12
# Statuses after which the pod is re-read once and the write retried.
_RETRY_AFTER_REGET = frozenset({404, 409, 503})
# The frontend bounds each router's control call by CONTROL_TIMEOUT
# (lib/llm/src/http/service/router_policy.rs): GET then answers 200 without
# the routers that timed out, and PATCH answers 503. The client's total
# timeout must exceed it, or the client gives up first and turns those
# answers into status 0 (a whole pod lost for one stuck router).
ROUTER_CONTROL_TIMEOUT_S = 5.0
DEFAULT_CLIENT_TIMEOUT_S = ROUTER_CONTROL_TIMEOUT_S + 3.0
# Pod phases after which no container runs again.
_FINISHED_POD_PHASES = frozenset({"Succeeded", "Failed"})


# ----------------------------------------------------------------- client


class AiohttpRouterPolicyClient:
    """``RouterPolicyClient`` over a dedicated ``aiohttp.ClientSession``.

    The session is created on first use inside the running event loop and
    closed by ``close()``. Every call returns a ``PolicyCallResult`` and never
    raises for HTTP, network or decoding failures: ``status`` is the HTTP
    status, or 0 when no well-formed response arrived (connection error,
    timeout, a pod without an IP, or a body that is not the documented JSON).
    ``timeout_s`` bounds each call and must stay above the server's
    ``ROUTER_CONTROL_TIMEOUT_S``.
    """

    def __init__(
        self,
        timeout_s: float = DEFAULT_CLIENT_TIMEOUT_S,
        scheme: str = "http",
        *,
        default_port: int = DEFAULT_FRONTEND_PORT,
        max_connections: int = 32,
    ) -> None:
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError("timeout_s must be positive and finite")
        if scheme not in ("http", "https"):
            raise ValueError(f"scheme must be 'http' or 'https', got {scheme!r}")
        self._timeout = aiohttp.ClientTimeout(total=float(timeout_s))
        self._scheme = scheme
        self._default_port = int(default_port)
        self._max_connections = int(max_connections)
        self._session: Optional[aiohttp.ClientSession] = None

    def url(self, pod: PodInfo) -> Optional[str]:
        """The pod's ``/router_policy`` URL, or None without a pod IP."""

        if not pod.ip:
            return None
        host = f"[{pod.ip}]" if ":" in pod.ip else pod.ip
        port = pod.http_port if pod.http_port is not None else self._default_port
        return f"{self._scheme}://{host}:{port}{ROUTER_POLICY_PATH}"

    async def get(self, pod: PodInfo) -> PolicyCallResult:
        return await self._call(pod, "GET", None, _parse_list)

    async def patch(
        self,
        pod: PodInfo,
        router_id: str,
        expected_revision: int,
        parameters: Mapping[str, float],
    ) -> PolicyCallResult:
        if not parameters:
            # The server rejects an empty patch with 400; do not send one.
            return PolicyCallResult(status=0, error="empty parameters")
        body = {
            "router_id": router_id,
            "expected_revision": int(expected_revision),
            "parameters": {key: float(value) for key, value in parameters.items()},
        }
        return await self._call(pod, "PATCH", body, _parse_single)

    async def reset(
        self, pod: PodInfo, router_id: str, expected_revision: int
    ) -> PolicyCallResult:
        body = {
            "router_id": router_id,
            "expected_revision": int(expected_revision),
            "reset": True,
        }
        return await self._call(pod, "PATCH", body, _parse_single)

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None and not session.closed:
            await session.close()

    async def _call(
        self,
        pod: PodInfo,
        method: str,
        body: Optional[dict[str, Any]],
        parse: Any,
    ) -> PolicyCallResult:
        url = self.url(pod)
        if url is None:
            return PolicyCallResult(status=0, error=f"pod {pod.name} has no IP")
        try:
            session = self._ensure_session()
            async with session.request(method, url, json=body) as response:
                raw = await response.read()
                status = response.status
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            return PolicyCallResult(
                status=0, error=_truncate(f"{method} {url}: {exc!r}")
            )
        if status != 200:
            return PolicyCallResult(status=status, error=_error_text(raw))
        try:
            return parse(json.loads(raw))
        except (ValueError, TypeError, KeyError) as exc:
            return PolicyCallResult(
                status=0, error=_truncate(f"malformed {method} response: {exc}")
            )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                connector=aiohttp.TCPConnector(limit=self._max_connections),
                trust_env=False,
            )
        return self._session


def _parse_router(item: Mapping[str, Any]) -> RouterInfo:
    configuration = item.get("configuration")
    parameters: Optional[dict[str, float]] = None
    revision = 0
    if configuration is not None:
        revision = int(configuration["revision"])
        parameters = {
            str(key): float(value)
            for key, value in dict(configuration["parameters"]).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    model = item.get("model")
    return RouterInfo(
        router_id=str(item["router_id"]),
        namespace=str(item["namespace"]),
        worker_type=str(item["worker_type"]),
        revision=revision,
        configuration=parameters,
        model=None if model is None else str(model),
    )


def _parse_list(payload: Any) -> PolicyCallResult:
    routers = payload["routers"]
    if not isinstance(routers, list):
        raise TypeError("routers is not a list")
    return PolicyCallResult(
        status=200, routers=tuple(_parse_router(item) for item in routers)
    )


def _parse_single(payload: Any) -> PolicyCallResult:
    return PolicyCallResult(status=200, router=_parse_router(payload))


def _error_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except ValueError:
        return _truncate(text)
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return _truncate(payload["error"])
    return _truncate(text)


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_ERROR_CHARS else text[:_MAX_ERROR_CHARS] + "..."


# --------------------------------------------------------------- actuator


@dataclass
class _Round:
    """Mutable bookkeeping of one ``reconcile`` call."""

    target: ActuationTarget
    initials: Mapping[str, Mapping[str, float]]
    errors: dict[str, int] = field(default_factory=dict)

    def error(self, kind: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1


@dataclass
class _PodOutcome:
    pod: PodInfo
    ok: bool  # the last GET of this pod succeeded
    get_status: Optional[int] = None
    routers: tuple[RouterInfo, ...] = ()  # in-scope routers, final state


class RouterActuator:
    """Converges the frontends' router policies to an ``ActuationTarget``.

    ``startup_maps`` is the parsed startup ``policy_parameters(wt)`` per
    worker type of the planner mode (``aggregated``, or ``prefill`` and
    ``decode``); its keys are the worker types the actuator manages. The
    actuator keeps a local cache of the initials it latched, so a write in
    the same round, or a state write that has not landed yet, never makes a
    router look like it has an unknown initial. ``reconcile`` never raises
    (except ``asyncio.CancelledError``).
    """

    def __init__(
        self,
        kube: RetuneKube,
        client: RouterPolicyClient,
        dgd: DgdRef,
        startup_maps: dict[str, dict[str, float]],
        metrics: RetuneMetricsSink,
        max_concurrency: int = 8,
        *,
        list_timeout_s: float = 10.0,
        admin_disabled_rounds: int = ADMIN_API_DISABLED_ROUNDS,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if admin_disabled_rounds < 1:
            raise ValueError("admin_disabled_rounds must be >= 1")
        self._kube = kube
        self._client = client
        self._dgd = dgd
        self._metrics = metrics
        self._max_concurrency = int(max_concurrency)
        self._list_timeout_s = float(list_timeout_s)
        self._admin_disabled_rounds = int(admin_disabled_rounds)
        # router_id -> initial configuration latched by this process.
        self._latched: dict[str, dict[str, float]] = {}
        # router_id -> name of the pod it was last seen on.
        self._router_pod: dict[str, str] = {}
        # router_id -> signature of a write the router rejected with 422;
        # the same write is not retried until the target changes.
        self._rejected: dict[str, str] = {}
        # pod uid -> consecutive rounds whose GET returned 404.
        self._get_404_rounds: dict[str, int] = {}
        self._last_pods: Optional[tuple[PodInfo, ...]] = None
        self._startup_maps: dict[str, dict[str, float]] = {}
        self.set_startup_maps(startup_maps)

    # ------------------------------------------------------------------ API

    @property
    def label_selector(self) -> str:
        return (
            f"{POD_LABEL_DGD_NAME}={self._dgd.name},"
            f"{POD_LABEL_COMPONENT_TYPE}=frontend"
        )

    @property
    def worker_types(self) -> tuple[str, ...]:
        return tuple(self._startup_maps)

    @property
    def last_pods(self) -> Optional[tuple[PodInfo, ...]]:
        """Frontend pods of the last round's listing; None if it failed."""

        return self._last_pods

    def set_startup_maps(self, startup_maps: Mapping[str, Mapping[str, float]]) -> None:
        """Replace the parsed startup maps (after a frontend spec change)."""

        if not startup_maps:
            raise ValueError("startup_maps must name at least one worker type")
        self._startup_maps = {
            worker_type: {key: float(value) for key, value in parameters.items()}
            for worker_type, parameters in startup_maps.items()
        }
        self._rejected.clear()

    async def close(self) -> None:
        """Close the policy client when it has a ``close`` coroutine (the
        ``RouterPolicyClient`` protocol does not require one)."""

        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    async def reconcile(
        self,
        target: ActuationTarget,
        *,
        initials: Mapping[str, Mapping[str, float]],
        expected_frontends: int,
    ) -> ActuationReport:
        """Run one round; see the module docstring. Never raises."""

        try:
            report = await self._reconcile(target, initials, expected_frontends)
        except asyncio.CancelledError:
            raise
        except Exception:  # never raise: the supervisor retries next round
            logger.exception("retune router actuator round failed")
            report = ActuationReport(
                converged=False,
                seen_router_ids=frozenset(initials) | frozenset(self._latched),
                errors={"internal": 1},
            )
        self._metrics.router_converged(report.converged)
        return report

    # ------------------------------------------------------------- the round

    async def _reconcile(
        self,
        target: ActuationTarget,
        initials: Mapping[str, Mapping[str, float]],
        expected_frontends: int,
    ) -> ActuationReport:
        state = _Round(target=target, initials=initials)
        target_error = self._target_error(target)
        if target_error is not None:
            logger.error("retune router actuator: invalid target: %s", target_error)
            state.error("invalid_target")
            # Latch and check parity without writing; never report converged.
            state.target = ActuationTarget(mode="observe")

        try:
            pods = await asyncio.to_thread(
                self._kube.list_pods,
                self.label_selector,
                timeout_s=self._list_timeout_s,
            )
        except Exception as exc:  # KubeError or a client bug: retry next round
            logger.warning(
                "retune router actuator: listing frontend pods failed: %s", exc
            )
            self._last_pods = None
            state.error("list_pods")
            return ActuationReport(
                converged=False,
                seen_router_ids=frozenset(initials) | frozenset(self._latched),
                errors=dict(state.errors),
            )
        self._last_pods = tuple(pods)

        live = [pod for pod in pods if _serving(pod)]
        ready = [pod for pod in live if pod.ready and pod.ip]
        targeted = [
            pod
            for pod in live
            if pod.ip and (pod.ready or state.target.include_not_ready)
        ]

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def run(pod: PodInfo) -> _PodOutcome:
            async with semaphore:
                return await self._process_pod(pod, state)

        outcomes = list(await asyncio.gather(*(run(pod) for pod in targeted)))

        self._track_admin_api(pods, outcomes)
        converged = target_error is None and self._converged(
            state, ready, outcomes, max(1, int(expected_frontends))
        )
        routers_present = self._routers_present(
            ready, outcomes, max(1, int(expected_frontends))
        )
        parity_ok, mismatches = self._parity(state, outcomes)
        seen = self._seen_router_ids(state, pods, outcomes)
        self._prune(seen)
        new_initials = {
            router_id: dict(configuration)
            for router_id, configuration in self._latched.items()
            if router_id in seen
            and not _same_map(initials.get(router_id), configuration)
        }
        admin_api_disabled = any(
            count >= self._admin_disabled_rounds
            for count in self._get_404_rounds.values()
        )
        if mismatches:
            logger.warning(
                "retune router actuator: P14 parity mismatch: %s", "; ".join(mismatches)
            )
        return ActuationReport(
            converged=converged,
            new_initials=new_initials,
            seen_router_ids=frozenset(seen),
            parity_ok=parity_ok,
            parity_mismatches=tuple(mismatches),
            admin_api_disabled=admin_api_disabled,
            errors=dict(state.errors),
            routers_present=routers_present,
        )

    def _routers_present(
        self,
        ready: Sequence[PodInfo],
        outcomes: Sequence[_PodOutcome],
        required_ready: int,
    ) -> bool:
        """P12: enough Ready frontends, each GET answered with a configured
        router for every worker type of the planner mode."""

        if len(ready) < required_ready:
            return False
        ready_names = {pod.name for pod in ready}
        answered = 0
        for outcome in outcomes:
            if outcome.pod.name not in ready_names:
                continue
            if not outcome.ok:
                return False
            covered = {
                router.worker_type
                for router in outcome.routers
                if router.configuration is not None
            }
            if covered != set(self._startup_maps):
                return False
            answered += 1
        return answered >= required_ready

    def _target_error(self, target: ActuationTarget) -> Optional[str]:
        if target.mode not in ("apply", "rollback", "observe"):
            return f"unknown mode {target.mode!r}"
        if target.mode != "apply":
            return None
        desired = target.desired_by_wt
        if desired is None:
            return "apply needs desired_by_wt"
        for worker_type in self._startup_maps:
            parameters = desired.get(worker_type)
            if parameters is None:
                return f"desired_by_wt has no {worker_type!r} map"
            for key in POLICY_PARAMETER_KEYS:
                value = parameters.get(key)
                if not _valid_parameter(value):
                    return f"desired_by_wt[{worker_type!r}][{key!r}] is {value!r}"
        return None

    # ------------------------------------------------------------- per pod

    async def _process_pod(self, pod: PodInfo, state: _Round) -> _PodOutcome:
        result = await self._client.get(pod)
        if result.status != 200:
            state.error(f"get:{_code(result.status)}")
            logger.debug(
                "retune router actuator: GET %s -> %s %s",
                pod.name,
                result.status,
                result.error,
            )
            return _PodOutcome(pod=pod, ok=False, get_status=result.status)
        routers = self._in_scope(pod, result.routers, state)
        retried = False
        while True:
            final: list[RouterInfo] = []
            retry = False
            for router in routers:
                current, needs_reget = await self._drive(pod, router, state)
                final.append(current)
                retry = retry or needs_reget
            if not retry or retried:
                return _PodOutcome(
                    pod=pod, ok=True, get_status=200, routers=tuple(final)
                )
            # 409, 503 or PATCH 404: re-read the pod once and retry.
            retried = True
            result = await self._client.get(pod)
            if result.status != 200:
                state.error(f"get:{_code(result.status)}")
                return _PodOutcome(pod=pod, ok=False, get_status=result.status)
            routers = self._in_scope(pod, result.routers, state)

    def _in_scope(
        self, pod: PodInfo, routers: Sequence[RouterInfo], state: _Round
    ) -> list[RouterInfo]:
        """Routers of the DGD's dynamo namespace and the planner's worker
        types; latch the initials of routers at revision 0."""

        scoped = []
        for router in routers:
            if router.namespace != self._dgd.dynamo_namespace:
                continue
            if router.worker_type not in self._startup_maps:
                continue
            self._router_pod[router.router_id] = pod.name
            if router.configuration is None:
                scoped.append(router)
                continue
            if router.revision == 0 and self._initial(router.router_id, state) is None:
                self._latch(router.router_id, router.configuration)
            scoped.append(router)
        return scoped

    def _initial(self, router_id: str, state: _Round) -> Optional[dict[str, float]]:
        latched = self._latched.get(router_id)
        if latched is not None:
            return latched
        persisted = state.initials.get(router_id)
        return None if persisted is None else dict(persisted)

    def _latch(self, router_id: str, configuration: Mapping[str, float]) -> None:
        self._latched[router_id] = {
            key: float(value) for key, value in configuration.items()
        }

    async def _drive(
        self, pod: PodInfo, router: RouterInfo, state: _Round
    ) -> tuple[RouterInfo, bool]:
        """Write what ``router`` needs; return its final state and whether
        the pod should be re-read and the router retried."""

        current = router
        # At most a learning reset followed by one PATCH.
        for _ in range(2):
            action = self._plan(current, state)
            if action is None:
                return current, False
            kind, parameters = action
            signature = self._signature(kind, current.worker_type, state)
            if self._rejected.get(current.router_id) == signature:
                return current, False
            if kind == "reset":
                result = await self._client.reset(
                    pod, current.router_id, current.revision
                )
            else:
                result = await self._client.patch(
                    pod, current.router_id, current.revision, parameters
                )
            if result.status != 200 or result.router is None:
                code = _code(result.status)
                state.error(f"{kind}:{code}")
                self._metrics.patch_error(code)
                logger.warning(
                    "retune router actuator: %s %s router %s rev %d -> %s %s",
                    kind,
                    pod.name,
                    current.router_id,
                    current.revision,
                    result.status,
                    result.error,
                )
                if result.status == 422:
                    self._rejected[current.router_id] = signature
                return current, result.status in _RETRY_AFTER_REGET
            self._rejected.pop(current.router_id, None)
            current = result.router
            if kind == "reset" and current.configuration is not None:
                # A reset restores every parameter to the router's startup
                # values: that response is the router's initial.
                self._latch(current.router_id, current.configuration)
        return current, False

    def _plan(
        self, router: RouterInfo, state: _Round
    ) -> Optional[tuple[str, dict[str, float]]]:
        """``("reset", {})``, ``("patch", diff)`` or None for one router."""

        mode = state.target.mode
        configuration = router.configuration
        if mode == "observe" or configuration is None:
            return None
        initial = self._initial(router.router_id, state)
        if initial is None and not state.target.learn_unknown:
            # Never latched, and learning by reset is off (advisory): the
            # planner never wrote it, so it is not the planner's to touch.
            return None
        if mode == "rollback":
            if initial is None or not _same_map(configuration, initial):
                return ("reset", {})
            return None
        # apply
        if initial is None:
            # Revision > 0 and never latched: learn the initial first.
            return ("reset", {})
        if not _same_map(initial, self._startup_maps[router.worker_type]):
            # P14 failed for this router; writing explicit values would put
            # planner-parsed values into keys the sweep never searched.
            return None
        desired = state.target.desired_by_wt[router.worker_type]  # type: ignore[index]
        diff = {
            key: float(desired[key])
            for key in POLICY_PARAMETER_KEYS
            if not _close(configuration.get(key), desired[key])
        }
        return ("patch", diff) if diff else None

    @staticmethod
    def _signature(kind: str, worker_type: str, state: _Round) -> str:
        """Identity of a write for the 422 memory: mode, kind and target."""

        desired = (state.target.desired_by_wt or {}).get(worker_type)
        return json.dumps(
            [state.target.mode, kind, desired], sort_keys=True, default=float
        )

    # ------------------------------------------------------------ verdicts

    def _left_alone(self, router: RouterInfo, state: _Round) -> bool:
        """A rollback without learning skips routers with no known initial."""

        return (
            state.target.mode == "rollback"
            and not state.target.learn_unknown
            and self._initial(router.router_id, state) is None
        )

    def _target_map(
        self, router: RouterInfo, state: _Round
    ) -> Optional[Mapping[str, float]]:
        if state.target.mode == "apply":
            return state.target.desired_by_wt[router.worker_type]  # type: ignore[index]
        return self._initial(router.router_id, state)

    def _converged(
        self,
        state: _Round,
        ready: Sequence[PodInfo],
        outcomes: Sequence[_PodOutcome],
        required_ready: int,
    ) -> bool:
        if len(ready) < required_ready:
            return False
        configured = 0
        for outcome in outcomes:
            if not outcome.ok:
                if outcome.pod.ready:
                    return False
                # An unreachable NotReady pod (crash loop, still starting) is
                # ignored: it comes back at revision 0, i.e. at its initial.
                continue
            covered: set[str] = set()
            for router in outcome.routers:
                if router.configuration is None:
                    continue
                covered.add(router.worker_type)
                configured += 1
                if self._left_alone(router, state):
                    continue
                expected = self._target_map(router, state)
                if expected is None or not _same_map(router.configuration, expected):
                    return False
            if outcome.pod.ready and covered != set(self._startup_maps):
                return False
        return configured > 0

    def _parity(
        self, state: _Round, outcomes: Sequence[_PodOutcome]
    ) -> tuple[Optional[bool], list[str]]:
        mismatches: list[str] = []
        known = 0
        unknown = 0
        for outcome in outcomes:
            for router in outcome.routers:
                if router.configuration is None:
                    continue
                initial = self._initial(router.router_id, state)
                if initial is None:
                    unknown += 1
                    continue
                known += 1
                startup = self._startup_maps[router.worker_type]
                differing = [
                    f"{key} initial={initial.get(key)} startup={startup.get(key)}"
                    for key in POLICY_PARAMETER_KEYS
                    if not _close(initial.get(key), startup.get(key))
                ]
                if differing:
                    mismatches.append(
                        f"{outcome.pod.name}/{router.router_id}/{router.worker_type}: "
                        + ", ".join(differing)
                    )
        if mismatches:
            return False, mismatches
        if known == 0 or unknown:
            return None, mismatches
        return True, mismatches

    def _seen_router_ids(
        self,
        state: _Round,
        pods: Sequence[PodInfo],
        outcomes: Sequence[_PodOutcome],
    ) -> set[str]:
        """Routers observed this round plus known ones not provably gone.

        A known router is gone when its pod no longer exists (or has
        finished: Succeeded/Failed) or answered GET without it; with an
        unknown pod, when every live frontend with an IP answered GET this
        round.
        """

        observed = {
            router.router_id for outcome in outcomes for router in outcome.routers
        }
        existing = {pod.name for pod in pods if pod.phase not in _FINISHED_POD_PHASES}
        answered = {outcome.pod.name for outcome in outcomes if outcome.ok}
        complete = all(pod.name in answered for pod in pods if _serving(pod) and pod.ip)
        seen = set(observed)
        for router_id in set(state.initials) | set(self._latched):
            if router_id in observed:
                continue
            pod_name = self._router_pod.get(router_id)
            if pod_name is None:
                gone = complete
            else:
                gone = pod_name not in existing or pod_name in answered
            if not gone:
                seen.add(router_id)
        return seen

    def _prune(self, seen: set[str]) -> None:
        for cache in (self._latched, self._router_pod, self._rejected):
            for router_id in [key for key in cache if key not in seen]:
                del cache[router_id]

    def _track_admin_api(
        self, pods: Sequence[PodInfo], outcomes: Sequence[_PodOutcome]
    ) -> None:
        live_uids = {pod.uid for pod in pods}
        for uid in [uid for uid in self._get_404_rounds if uid not in live_uids]:
            del self._get_404_rounds[uid]
        for outcome in outcomes:
            uid = outcome.pod.uid
            if outcome.get_status == 404:
                self._get_404_rounds[uid] = self._get_404_rounds.get(uid, 0) + 1
            elif outcome.get_status is not None:
                self._get_404_rounds.pop(uid, None)


# ---------------------------------------------------------------- helpers


def _serving(pod: PodInfo) -> bool:
    """Not terminating and not finished: its frontend may still answer."""

    return not pod.terminating and pod.phase not in _FINISHED_POD_PHASES


def _valid_parameter(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def _code(status: int) -> str:
    return str(status) if status else "network"


def _close(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(float(left), float(right), rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _same_map(
    left: Optional[Mapping[str, float]], right: Optional[Mapping[str, float]]
) -> bool:
    """Equal on the 8 policy parameters (a missing key is never equal)."""

    if left is None or right is None:
        return False
    return all(_close(left.get(key), right.get(key)) for key in POLICY_PARAMETER_KEYS)
