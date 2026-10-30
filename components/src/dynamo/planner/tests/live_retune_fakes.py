# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory fakes of the Kubernetes side of live retune.

``FakeRetuneKube`` implements ``contracts.RetuneKube`` over plain dicts with
the API-server behaviour the live modules depend on: ``resourceVersion``
compare-and-swap on Lease replace, ``AlreadyExists`` on create, Strict field
validation and silent pruning on Request create, label selectors, cascading
Request deletion, and injectable faults (including writes that time out but
land). ``FakeDgdrOperator`` drives the Requests in a ``FakeRetuneKube``
through the DGDR v1beta2 lifecycle (Accepted -> Run -> Completed with
Candidates) the way the operator at #17483 does, lowering each Request into
``Run.status.aisimulate.input`` with the same merge-patch rules.

This is a plain module, not a conftest: tests import what they need.
"""

from __future__ import annotations

import copy
import functools
import itertools
import json
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional

from dynamo.planner.retune.live.contracts import (
    DGDR_ROUTER_KNOBS,
    POD_LABEL_COMPONENT_TYPE,
    POD_LABEL_DGD_NAME,
    RUN_UID_LABEL,
    KubeAlreadyExists,
    KubeConflict,
    KubeError,
    KubeInvalid,
    KubeNotFound,
    PodInfo,
)

# Operator constants (deploy/operator/internal/controller at #17483).
REQUEST_GENERATION_ANNOTATION = "nvidia.com/dgdr-request-generation"
CANDIDATE_ID_LABEL = "nvidia.com/dgdr-candidate-id"
TRACE_MOUNT_PATH = "/var/run/aisimulate/trace"
DEFAULT_RUNTIME_IMAGE = "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.8.0"
DEFAULT_DYNAMO_VERSION = "1.8.0"
ROUTER_ADAPTER = "dynamo.router"

_METRIC_TARGETS = {
    "Throughput": "throughput",
    "ThroughputPerGPU": "throughput_per_gpu",
    "ThroughputPerUser": "throughput_per_user",
    "TTFT": "ttft",
    "E2ELatency": "e2e_latency",
    "Goodput": "goodput",
    "GoodputPerGPU": "goodput_per_gpu",
}

KubeMethod = Literal[
    "get_dgd",
    "list_pods",
    "create_request",
    "get_request",
    "list_requests",
    "delete_request",
    "get_run",
    "list_candidates",
    "get_lease",
    "create_lease",
    "replace_lease",
    "delete_lease",
]


# ------------------------------------------------------------------ helpers


def merge_patch(target: Any, patch: Any) -> Any:
    """RFC 7386 JSON merge patch; returns a new value, never mutates."""

    if not isinstance(patch, Mapping):
        return copy.deepcopy(patch)
    result = dict(target) if isinstance(target, Mapping) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def nested_get(obj: Any, *path: str, default: Any = None) -> Any:
    for key in path:
        if not isinstance(obj, Mapping) or key not in obj:
            return default
        obj = obj[key]
    return obj


def _nested_set(obj: dict, path: tuple[str, ...], value: Any) -> None:
    for key in path[:-1]:
        child = obj.get(key)
        if not isinstance(child, dict):
            child = {}
            obj[key] = child
        obj = child
    obj[path[-1]] = value


def _nested_pop(obj: Any, path: tuple[str, ...]) -> bool:
    for key in path[:-1]:
        if not isinstance(obj, dict) or key not in obj:
            return False
        obj = obj[key]
    if isinstance(obj, dict) and path[-1] in obj:
        del obj[path[-1]]
        return True
    return False


def _split(path: str) -> tuple[str, ...]:
    return tuple(path.split("."))


def rfc3339(epoch_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_s))


_SELECTOR_TERM = re.compile(
    r"^\s*(!?)\s*([A-Za-z0-9._/-]+)\s*(?:(==|=|!=)\s*(\S*))?\s*$"
)


def match_labels(labels: Mapping[str, str] | None, selector: str) -> bool:
    """Equality-based label selector: ``k=v``, ``k==v``, ``k!=v``, ``k``, ``!k``."""

    labels = labels or {}
    if not selector.strip():
        return True
    for term in selector.split(","):
        found = _SELECTOR_TERM.match(term)
        if found is None:
            raise ValueError(f"unsupported label selector term {term!r}")
        negate, key, op, value = found.groups()
        if op is None:
            present = key in labels
            if present == bool(negate):
                return False
        elif negate:
            raise ValueError(f"unsupported label selector term {term!r}")
        elif op in ("=", "=="):
            if labels.get(key) != value:
                return False
        elif labels.get(key) == value:  # !=
            return False
    return True


def lower_request_input(
    request: Mapping[str, Any],
    *,
    run_name: str,
    runtime_image: str = DEFAULT_RUNTIME_IMAGE,
    dynamo_version: str = DEFAULT_DYNAMO_VERSION,
    gpus_per_node: int = 8,
) -> dict[str, Any]:
    """``Run.status.aisimulate.input`` for a Request, as the operator lowers it.

    Mirrors ``aisimulate.LowerWithUnderlay`` and ``dgdrRunInput`` at #17483:
    the public role underlay, the ``dgdr_run`` static underlay, the typed
    underlay, the derived defaults, ``inputOverrides`` as the final merge
    patch, then the operator-owned ``dgdr_run`` fields and the trace source.
    """

    spec = request.get("spec") or {}
    max_candidates = int(nested_get(spec, "recommendation", "maxCandidates", default=1))
    underlay: dict[str, Any] = {
        "engine": {"workers": {"aggregated": {}, "prefill": {}, "decode": {}}}
    }
    underlay = merge_patch(
        underlay, {"dgdr_run": {"name": run_name, "max_candidates": max_candidates}}
    )
    typed: dict[str, Any] = {}
    model = nested_get(spec, "model", "name")
    if model is not None:
        _nested_set(typed, ("engine", "model"), model)
    if spec.get("backends"):
        _nested_set(typed, ("engine", "backend", "choices"), list(spec["backends"]))
    sku = nested_get(spec, "hardware", "gpu", "sku")
    if sku:
        _nested_set(typed, ("engine", "hardware"), sku)
    budget = nested_get(spec, "hardware", "gpu", "budget")
    if budget is not None:
        _nested_set(
            typed, ("optimization", "constraints", "max_candidate_gpus"), int(budget)
        )
    trace = nested_get(spec, "workload", "trace")
    trace_source = None
    if trace is not None:
        path = nested_get(trace, "source", "pvc", "path", default="")
        mounted = TRACE_MOUNT_PATH + (
            "/trace.jsonl.gz" if path.endswith(".gz") else "/trace.jsonl"
        )
        trace_source = {
            "type": "trace",
            "format": trace.get("format"),
            "paths": [mounted],
        }
        _nested_set(typed, ("traffic", "source"), dict(trace_source))
        _nested_set(typed, ("traffic", "load", "type"), "trace_timestamps")
        speedup = nested_get(trace, "replay", "arrivalSpeedupRatio")
        if speedup is not None:
            _nested_set(typed, ("traffic", "load", "speedup"), speedup)
    objective = spec.get("objective") or {}
    if objective.get("metric") is not None:
        _nested_set(
            typed, ("optimization", "target"), _METRIC_TARGETS[objective["metric"]]
        )
    sla = objective.get("sla") or {}
    for api_key, input_key in (
        ("ttftMs", "ttft_ms"),
        ("itlMs", "itl_ms"),
        ("e2eMs", "e2e_ms"),
    ):
        if sla.get(api_key) is not None:
            _nested_set(typed, ("evaluation", "sla", input_key), sla[api_key])
    underlay = merge_patch(underlay, typed)
    if "traffic" in underlay:
        source_type = nested_get(underlay, "traffic", "source", "type")
        if source_type != "trace" and nested_get(underlay, "traffic", "stop") is None:
            _nested_set(underlay, ("traffic", "stop", "requests"), 1000.0)
    if nested_get(underlay, "optimizer", "max_trials") is None:
        _nested_set(underlay, ("optimizer", "max_trials"), 320.0)
    if nested_get(underlay, "evaluation", "sla") and (
        nested_get(underlay, "optimization", "strict_sla") is None
    ):
        _nested_set(underlay, ("optimization", "strict_sla"), True)
    overrides = nested_get(spec, "aisimulate", "inputOverrides", default={}) or {}
    effective = merge_patch(underlay, overrides)
    reserved = {
        "name": run_name,
        "max_candidates": max_candidates,
        "snapshot_dir": "/var/run/aisimulate/output",
        "namespace": nested_get(request, "metadata", "namespace"),
        "runtime_image": runtime_image,
        "runtime_version_override": dynamo_version,
        "derive_runtime_images": True,
        "num_gpus_per_node": gpus_per_node,
    }
    for key, value in reserved.items():
        _nested_set(effective, ("dgdr_run", key), value)
    if trace_source is not None:
        traffic = effective.setdefault("traffic", {})
        traffic.setdefault("source", {}).update(trace_source)
    # Round-trip through JSON like the API server does.
    return json.loads(json.dumps(effective))


def router_points(effective_input: Mapping[str, Any]) -> list[dict[str, float]]:
    """The router points (public knob names) an input's router section spans."""

    router = effective_input.get("router") or {}
    axes: list[tuple[str, list[float]]] = []
    for public in DGDR_ROUTER_KNOBS:
        value = router.get(public)
        if isinstance(value, Mapping):
            axes.append((public, [float(v) for v in value["choices"]]))
        elif value is not None:
            axes.append((public, [float(value)]))
    return [
        dict(zip([name for name, _ in axes], values))
        for values in itertools.product(*(values for _, values in axes))
    ]


def engine_parameters(effective_input: Mapping[str, Any]) -> dict[str, Any]:
    """The non-router parameters every Candidate of a pinned engine shares."""

    engine = effective_input.get("engine") or {}
    workers = engine.get("workers") or {}
    params: dict[str, Any] = {
        "backend": engine.get("backend"),
        "deployment_mode": "disagg" if engine.get("mode") == "disaggregated" else "agg",
        "model": engine.get("model"),
        "hardware": engine.get("hardware"),
    }
    for role, worker in sorted(workers.items()):
        if isinstance(worker, Mapping):
            preset = nested_get(worker, "parallelism", "preset", default=[{}])
            params[f"{role}_parallelism"] = dict(preset[0]) if preset else {}
    return json.loads(json.dumps(params))


def candidate_parameters(
    effective_input: Mapping[str, Any], point_public: Mapping[str, float]
) -> dict[str, Any]:
    """``Candidate.spec.parameters`` of one evaluated router point."""

    router = effective_input.get("router") or {}
    return {
        **engine_parameters(effective_input),
        "adapters": {
            ROUTER_ADAPTER: {
                "policy": router.get("policy", "kv_router"),
                "prefill_load_model": dict(
                    router.get("prefill_load_model") or {"type": "none"}
                ),
                **{name: float(value) for name, value in point_public.items()},
            }
        },
    }


def default_metrics(point_public: Mapping[str, float]) -> dict[str, float]:
    """Deterministic replay summary metrics of a router point (for tests)."""

    credit = point_public.get("overlap_score_credit", 1.0)
    scale = point_public.get("prefill_load_scale", 1.0)
    temperature = point_public.get("temperature", 0.0)
    goodput = 1000.0 + 100.0 * credit - 3.0 * abs(scale - 2.0) - 10.0 * temperature
    return {
        "goodput_output_throughput_tok_s": goodput,
        "output_throughput_tok_s": goodput * 1.1,
        "gpu_hours": 2.0,
        "duration_ms": 3_600_000.0,
        "mean_ttft_ms": 100.0 + scale,
        "num_ttft_samples": 1000.0,
        "mean_e2e_latency_ms": 1000.0 + scale,
        "num_e2e_latency_samples": 1000.0,
        "mean_output_token_throughput_per_user": 50.0 + credit,
    }


# ------------------------------------------------------------------- faults


@dataclass
class Fault:
    """One injected failure of a ``FakeRetuneKube`` method.

    ``landed=True`` performs the call and then raises (a write that timed
    out but reached the API server). ``match`` limits the fault to calls
    whose first positional argument (a name, selector or body) satisfies it.
    """

    error: Exception
    times: int = 1
    landed: bool = False
    match: Optional[Callable[[Any], bool]] = None


@dataclass
class KubeCall:
    method: str
    target: Any
    kwargs: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- kube


class FakeRetuneKube:
    """In-memory ``contracts.RetuneKube`` (thread-safe).

    Stores DGDs, pods (``PodInfo``), Requests, Runs, Candidates and Leases
    as plain dicts. Objects come back as deep copies, so callers can never
    alias the store. ``now`` stamps ``creationTimestamp``.

    Request create honours ``field_validation='Strict'``: a path listed in
    ``unknown_fields`` (dotted, e.g. ``spec.workload.trace``) is rejected
    with ``KubeInvalid``; a path in ``pruned_fields`` is dropped silently
    (an API server that ignores the validation directive). ``reject_create``
    rejects every create with ``KubeInvalid``. ``on_request_created`` hooks
    run after each stored (non dry-run) create.
    """

    def __init__(
        self,
        namespace: str = "test-ns",
        *,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.namespace = namespace
        self.now = now
        self.dgds: dict[str, dict] = {}
        self.pods: dict[str, PodInfo] = {}
        self.requests: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        # Each Run's lowered input, recorded in its status by ``create_job``.
        self.run_inputs: dict[str, dict] = {}
        self.candidates: dict[str, dict] = {}
        self.leases: dict[str, dict] = {}
        self.calls: list[KubeCall] = []
        self.deleted_requests: list[tuple[str, str]] = []
        self.unknown_fields: set[str] = set()
        self.pruned_fields: set[str] = set()
        self.reject_create: Optional[str] = None
        self.on_request_created: list[Callable[[FakeRetuneKube, dict], None]] = []
        self.cascade_deletes = True
        self._faults: dict[str, deque[Fault]] = {}
        self._rv = itertools.count(1)
        self._lock = threading.RLock()

    # ------------------------------------------------------------ faults

    def fail(
        self,
        method: KubeMethod,
        error: Exception | None = None,
        *,
        times: int = 1,
        landed: bool = False,
        match: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        """Make the next ``times`` matching calls of ``method`` raise ``error``."""

        with self._lock:
            self._faults.setdefault(method, deque()).append(
                Fault(
                    error=error or KubeError(f"injected {method} failure"),
                    times=times,
                    landed=landed,
                    match=match,
                )
            )

    def clear_faults(self) -> None:
        with self._lock:
            self._faults.clear()

    def calls_of(self, method: str) -> list[KubeCall]:
        with self._lock:
            return [call for call in self.calls if call.method == method]

    def _call(
        self,
        method: str,
        target: Any,
        kwargs: dict[str, Any],
        operation: Callable[[], Any],
    ) -> Any:
        with self._lock:
            self.calls.append(KubeCall(method, copy.deepcopy(target), dict(kwargs)))
            fault = self._take_fault(method, target)
            if fault is not None and not fault.landed:
                raise fault.error
            result = operation()
            if fault is not None:
                raise fault.error
            return result

    def _take_fault(self, method: str, target: Any) -> Optional[Fault]:
        queue = self._faults.get(method)
        if not queue:
            return None
        for fault in list(queue):
            if fault.match is None or fault.match(target):
                fault.times -= 1
                if fault.times <= 0:
                    queue.remove(fault)
                return fault
        return None

    def _stamp(self, obj: dict, *, kind: str, api_version: str) -> dict:
        metadata = obj.setdefault("metadata", {})
        obj.setdefault("apiVersion", api_version)
        obj.setdefault("kind", kind)
        metadata.setdefault("namespace", self.namespace)
        metadata.setdefault("uid", str(uuid.uuid4()))
        metadata.setdefault("creationTimestamp", rfc3339(self.now()))
        metadata.setdefault("generation", 1)
        metadata["resourceVersion"] = str(next(self._rv))
        return obj

    def bump(self, obj: dict) -> None:
        """Give a stored object a new ``resourceVersion``."""

        obj["metadata"]["resourceVersion"] = str(next(self._rv))

    # --------------------------------------------------------------- DGD

    def put_dgd(self, dgd: dict) -> dict:
        with self._lock:
            stored = copy.deepcopy(dgd)
            self._stamp(
                stored, kind="DynamoGraphDeployment", api_version="nvidia.com/v1beta1"
            )
            self.dgds[stored["metadata"]["name"]] = stored
            return copy.deepcopy(stored)

    def get_dgd(self, name: str, *, timeout_s: float = 10.0) -> dict:
        def operation() -> dict:
            if name not in self.dgds:
                raise KubeNotFound(f"DynamoGraphDeployment {name!r} not found")
            return copy.deepcopy(self.dgds[name])

        return self._call("get_dgd", name, {"timeout_s": timeout_s}, operation)

    # -------------------------------------------------------------- pods

    def add_pod(
        self,
        name: str,
        *,
        dgd: str | None = None,
        component_type: str = "frontend",
        ready: bool = True,
        terminating: bool = False,
        ip: str | None = None,
        http_port: int | None = 8000,
        labels: Mapping[str, str] | None = None,
    ) -> PodInfo:
        all_labels = dict(labels or {})
        if dgd is not None:
            all_labels.setdefault(POD_LABEL_DGD_NAME, dgd)
        if component_type:
            all_labels.setdefault(POD_LABEL_COMPONENT_TYPE, component_type)
        pod = PodInfo(
            name=name,
            uid=str(uuid.uuid4()),
            ip=ip if ip is not None else f"10.0.0.{len(self.pods) + 1}",
            ready=ready,
            terminating=terminating,
            labels=all_labels,
            http_port=http_port,
        )
        with self._lock:
            self.pods[name] = pod
        return pod

    def update_pod(self, name: str, **changes: Any) -> PodInfo:
        with self._lock:
            pod = replace(self.pods[name], **changes)
            self.pods[name] = pod
            return pod

    def remove_pod(self, name: str) -> None:
        with self._lock:
            self.pods.pop(name, None)

    def list_pods(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[PodInfo]:
        def operation() -> list[PodInfo]:
            return [
                pod
                for pod in self.pods.values()
                if match_labels(pod.labels, label_selector)
            ]

        return self._call(
            "list_pods", label_selector, {"timeout_s": timeout_s}, operation
        )

    # ----------------------------------------------------------- Requests

    def create_request(
        self,
        body: dict,
        *,
        dry_run: bool = False,
        timeout_s: float = 10.0,
    ) -> dict:
        def operation() -> dict:
            if self.reject_create is not None:
                raise KubeInvalid(self.reject_create)
            stored = copy.deepcopy(body)
            for path in sorted(self.unknown_fields):
                if nested_get(stored, *_split(path)) is not None:
                    raise KubeInvalid(f'strict decoding error: unknown field "{path}"')
            for path in self.pruned_fields:
                _nested_pop(stored, _split(path))
            name = nested_get(stored, "metadata", "name")
            if not name:
                raise KubeInvalid("metadata.name is required")
            if name in self.requests:
                raise KubeAlreadyExists(
                    f"dynamographdeploymentrequests {name!r} already exists"
                )
            self._stamp(
                stored,
                kind="DynamoGraphDeploymentRequest",
                api_version="nvidia.com/v1beta2",
            )
            if dry_run:
                return stored
            self.requests[name] = stored
            for hook in list(self.on_request_created):
                hook(self, copy.deepcopy(stored))
            return copy.deepcopy(stored)

        return self._call(
            "create_request",
            body,
            {"dry_run": dry_run, "timeout_s": timeout_s},
            operation,
        )

    def get_request(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        def operation() -> Optional[dict]:
            found = self.requests.get(name)
            return None if found is None else copy.deepcopy(found)

        return self._call("get_request", name, {"timeout_s": timeout_s}, operation)

    def list_requests(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[dict]:
        def operation() -> list[dict]:
            return [
                copy.deepcopy(obj)
                for obj in self.requests.values()
                if match_labels(obj["metadata"].get("labels"), label_selector)
            ]

        return self._call(
            "list_requests", label_selector, {"timeout_s": timeout_s}, operation
        )

    def delete_request(
        self,
        name: str,
        *,
        propagation: Literal["Background", "Foreground"] = "Background",
        timeout_s: float = 10.0,
    ) -> None:
        def operation() -> None:
            request = self.requests.pop(name, None)
            if request is None:
                return
            self.deleted_requests.append((name, propagation))
            if not self.cascade_deletes:
                return
            request_uid = request["metadata"]["uid"]
            for run_name, run in list(self.runs.items()):
                owners = run["metadata"].get("ownerReferences") or []
                if any(owner.get("uid") == request_uid for owner in owners):
                    self._delete_run(run_name)

        return self._call(
            "delete_request",
            name,
            {"propagation": propagation, "timeout_s": timeout_s},
            operation,
        )

    # --------------------------------------------------- Runs, Candidates

    def _delete_run(self, run_name: str) -> None:
        run = self.runs.pop(run_name, None)
        self.run_inputs.pop(run_name, None)
        if run is None:
            return
        run_uid = run["metadata"]["uid"]
        for candidate_name, candidate in list(self.candidates.items()):
            labels = candidate["metadata"].get("labels") or {}
            if labels.get(RUN_UID_LABEL) == run_uid:
                del self.candidates[candidate_name]

    def get_run(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        def operation() -> Optional[dict]:
            found = self.runs.get(name)
            return None if found is None else copy.deepcopy(found)

        return self._call("get_run", name, {"timeout_s": timeout_s}, operation)

    def list_candidates(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[dict]:
        def operation() -> list[dict]:
            return [
                copy.deepcopy(obj)
                for obj in self.candidates.values()
                if match_labels(obj["metadata"].get("labels"), label_selector)
            ]

        return self._call(
            "list_candidates", label_selector, {"timeout_s": timeout_s}, operation
        )

    # ------------------------------------------------------------- Leases

    def get_lease(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        def operation() -> Optional[dict]:
            found = self.leases.get(name)
            return None if found is None else copy.deepcopy(found)

        return self._call("get_lease", name, {"timeout_s": timeout_s}, operation)

    def create_lease(self, body: dict, *, timeout_s: float = 10.0) -> dict:
        def operation() -> dict:
            stored = copy.deepcopy(body)
            name = nested_get(stored, "metadata", "name")
            if not name:
                raise KubeInvalid("metadata.name is required")
            if name in self.leases:
                raise KubeAlreadyExists(f"leases {name!r} already exists")
            stored["metadata"].pop("resourceVersion", None)
            self._stamp(stored, kind="Lease", api_version="coordination.k8s.io/v1")
            self.leases[name] = stored
            return copy.deepcopy(stored)

        return self._call("create_lease", body, {"timeout_s": timeout_s}, operation)

    def replace_lease(self, name: str, body: dict, *, timeout_s: float = 10.0) -> dict:
        def operation() -> dict:
            current = self.leases.get(name)
            if current is None:
                raise KubeNotFound(f"leases {name!r} not found")
            sent = nested_get(body, "metadata", "resourceVersion")
            if sent != current["metadata"]["resourceVersion"]:
                raise KubeConflict(
                    f"leases {name!r}: resourceVersion {sent!r} is stale "
                    f"(current {current['metadata']['resourceVersion']!r})"
                )
            stored = copy.deepcopy(body)
            metadata = stored.setdefault("metadata", {})
            for key in ("uid", "creationTimestamp", "namespace", "generation"):
                metadata[key] = current["metadata"].get(key)
            metadata["name"] = name
            stored.setdefault("apiVersion", current.get("apiVersion"))
            stored.setdefault("kind", current.get("kind"))
            self.bump(stored)
            self.leases[name] = stored
            return copy.deepcopy(stored)

        return self._call(
            "replace_lease", (name, body), {"timeout_s": timeout_s}, operation
        )

    def delete_lease(self, name: str, *, timeout_s: float = 10.0) -> None:
        def operation() -> None:
            self.leases.pop(name, None)

        return self._call("delete_lease", name, {"timeout_s": timeout_s}, operation)

    # ------------------------------------------------- operator simulation

    def set_request_condition(
        self,
        name: str,
        *,
        status: str,
        reason: str,
        message: str = "",
        observed_generation: int | None = None,
    ) -> None:
        """Set the Request's ``Accepted`` condition."""

        with self._lock:
            request = self.requests[name]
            generation = request["metadata"].get("generation", 1)
            conditions = [
                condition
                for condition in request.setdefault("status", {}).get("conditions", [])
                if condition.get("type") != "Accepted"
            ]
            conditions.append(
                {
                    "type": "Accepted",
                    "status": status,
                    "reason": reason,
                    "message": message,
                    "observedGeneration": (
                        generation
                        if observed_generation is None
                        else observed_generation
                    ),
                    "lastTransitionTime": rfc3339(self.now()),
                }
            )
            request["status"]["conditions"] = conditions
            self.bump(request)

    def accept_request(
        self,
        name: str,
        *,
        runtime_image: str = DEFAULT_RUNTIME_IMAGE,
        dynamo_version: str = DEFAULT_DYNAMO_VERSION,
        mutate_input: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Create the Request's Run (``<name>-<generation>``) and mark it Accepted.

        As with the operator, the new Run has no status yet: ``create_job``
        later records its lowered input (kept in ``run_inputs`` until then)
        together with ``jobRef``.
        """

        with self._lock:
            request = self.requests[name]
            generation = int(request["metadata"].get("generation", 1))
            run_name = f"{name}-{generation}"
            effective = lower_request_input(
                request,
                run_name=run_name,
                runtime_image=runtime_image,
                dynamo_version=dynamo_version,
            )
            if mutate_input is not None:
                mutate_input(effective)
            run: dict[str, Any] = {
                "metadata": {
                    "name": run_name,
                    "annotations": {REQUEST_GENERATION_ANNOTATION: str(generation)},
                    "ownerReferences": [
                        {
                            "apiVersion": "nvidia.com/v1beta2",
                            "kind": "DynamoGraphDeploymentRequest",
                            "name": name,
                            "uid": request["metadata"]["uid"],
                            "controller": True,
                            "blockOwnerDeletion": True,
                        }
                    ],
                },
                "spec": copy.deepcopy(
                    {k: v for k, v in request.get("spec", {}).items() if k != "v1beta1"}
                ),
                "status": {},
            }
            self._stamp(
                run, kind="DynamoGraphDeploymentRun", api_version="nvidia.com/v1beta2"
            )
            self.runs[run_name] = run
            self.run_inputs[run_name] = effective
            status = request.setdefault("status", {})
            status["observedGeneration"] = generation
            status["activeRunRef"] = {"name": run_name}
            self.set_request_condition(
                name,
                status="True",
                reason="RunCreated",
                message=f"Run {run_name!r} exists for generation {generation}",
            )
            return copy.deepcopy(run)

    def create_job(self, run_name: str, *, record_input: bool = True) -> None:
        """Create the Run's Job, as the operator's ``reconcileRunStatus`` does.

        One status write records ``status.aisimulate.input`` and ``jobRef``
        with ``Completed=False`` (``JobPending``). ``record_input=False``
        leaves the input out (an operator that does not record it).
        """

        with self._lock:
            run = self.runs[run_name]
            status = run.setdefault("status", {})
            if record_input:
                status["aisimulate"] = {
                    "input": copy.deepcopy(self.run_inputs[run_name])
                }
            status["jobRef"] = {"name": f"{run_name}-job", "uid": str(uuid.uuid4())}
            status["phase"] = "Pending"
            self._set_completed(
                run, "False", "JobPending", "AISimulate Job is waiting to start"
            )

    def start_run(self, run_name: str, *, record_input: bool = True) -> None:
        """Start the Run's Job (creating it first, see ``create_job``)."""

        with self._lock:
            run = self.runs[run_name]
            if not run["status"].get("jobRef"):
                self.create_job(run_name, record_input=record_input)
            status = run["status"]
            status["startTime"] = rfc3339(self.now())
            status["phase"] = "Running"
            self._set_completed(run, "False", "JobRunning", "AISimulate Job is running")

    def add_candidate(
        self,
        run_name: str,
        parameters: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> dict:
        with self._lock:
            run = self.runs[run_name]
            canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
            candidate_id = (
                "evaluated-point-" + uuid.uuid5(uuid.NAMESPACE_OID, canonical).hex[:12]
            )
            candidate = {
                "metadata": {
                    "name": f"{run_name}-{candidate_id}",
                    "labels": {
                        RUN_UID_LABEL: run["metadata"]["uid"],
                        CANDIDATE_ID_LABEL: candidate_id,
                    },
                    "ownerReferences": [
                        {
                            "apiVersion": "nvidia.com/v1beta2",
                            "kind": "DynamoGraphDeploymentRun",
                            "name": run_name,
                            "uid": run["metadata"]["uid"],
                            "controller": True,
                        }
                    ],
                },
                "spec": {"parameters": json.loads(canonical)},
                "status": {"metrics": json.loads(json.dumps(dict(metrics)))},
            }
            self._stamp(
                candidate,
                kind="DynamoGraphDeploymentCandidate",
                api_version="nvidia.com/v1beta2",
            )
            self.candidates[candidate["metadata"]["name"]] = candidate
            return copy.deepcopy(candidate)

    def complete_run(
        self,
        run_name: str,
        *,
        reason: str = "SearchSucceeded",
        message: str = "",
        candidates: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]] = (),
    ) -> None:
        """Publish ``candidates`` (parameters, metrics) and complete the Run."""

        with self._lock:
            run = self.runs[run_name]
            refs = [
                {
                    "name": self.add_candidate(run_name, params, metrics)["metadata"][
                        "name"
                    ]
                }
                for params, metrics in candidates
            ]
            status = run["status"]
            status["candidateRefs"] = refs
            status["phase"] = "Succeeded" if reason == "SearchSucceeded" else "Failed"
            status["completionTime"] = rfc3339(self.now())
            self._set_completed(run, "True", reason, message)
            owner = (run["metadata"].get("ownerReferences") or [{}])[0]
            request = self.requests.get(owner.get("name", ""))
            if request is not None:
                request_status = request.setdefault("status", {})
                active = request_status.get("activeRunRef") or {}
                if active.get("name") == run_name:
                    request_status.pop("activeRunRef", None)
                recent = [
                    ref
                    for ref in request_status.get("recentRunRefs", [])
                    if ref.get("name") != run_name
                ]
                request_status["recentRunRefs"] = [{"name": run_name}, *recent]
                self.bump(request)

    def _set_completed(self, run: dict, status: str, reason: str, message: str) -> None:
        conditions = [
            condition
            for condition in run["status"].get("conditions", [])
            if condition.get("type") != "Completed"
        ]
        conditions.append(
            {
                "type": "Completed",
                "status": status,
                "reason": reason,
                "message": message,
                "lastTransitionTime": rfc3339(self.now()),
            }
        )
        run["status"]["conditions"] = conditions
        self.bump(run)

    def run_of(self, request_name: str) -> Optional[dict]:
        """The stored Run a Request owns, if any (the newest)."""

        with self._lock:
            request = self.requests.get(request_name)
            if request is None:
                return None
            owned = [
                run
                for run in self.runs.values()
                if any(
                    owner.get("uid") == request["metadata"]["uid"]
                    for owner in run["metadata"].get("ownerReferences") or []
                )
            ]
            return copy.deepcopy(owned[-1]) if owned else None


Evaluator = Callable[[dict, Mapping[str, float]], Optional[Mapping[str, Any]]]


class FakeDgdrOperator:
    """Drives every Request of a ``FakeRetuneKube`` one phase per ``step()``.

    Phases: accept (a Run with no status, as before its Job exists), start
    (the lowered input, jobRef and startTime; the operator records the
    input together with jobRef), complete (one Candidate per router point
    the input spans, metrics from ``evaluate(request, point_public)``;
    ``None`` leaves the point unpublished, as AISimulate drops an
    infeasible point). ``hold`` keeps
    named Requests in their current phase; ``fail_runs`` maps a Request
    name to the ``Completed`` reason its Run ends with. ``mutate_input``
    edits a Run's lowered input and ``mutate_parameters`` a Candidate's
    parameters (both get the Request name), to simulate version skew.
    ``max_phase`` stops every Request at ``accepted`` or ``started``.
    """

    def __init__(
        self,
        kube: FakeRetuneKube,
        *,
        evaluate: Optional[Evaluator] = None,
        runtime_image: str = DEFAULT_RUNTIME_IMAGE,
        dynamo_version: str = DEFAULT_DYNAMO_VERSION,
        mutate_input: Optional[Callable[[str, dict], None]] = None,
        mutate_parameters: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.kube = kube
        self.evaluate: Evaluator = evaluate or (
            lambda _request, point: default_metrics(point)
        )
        self.runtime_image = runtime_image
        self.dynamo_version = dynamo_version
        self.mutate_input = mutate_input
        self.mutate_parameters = mutate_parameters
        self.max_phase: Optional[Literal["accepted", "started"]] = None
        self.hold: set[str] = set()
        self.fail_runs: dict[str, str] = {}
        self.steps = 0

    def step(self) -> int:
        """Advance every Request one phase; returns how many moved."""

        moved = 0
        with self.kube._lock:
            self.steps += 1
            for name in list(self.kube.requests):
                if name in self.hold:
                    continue
                run = self.kube.run_of(name)
                if run is None:
                    hook = self.mutate_input
                    self.kube.accept_request(
                        name,
                        runtime_image=self.runtime_image,
                        dynamo_version=self.dynamo_version,
                        mutate_input=(
                            None if hook is None else functools.partial(hook, name)
                        ),
                    )
                    moved += 1
                    continue
                run_name = run["metadata"]["name"]
                completed = next(
                    (
                        condition
                        for condition in run["status"].get("conditions", [])
                        if condition.get("type") == "Completed"
                    ),
                    None,
                )
                if completed is not None and completed.get("status") == "True":
                    continue
                if self.max_phase == "accepted":
                    continue
                if not run["status"].get("startTime"):
                    self.kube.start_run(run_name)
                    moved += 1
                    continue
                if self.max_phase == "started":
                    continue
                reason = self.fail_runs.get(name)
                if reason is not None:
                    self.kube.complete_run(run_name, reason=reason, message="injected")
                    moved += 1
                    continue
                effective = run["status"]["aisimulate"]["input"]
                request = self.kube.requests[name]
                published = []
                for point in router_points(effective):
                    metrics = self.evaluate(copy.deepcopy(request), point)
                    if metrics is not None:
                        parameters = candidate_parameters(effective, point)
                        if self.mutate_parameters is not None:
                            self.mutate_parameters(name, parameters)
                        published.append((parameters, metrics))
                self.kube.complete_run(run_name, candidates=published)
                moved += 1
        return moved

    def run_to_completion(self, max_steps: int = 10) -> None:
        for _ in range(max_steps):
            if not self.step():
                return
