# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend HTTP vs Runtime EPP token-parity helpers for issue #16592.

This module launches the standard ``/epp`` binary (not standalone / MockPicker)
against a TEST-ONLY loopback Kubernetes API. The fixture persists metadata the
runtime actually PATCHes and never invents token results, model cards, or
router state.

Standard ``/epp`` hardcodes gRPC 9002 and health 9003. These helpers are not
host-network portable concurrent coverage: they require a dedicated network
container, serial module execution, a whole-session lock, and a fail-fast
occupancy check. They do not kill an incumbent process and do not add
production port overrides.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import socket
import sys
import tempfile
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import parse_qs, urlparse

import grpc
import requests
from filelock import FileLock
from grpc_tools import protoc

from tests.router.helper import generate_random_suffix
from tests.router.mocker_process import MockerProcess
from tests.router.router_process import (
    FrontendRouterProcess,
    RuntimeEppProcess,
    fetch_loopback_tls_cert,
)
from tests.utils.constants import DynamoPortRange
from tests.utils.http_checks import check_http_ok
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)

EPP_GRPC_PORT = 9002
EPP_HEALTH_PORT = 9003
EPP_BINARY_ENV = "DYN_EPP_BINARY"
DEFAULT_EPP_BINARY = "/epp"
EPP_SESSION_LOCK_PATH = Path(tempfile.gettempdir()) / "dynamo-epp-parity-9002.lock"
MAX_KUBE_BODY_BYTES = 1_048_576
WATCH_HISTORY_BOUND = 256
PODINFO_DIR = Path("/etc/podinfo")
EXT_PROC_PROTO_ROOT = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "inference-gateway"
    / "ext-proc"
    / "proto"
)
WORKER_CONTAINER_NAME = "mocker"
WORKER_A_CONTAINER_NAME = "mocker-a"
WORKER_B_CONTAINER_NAME = "mocker-b"
PROVIDER_CONTAINER_NAME = "kv-indexer"
FRONTEND_CONTAINER_NAME = "frontend"
EPP_CONTAINER_NAME = "epp"
KV_INDEXER_QUERY_ENDPOINT = "kv_indexer_query"
WORKER_INSTANCE_ID_HEADER = "x-dynamo-worker-instance-id"
DP_RANK_HEADER = "x-dynamo-dp-rank"
ROUTING_MODE_HEADER = "x-dynamo-routing-mode"
DESTINATION_ENDPOINT_HEADER = "x-gateway-destination-endpoint"
AGGREGATED_ROUTING_MODE = "aggregated"
READINESS_DEADLINE_S = 45.0
PROVIDER_QUERY_SETTLE_S = 0.25
WORKER_BLOCK_SIZE = 16
WORKER_DP_SIZE = 1
U64_MAX = (1 << 64) - 1
U32_MAX = (1 << 32) - 1
FULL_DUPLEX_STREAMED = 4
PROCESS_TIMEOUT_S = 240.0
EXT_PROC_TIMEOUT_S = 45.0
FRONTEND_QUERY_TIMEOUT_S = 45.0
TLS_HANDSHAKE_TIMEOUT_S = 5.0
KUBE_API_MANAGER_SHUTDOWN_TIMEOUT_S = 5.0
INITIAL_EVENTS_END_ANNOTATION = "k8s.io/initial-events-end"
_SCOPED_CLEAR_ENV = (
    "NATS_SERVER",
    "ETCD_ENDPOINTS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "GRPC_PROXY",
    "grpc_proxy",
)
SYNTHETIC_EXT_PROC_SSE = (
    "data: {"
    '"id":"epp-parity-synthetic",'
    '"object":"chat.completion.chunk",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"x-epp-parity":"synthetic-lifecycle-completion-not-generation"'
    "}\n\n"
    "data: [DONE]\n\n"
)


def epp_binary_path() -> str:
    configured = os.environ.get(EPP_BINARY_ENV, "").strip()
    return configured or DEFAULT_EPP_BINARY


def _port_is_occupied(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def acquire_epp_session_lock() -> FileLock:
    """Exclusive session lock for the hardcoded /epp listener ports.

    Fail-fast if another session already holds the lock. Does not kill or
    displace an incumbent process.
    """
    lock = FileLock(str(EPP_SESSION_LOCK_PATH))
    if not lock.acquire(blocking=False):
        raise RuntimeError(
            "EPP frontend parity requires exclusive serial execution in a "
            f"dedicated network container; session lock is held ({EPP_SESSION_LOCK_PATH}). "
            "This is not host-network portable concurrent coverage."
        )
    return lock


def assert_epp_ports_free() -> None:
    """Fail fast if hardcoded /epp ports 9002/9003 are already bound.

    Does not kill the occupant. Serial dedicated-container execution is required.
    """
    busy = [
        port for port in (EPP_GRPC_PORT, EPP_HEALTH_PORT) if _port_is_occupied(port)
    ]
    if busy:
        raise RuntimeError(
            "Standard /epp hardcodes gRPC 9002 and health 9003; refusing to start "
            f"while occupied: {busy}. The incumbent process was not killed. "
            "Run this module serially in its own network container."
        )


@dataclass(frozen=True)
class PodIdentity:
    """Identity the real runtime PodInfo will read (file first, then env)."""

    pod_name: str
    pod_uid: str
    pod_namespace: str
    source: str


@dataclass(frozen=True)
class RouteObservation:
    """Typed first-selection observation from Frontend SSE or EPP headers."""

    worker_id: int
    dp_rank: int
    token_ids: list[int]
    routing_mode: str | None = None
    destination: str | None = None


def resolve_pod_identity() -> PodIdentity:
    """Prefer mounted /etc/podinfo, matching runtime PodInfo priority."""

    def read_file_or_none(name: str) -> str | None:
        path = PODINFO_DIR / name
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    file_name = read_file_or_none("pod_name")
    file_uid = read_file_or_none("pod_uid")
    file_namespace = read_file_or_none("pod_namespace")
    if file_name or file_uid or file_namespace:
        suffix = generate_random_suffix()
        return PodIdentity(
            pod_name=file_name or f"epp-parity-{suffix}",
            pod_uid=file_uid or str(uuid.uuid4()),
            pod_namespace=file_namespace or f"epp-parity-{suffix}",
            source="podinfo",
        )
    suffix = generate_random_suffix()
    return PodIdentity(
        pod_name=f"epp-parity-{suffix}",
        pod_uid=str(uuid.uuid4()),
        pod_namespace=f"epp-parity-{suffix}",
        source="env",
    )


def write_loopback_kubeconfig(
    path: Path, api_host: str, api_port: int, namespace: str
) -> Path:
    """Write a kubeconfig that can only reach this fixture's loopback listener."""
    if api_host not in {"127.0.0.1", "localhost"}:
        raise ValueError("loopback kubeconfig host must be 127.0.0.1 or localhost")
    path.write_text(
        (
            "apiVersion: v1\n"
            "kind: Config\n"
            "clusters:\n"
            "- name: epp-parity\n"
            "  cluster:\n"
            f"    server: http://{api_host}:{api_port}\n"
            "contexts:\n"
            "- name: epp-parity\n"
            "  context:\n"
            "    cluster: epp-parity\n"
            "    user: epp-parity\n"
            f"    namespace: {namespace}\n"
            "current-context: epp-parity\n"
            "users:\n"
            "- name: epp-parity\n"
            "  user: {}\n"
        ),
        encoding="utf-8",
    )
    return path


def discovery_process_env(
    kubeconfig: Path,
    identity: PodIdentity,
    container_name: str,
) -> dict[str, str | None]:
    """Kubernetes discovery identity for a child process.

    Clears cluster service host/port so children cannot reach a real API. None
    values are deletions applied by the process wrappers.
    """
    return {
        "KUBECONFIG": str(kubeconfig),
        "KUBERNETES_SERVICE_HOST": None,
        "KUBERNETES_SERVICE_PORT": None,
        "KUBERNETES_SERVICE_PORT_HTTPS": None,
        "DYN_DISCOVERY_BACKEND": "kubernetes",
        "DYN_KUBE_DISCOVERY_MODE": "container",
        "CONTAINER_NAME": container_name,
        "POD_NAME": identity.pod_name,
        "POD_UID": identity.pod_uid,
        "POD_NAMESPACE": identity.pod_namespace,
    }


def loopback_process_env(
    kubeconfig: Path,
    identity: PodIdentity,
    container_name: str,
) -> dict[str, str | None]:
    """Identity plus TCP/ZMQ and scoped clears for the loopback children."""
    env = discovery_process_env(kubeconfig, identity, container_name)
    env["DYN_REQUEST_PLANE"] = "tcp"
    env["DYN_EVENT_PLANE"] = "zmq"
    for name in _SCOPED_CLEAR_ENV:
        env[name] = None
    return env


@dataclass
class _WatchClient:
    kind: str
    start_rv: int
    query: dict[str, list[str]]
    events: queue.Queue[dict[str, Any] | None]
    closed: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class _HistoryEvent:
    kind: str
    event_type: str
    obj: dict[str, Any]


class LoopbackKubeAPI:
    """Minimum Pod + DynamoWorkerMetadata API used by runtime kube discovery.

    Accepts only fixed resource paths/verbs/namespaces. Unknown requests are
    recorded as fixture failures. The fixture does not proxy and does not
    write to a real cluster.
    """

    def __init__(self, namespace: str, port: int) -> None:
        self.namespace = namespace
        self.port = port
        self.host = "127.0.0.1"
        self.failures: list[str] = []
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._resource_version = 1
        self._pods: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._history: list[_HistoryEvent] = []
        self._watchers: list[_WatchClient] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """Bind the loopback HTTP server on this fixture's allocated port."""
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: object) -> None:
                logger.debug("loopback kube api: " + fmt, *args)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                api._handle(self, "GET")

            def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                api._handle(self, "PATCH")

            def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                api._handle(self, "DELETE")

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                api._handle(self, "POST")

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="epp-parity-kube-api",
            daemon=True,
        )
        self._thread.start()
        logger.info("Loopback Kubernetes API listening on %s", self.url)

    def stop(self) -> None:
        """Close watchers and the HTTP server; does not touch a real cluster."""
        self._closed.set()
        with self._cv:
            for watcher in list(self._watchers):
                watcher.closed.set()
                watcher.events.put(None)
            self._watchers.clear()
            self._cv.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def upsert_worker_pod(
        self,
        *,
        identity: PodIdentity,
        dynamo_namespace: str,
        container_name: str,
        http_port: int,
    ) -> None:
        """Publish the ready worker Pod for discovery and owner-UID join."""
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": identity.pod_name,
                "namespace": identity.pod_namespace,
                "uid": identity.pod_uid,
                "labels": {
                    "nvidia.com/dynamo-namespace": dynamo_namespace,
                    "nvidia.com/dynamo-component-class": "worker",
                    "nvidia.com/dynamo-discovery-backend": "kubernetes",
                    "nvidia.com/dynamo-discovery-enabled": "true",
                },
            },
            "spec": {
                "containers": [
                    {
                        "name": container_name,
                        "ports": [
                            {
                                "name": "http",
                                "containerPort": http_port,
                                "protocol": "TCP",
                            }
                        ],
                    }
                ]
            },
            "status": {
                "phase": "Running",
                "podIP": "127.0.0.1",
                "containerStatuses": [
                    {
                        "name": container_name,
                        "ready": True,
                        "containerID": f"containerd://epp-parity-{container_name}",
                        "image": "mocker:test",
                        "imageID": "containerd://epp-parity-image",
                    }
                ],
            },
        }
        self._store_pod(pod, event_type="ADDED")

    def add_ready_container(
        self,
        *,
        identity: PodIdentity,
        dynamo_namespace: str,
        container_name: str,
        http_port: int,
    ) -> None:
        """Add one Ready container without replacing other containers on the Pod."""
        container_spec = {
            "name": container_name,
            "ports": [
                {
                    "name": "http",
                    "containerPort": http_port,
                    "protocol": "TCP",
                }
            ],
        }
        container_status = {
            "name": container_name,
            "ready": True,
            "containerID": f"containerd://epp-parity-{container_name}",
            "image": "mocker:test",
            "imageID": "containerd://epp-parity-image",
        }
        with self._cv:
            existing = self._pods.get(identity.pod_name)
            if existing is None:
                self.upsert_worker_pod(
                    identity=identity,
                    dynamo_namespace=dynamo_namespace,
                    container_name=container_name,
                    http_port=http_port,
                )
                return
            stored = json.loads(json.dumps(existing))
            containers = stored.setdefault("spec", {}).setdefault("containers", [])
            statuses = stored.setdefault("status", {}).setdefault(
                "containerStatuses", []
            )
            if any(item.get("name") == container_name for item in containers):
                raise AssertionError(
                    f"fixture Pod already has container {container_name!r}; "
                    "refusing to overwrite an existing Ready container"
                )
            containers.append(container_spec)
            statuses.append(container_status)
            labels = stored.setdefault("metadata", {}).setdefault("labels", {})
            labels.setdefault("nvidia.com/dynamo-namespace", dynamo_namespace)
            self._store_pod(stored, event_type="MODIFIED")

    def listed_pods(self) -> list[dict[str, Any]]:
        with self._lock:
            return [json.loads(json.dumps(pod)) for pod in self._pods.values()]

    def listed_metadata(self) -> list[dict[str, Any]]:
        with self._lock:
            return [json.loads(json.dumps(obj)) for obj in self._metadata.values()]

    def get_failures(self) -> list[str]:
        """Return a copy of recorded fixture-API failures."""
        with self._lock:
            return list(self.failures)

    def _next_rv(self) -> str:
        self._resource_version += 1
        return str(self._resource_version)

    def _store_pod(self, pod: dict[str, Any], event_type: str) -> dict[str, Any]:
        """Deep-copy and persist a Pod, assigning RV/UID, then notify watchers."""
        with self._cv:
            stored = json.loads(json.dumps(pod))
            stored.setdefault("metadata", {})["resourceVersion"] = self._next_rv()
            stored["metadata"].setdefault(
                "uid", self._uid_for("pod", stored["metadata"]["name"])
            )
            name = stored["metadata"]["name"]
            existed = name in self._pods
            self._pods[name] = stored
            self._publish("Pod", event_type if not existed else "MODIFIED", stored)
            return stored

    def _store_metadata(self, obj: dict[str, Any], event_type: str) -> dict[str, Any]:
        """Deep-copy and persist DynamoWorkerMetadata in this fixture namespace."""
        with self._cv:
            stored = json.loads(json.dumps(obj))
            metadata = stored.setdefault("metadata", {})
            metadata["namespace"] = self.namespace
            metadata["resourceVersion"] = self._next_rv()
            metadata.setdefault("uid", self._uid_for("dwm", metadata["name"]))
            metadata.setdefault("generation", 1)
            name = metadata["name"]
            existed = name in self._metadata
            self._metadata[name] = stored
            self._publish(
                "DynamoWorkerMetadata",
                event_type if not existed else "MODIFIED",
                stored,
            )
            return stored

    def _uid_for(self, kind: str, name: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{kind}:{self.namespace}:{name}"))

    def _publish(self, kind: str, event_type: str, obj: dict[str, Any]) -> None:
        """Append a bounded history event and fan it to matching live watchers.

        Caller holds self._cv. Persist then fan out so LIST-to-WATCH cannot drop
        a mutation that happened before the watcher registered. History older
        than WATCH_HISTORY_BOUND is dropped.
        """
        event = {"type": event_type, "object": obj}
        self._history.append(_HistoryEvent(kind=kind, event_type=event_type, obj=obj))
        overflow = len(self._history) - WATCH_HISTORY_BOUND
        if overflow > 0:
            del self._history[:overflow]
        labels = obj.get("metadata", {}).get("labels") or {}
        for watcher in list(self._watchers):
            if watcher.closed.is_set() or watcher.kind != kind:
                continue
            if not _matches_labels(labels, watcher.query):
                continue
            watcher.events.put(event)
        self._cv.notify_all()

    def _record_failure(self, method: str, path: str, reason: str) -> None:
        detail = f"{method} {path}: {reason}"
        self.failures.append(detail)
        logger.error("Loopback Kubernetes API fixture failure: %s", detail)

    def _handle(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        """Route one request; unknown paths and handler bugs are recorded failures."""
        parsed = urlparse(handler.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            body = self._read_body(handler)
        except ValueError as exc:
            self._record_failure(method, path, str(exc))
            self._write_status(handler, 413, {"message": str(exc), "code": 413})
            return

        try:
            handled = self._dispatch(handler, method, path, query, body)
        except Exception as exc:  # unexpected fixture bug, not a kube proxy
            self._record_failure(method, path, f"handler error: {exc}")
            self._write_status(
                handler, 500, {"message": "fixture handler error", "code": 500}
            )
            return
        if handled:
            return
        self._record_failure(method, path, "unknown request")
        self._write_status(
            handler,
            404,
            {
                "kind": "Status",
                "apiVersion": "v1",
                "status": "Failure",
                "reason": "NotFound",
                "code": 404,
                "message": "loopback fixture does not serve this path",
            },
        )

    def _read_body(self, handler: BaseHTTPRequestHandler) -> bytes:
        """Read at most MAX_KUBE_BODY_BYTES; oversized or invalid length is 413."""
        length_header = handler.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_KUBE_BODY_BYTES:
            raise ValueError("request body exceeds fixture size bound")
        if length == 0:
            return b""
        return handler.rfile.read(length)

    def _dispatch(
        self,
        handler: BaseHTTPRequestHandler,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: bytes,
    ) -> bool:
        """Serve only Pod and DynamoWorkerMetadata verbs in this fixture namespace.

        Returns False for unknown paths so _handle records a fixture failure.
        """
        pods_prefix = f"/api/v1/namespaces/{self.namespace}/pods"
        dwm_prefix = f"/apis/nvidia.com/v1alpha1/namespaces/{self.namespace}/dynamoworkermetadatas"
        if path == pods_prefix:
            if method != "GET":
                return False
            if _is_watch(query):
                self._watch(handler, "Pod", query)
                return True
            self._write_json(handler, 200, self._list_pods(query, handler.headers))
            return True
        if path.startswith(pods_prefix + "/"):
            name = path[len(pods_prefix) + 1 :]
            if "/" in name or not name:
                return False
            if method != "GET":
                return False
            with self._lock:
                pod = self._pods.get(name)
            if pod is None:
                self._write_status(handler, 404, _not_found("pods", name))
                return True
            self._write_json(handler, 200, pod)
            return True
        if path == dwm_prefix:
            if method != "GET":
                return False
            if _is_watch(query):
                self._watch(handler, "DynamoWorkerMetadata", query)
                return True
            self._write_json(handler, 200, self._list_metadata(query, handler.headers))
            return True
        if path.startswith(dwm_prefix + "/"):
            name = path[len(dwm_prefix) + 1 :]
            if "/" in name or not name:
                return False
            if method == "GET":
                with self._lock:
                    obj = self._metadata.get(name)
                if obj is None:
                    self._write_status(
                        handler, 404, _not_found("dynamoworkermetadatas", name)
                    )
                    return True
                self._write_json(handler, 200, obj)
                return True
            if method == "DELETE":
                self._delete_metadata(name)
                self._write_status(
                    handler, 200, {"kind": "Status", "status": "Success", "code": 200}
                )
                return True
            if method == "PATCH":
                try:
                    payload = json.loads(body.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    self._record_failure(method, path, "PATCH body is not JSON")
                    self._write_status(
                        handler, 400, {"message": "invalid JSON", "code": 400}
                    )
                    return True
                if not isinstance(payload, dict):
                    self._record_failure(method, path, "PATCH body is not an object")
                    self._write_status(
                        handler, 400, {"message": "invalid JSON object", "code": 400}
                    )
                    return True
                payload.setdefault("apiVersion", "nvidia.com/v1alpha1")
                payload.setdefault("kind", "DynamoWorkerMetadata")
                metadata = payload.setdefault("metadata", {})
                metadata["name"] = name
                stored = self._store_metadata(payload, "ADDED")
                self._write_json(handler, 200, stored)
                return True
        return False

    def _list_pods(
        self, query: dict[str, list[str]], headers: Mapping[str, str]
    ) -> dict[str, Any]:
        with self._lock:
            items = [
                pod
                for pod in self._pods.values()
                if _matches_labels(pod.get("metadata", {}).get("labels") or {}, query)
            ]
            rv = str(self._resource_version)
        items, continue_token = _paginate(items, query)
        if _wants_partial_meta(headers):
            return _partial_list("Pod", "v1", items, rv, continue_token)
        return {
            "kind": "PodList",
            "apiVersion": "v1",
            "metadata": _list_metadata(rv, continue_token),
            "items": items,
        }

    def _list_metadata(
        self, query: dict[str, list[str]], headers: Mapping[str, str]
    ) -> dict[str, Any]:
        with self._lock:
            items = [
                obj
                for obj in self._metadata.values()
                if _matches_labels(obj.get("metadata", {}).get("labels") or {}, query)
            ]
            rv = str(self._resource_version)
        items, continue_token = _paginate(items, query)
        if _wants_partial_meta(headers):
            return _partial_list(
                "DynamoWorkerMetadata", "nvidia.com/v1alpha1", items, rv, continue_token
            )
        return {
            "kind": "DynamoWorkerMetadataList",
            "apiVersion": "nvidia.com/v1alpha1",
            "metadata": _list_metadata(rv, continue_token),
            "items": items,
        }

    def _delete_metadata(self, name: str) -> None:
        with self._cv:
            obj = self._metadata.pop(name, None)
            if obj is None:
                return
            deleted = json.loads(json.dumps(obj))
            deleted.setdefault("metadata", {})["resourceVersion"] = self._next_rv()
            self._publish("DynamoWorkerMetadata", "DELETED", deleted)

    def _watch(
        self,
        handler: BaseHTTPRequestHandler,
        kind: str,
        query: dict[str, list[str]],
    ) -> None:
        """Stream chunked watch events until timeout, close, or client disconnect.

        sendInitialEvents emits a snapshot plus bookmark; otherwise history after
        resourceVersion is replayed. Callers that outrun WATCH_HISTORY_BOUND can
        miss events.
        """
        start_rv = _requested_rv(query)
        send_initial = _query_flag(query, "sendInitialEvents")
        watcher = _WatchClient(
            kind=kind,
            start_rv=start_rv,
            query=query,
            events=queue.Queue(),
        )
        snapshot: list[dict[str, Any]] = []
        replay: list[dict[str, Any]] = []
        with self._cv:
            store = self._pods if kind == "Pod" else self._metadata
            current_rv = self._resource_version
            if send_initial:
                snapshot = [
                    obj
                    for obj in store.values()
                    if _matches_labels(
                        obj.get("metadata", {}).get("labels") or {}, query
                    )
                ]
                watcher.start_rv = current_rv
            else:
                for item in self._history:
                    if item.kind != kind:
                        continue
                    if _object_rv(item.obj) <= start_rv:
                        continue
                    if not _matches_labels(
                        item.obj.get("metadata", {}).get("labels") or {}, query
                    ):
                        continue
                    replay.append({"type": item.event_type, "object": item.obj})
            self._watchers.append(watcher)
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        try:
            if send_initial:
                for obj in snapshot:
                    _write_chunk(handler, {"type": "ADDED", "object": obj})
                bookmark = {
                    "apiVersion": "v1" if kind == "Pod" else "nvidia.com/v1alpha1",
                    "kind": kind,
                    "metadata": {
                        "resourceVersion": str(current_rv),
                        "annotations": {INITIAL_EVENTS_END_ANNOTATION: "true"},
                    },
                }
                _write_chunk(handler, {"type": "BOOKMARK", "object": bookmark})
            else:
                for event in replay:
                    _write_chunk(handler, event)
            timeout_s = _watch_timeout_s(query)
            deadline = time.monotonic() + timeout_s
            while not self._closed.is_set() and time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    event = watcher.events.get(timeout=min(1.0, max(remaining, 0.05)))
                except queue.Empty:
                    continue
                if event is None:
                    break
                obj = event.get("object") or {}
                if not _matches_labels(
                    obj.get("metadata", {}).get("labels") or {}, query
                ):
                    continue
                event_rv = _object_rv(obj)
                if event_rv <= watcher.start_rv:
                    continue
                _write_chunk(handler, event)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Watch client disconnected for %s", kind)
        finally:
            try:
                _end_chunks(handler)
            except (BrokenPipeError, ConnectionResetError):
                pass
            watcher.closed.set()
            with self._lock:
                if watcher in self._watchers:
                    self._watchers.remove(watcher)

    def _write_json(
        self, handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
    ) -> None:
        raw = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)

    def _write_status(
        self, handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
    ) -> None:
        payload.setdefault("kind", "Status")
        payload.setdefault("apiVersion", "v1")
        self._write_json(handler, status, payload)


def _is_watch(query: dict[str, list[str]]) -> bool:
    return _query_flag(query, "watch")


def _query_flag(query: dict[str, list[str]], name: str) -> bool:
    values = query.get(name, [])
    if not values:
        return False
    return values[0].lower() in {"1", "true", "yes"}


def _requested_rv(query: dict[str, list[str]]) -> int:
    raw = query.get("resourceVersion", ["0"])[0]
    try:
        return int(raw or "0")
    except ValueError:
        return 0


def _object_rv(obj: Mapping[str, Any]) -> int:
    raw = obj.get("metadata", {}).get("resourceVersion", "0")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _watch_timeout_s(query: dict[str, list[str]]) -> float:
    """Watch deadline from timeoutSeconds; invalid values default to 30s, minimum 1s."""
    raw = query.get("timeoutSeconds", ["30"])[0]
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _matches_labels(labels: Mapping[str, str], query: dict[str, list[str]]) -> bool:
    """Require every labelSelector key=value pair; empty selector matches all."""
    selector = query.get("labelSelector", [""])[0]
    if not selector:
        return True
    for part in selector.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            return False
        key, value = item.split("=", 1)
        if labels.get(key.strip()) != value.strip():
            return False
    return True


def _paginate(
    items: list[dict[str, Any]], query: dict[str, list[str]]
) -> tuple[list[dict[str, Any]], str | None]:
    """Page items by limit/continue.

    Invalid or nonpositive limit returns the full unpaged list. Invalid
    continue resets the start index to zero and keeps a positive limit.
    """
    try:
        limit = int(query.get("limit", ["0"])[0] or 0)
    except ValueError:
        limit = 0
    try:
        start = int(query.get("continue", ["0"])[0] or 0)
    except ValueError:
        start = 0
    if limit <= 0:
        return items, None
    window = items[start : start + limit]
    nxt = start + limit
    continue_token = str(nxt) if nxt < len(items) else None
    return window, continue_token


def _wants_partial_meta(headers: Mapping[str, str]) -> bool:
    accept = headers.get("Accept") or headers.get("accept") or ""
    return "PartialObjectMetadataList" in accept


def _list_metadata(rv: str, continue_token: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"resourceVersion": rv}
    if continue_token is not None:
        metadata["continue"] = continue_token
    return metadata


def _partial_list(
    kind: str,
    api_version: str,
    items: list[dict[str, Any]],
    rv: str,
    continue_token: str | None,
) -> dict[str, Any]:
    return {
        "kind": "PartialObjectMetadataList",
        "apiVersion": "meta.k8s.io/v1",
        "metadata": _list_metadata(rv, continue_token),
        "items": [
            {
                "apiVersion": "meta.k8s.io/v1",
                "kind": "PartialObjectMetadata",
                "metadata": item.get("metadata", {}),
            }
            for item in items
        ],
    }


def _not_found(resource: str, name: str) -> dict[str, Any]:
    return {
        "kind": "Status",
        "apiVersion": "v1",
        "status": "Failure",
        "reason": "NotFound",
        "code": 404,
        "message": f"{resource} {name!r} not found",
    }


def _write_chunk(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    """Write one chunked JSON watch event; caller owns the handler socket."""
    data = json.dumps(payload).encode("utf-8") + b"\n"
    handler.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
    handler.wfile.flush()


def _end_chunks(handler: BaseHTTPRequestHandler) -> None:
    """Terminate the chunked watch body; caller handles client disconnect."""
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()


def compile_ext_proc_protos(output_dir: Path) -> None:
    """Compile vendored ext_proc protos into tmp_path. Never writes the source tree."""
    include = Path(sys.modules["grpc_tools"].__file__).resolve().parent / "_proto"
    output_dir.mkdir(parents=True, exist_ok=True)
    proto_files = [
        "envoy/config/core/v3/base.proto",
        "envoy/extensions/filters/http/ext_proc/v3/processing_mode.proto",
        "envoy/type/v3/http_status.proto",
        "envoy/service/ext_proc/v3/external_processor.proto",
    ]
    args = [
        "grpc_tools.protoc",
        f"-I{EXT_PROC_PROTO_ROOT}",
        f"-I{include}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        *proto_files,
    ]
    rc = protoc.main(args)
    if rc != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {rc}")
    for directory in (
        output_dir / "envoy",
        output_dir / "envoy" / "config",
        output_dir / "envoy" / "config" / "core",
        output_dir / "envoy" / "config" / "core" / "v3",
        output_dir / "envoy" / "extensions",
        output_dir / "envoy" / "extensions" / "filters",
        output_dir / "envoy" / "extensions" / "filters" / "http",
        output_dir / "envoy" / "extensions" / "filters" / "http" / "ext_proc",
        output_dir / "envoy" / "extensions" / "filters" / "http" / "ext_proc" / "v3",
        output_dir / "envoy" / "service",
        output_dir / "envoy" / "service" / "ext_proc",
        output_dir / "envoy" / "service" / "ext_proc" / "v3",
        output_dir / "envoy" / "type",
        output_dir / "envoy" / "type" / "v3",
    ):
        (directory / "__init__.py").touch()


class _ExtProcModules:
    """Compile vendored ext_proc stubs into tmp_path and load them for one stack.

    Generated files never enter the source tree. close() drops the temporary
    sys.path entry; it does not delete tmp_path (pytest owns that).
    """

    def __init__(self, output_dir: Path) -> None:
        compile_ext_proc_protos(output_dir)
        self._output_dir = str(output_dir)
        if self._output_dir not in sys.path:
            sys.path.insert(0, self._output_dir)
        from envoy.service.ext_proc.v3 import (  # type: ignore[import-not-found]
            external_processor_pb2,
            external_processor_pb2_grpc,
        )

        self.pb2 = external_processor_pb2
        self.grpc = external_processor_pb2_grpc

    def close(self) -> None:
        try:
            sys.path.remove(self._output_dir)
        except ValueError:
            return


class _RequestStream:
    """Thread-safe iterator the ext_proc stub consumes; close() ends the stream."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()

    def send(self, message: Any) -> None:
        self._queue.put(message)

    def close(self) -> None:
        self._queue.put(None)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        item = self._queue.get()
        if item is None:
            raise StopIteration
        return item


def _loopback_epp_channel(timeout_s: float) -> grpc.Channel:
    """Open a TLS channel to 127.0.0.1:9002 trusting only this fetch's peer cert.

    Disables grpc HTTP proxy for this channel only. Caller must close the channel.
    """
    pem = fetch_loopback_tls_cert(
        "127.0.0.1", EPP_GRPC_PORT, min(timeout_s, TLS_HANDSHAKE_TIMEOUT_S)
    ).encode("ascii")
    credentials = grpc.ssl_channel_credentials(root_certificates=pem)
    channel = grpc.secure_channel(
        f"127.0.0.1:{EPP_GRPC_PORT}",
        credentials,
        options=(
            ("grpc.ssl_target_name_override", "localhost"),
            ("grpc.default_authority", "localhost"),
            # This channel only. Child env and requests.trust_env do not
            # control parent-process grpcio proxy inheritance.
            ("grpc.enable_http_proxy", 0),
        ),
    )
    grpc.channel_ready_future(channel).result(
        timeout=min(timeout_s, TLS_HANDSHAKE_TIMEOUT_S)
    )
    return channel


def _header_map(pb2: Any, headers: list[tuple[str, str]]) -> Any:
    from envoy.config.core.v3 import base_pb2  # type: ignore[import-not-found]

    return base_pb2.HeaderMap(
        headers=[
            base_pb2.HeaderValue(key=key, value=value, raw_value=value.encode("utf-8"))
            for key, value in headers
        ]
    )


def _request_body_mutation_status(response: Any) -> str | None:
    """Classify a ProcessingResponse RequestBody mutation, or None if absent."""
    if not response.HasField("request_body"):
        return None
    body = response.request_body
    if not body.HasField("response"):
        return "empty"
    mutation = body.response.body_mutation
    if mutation.HasField("streamed_response"):
        if mutation.streamed_response.end_of_stream:
            return "terminal_streamed"
        return "nonterminal_streamed"
    if mutation.HasField("body"):
        return "body"
    if mutation.HasField("clear_body"):
        return "clear_body"
    return "empty"


def _extract_unique_terminal_request_body(pb2: Any, responses: list[Any]) -> bytes:
    """Require exactly one terminal streamed RequestBody on the complete stream."""
    bodies: list[Any] = []
    for response in responses:
        if response.HasField("immediate_response"):
            details = response.immediate_response.details
            raise AssertionError(
                f"EPP returned ImmediateResponse instead of a body mutation: {details}"
            )
        if response.HasField("request_body"):
            bodies.append(response)
    if not bodies:
        raise AssertionError(
            "EPP stream completed without a RequestBody token mutation"
        )
    if len(bodies) != 1:
        raise AssertionError(
            f"EPP stream contained {len(bodies)} RequestBody mutations; expected exactly one"
        )
    status = _request_body_mutation_status(bodies[0])
    mutation = bodies[0].request_body.response.body_mutation
    if status == "terminal_streamed":
        return bytes(mutation.streamed_response.body)
    if status == "body":
        raise AssertionError(
            "EPP RequestBody mutation used a non-streamed body instead of a "
            "terminal streamed_response"
        )
    if status == "clear_body":
        raise AssertionError("EPP RequestBody mutation cleared the body")
    if status == "nonterminal_streamed":
        raise AssertionError("EPP RequestBody streamed_response is not terminal")
    raise AssertionError("EPP RequestBody mutation is empty")


def decode_epp_token_data(body: bytes) -> list[int]:
    """Parse nvext.token_data from a mutated JSON body.

    Rejects non-JSON, missing nvext, empty token_data, and non-integer tokens.
    """
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssertionError("EPP mutated body is not JSON") from exc
    if not isinstance(parsed, dict):
        raise AssertionError("EPP mutated body is not a JSON object")
    nvext = parsed.get("nvext")
    if not isinstance(nvext, dict):
        raise AssertionError("EPP mutated body is missing nvext object")
    token_data = nvext.get("token_data")
    if not isinstance(token_data, list) or not token_data:
        raise AssertionError("EPP nvext.token_data is absent or empty")
    if not all(
        isinstance(token, int) and not isinstance(token, bool) for token in token_data
    ):
        raise AssertionError("EPP nvext.token_data contains a non-integer")
    return [int(token) for token in token_data]


def _header_text(header: Any) -> str:
    raw = bytes(header.raw_value) if header.raw_value else b""
    if raw:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AssertionError("EPP routing header raw_value is not UTF-8") from exc
    value = header.value
    if not isinstance(value, str):
        raise AssertionError("EPP routing header value is not a string")
    return value


def _collect_set_headers(responses: list[Any]) -> dict[str, list[str]]:
    collected: dict[str, list[str]] = {}
    for response in responses:
        if not response.HasField("request_headers"):
            continue
        headers_response = response.request_headers
        if not headers_response.HasField("response"):
            continue
        mutation = headers_response.response.header_mutation
        for option in mutation.set_headers:
            if not option.HasField("header"):
                continue
            header = option.header
            key = header.key.lower()
            collected.setdefault(key, []).append(_header_text(header))
    return collected


def _require_unique_header(collected: dict[str, list[str]], name: str) -> str:
    values = collected.get(name.lower(), [])
    if not values:
        raise AssertionError(f"EPP stream missing unique {name} header mutation")
    unique = set(values)
    if len(values) != 1 or len(unique) != 1:
        raise AssertionError(
            f"EPP stream has duplicate or ambiguous {name} header mutations: {values}"
        )
    value = values[0].strip()
    if not value:
        raise AssertionError(f"EPP {name} header mutation is empty")
    return value


def _parse_unsigned_header(name: str, value: str, maximum: int) -> int:
    if not value.isdigit():
        raise AssertionError(f"EPP {name} header is malformed: {value!r}")
    parsed = int(value)
    if parsed > maximum:
        raise AssertionError(f"EPP {name} header is out of range: {value!r}")
    return parsed


def _parse_u64_header(name: str, value: str) -> int:
    return _parse_unsigned_header(name, value, U64_MAX)


def _parse_u32_header(name: str, value: str) -> int:
    return _parse_unsigned_header(name, value, U32_MAX)


def _extract_unique_routing_headers(responses: list[Any]) -> dict[str, Any]:
    collected = _collect_set_headers(responses)
    worker_id = _parse_u64_header(
        WORKER_INSTANCE_ID_HEADER,
        _require_unique_header(collected, WORKER_INSTANCE_ID_HEADER),
    )
    dp_rank = _parse_u32_header(
        DP_RANK_HEADER, _require_unique_header(collected, DP_RANK_HEADER)
    )
    routing_mode = _require_unique_header(collected, ROUTING_MODE_HEADER)
    if routing_mode != AGGREGATED_ROUTING_MODE:
        raise AssertionError(
            f"EPP {ROUTING_MODE_HEADER} must be {AGGREGATED_ROUTING_MODE!r}, "
            f"got {routing_mode!r}"
        )
    destination = _require_unique_header(collected, DESTINATION_ENDPOINT_HEADER)
    return {
        "worker_id": worker_id,
        "dp_rank": dp_rank,
        "routing_mode": routing_mode,
        "destination": destination,
    }


class ExtProcParityDriver:
    """Drive a complete ext_proc request/response lifecycle against real /epp."""

    def __init__(self, modules: _ExtProcModules) -> None:
        self._modules = modules

    def collect_token_data(
        self,
        payload: dict[str, Any],
        *,
        path: str,
        request_id: str,
        timeout_s: float = EXT_PROC_TIMEOUT_S,
    ) -> list[int]:
        """Drive a full-duplex ext_proc stream and return EPP nvext.token_data.

        Completes the response half/stream lifecycle with SYNTHETIC_EXT_PROC_SSE.
        Rejects ImmediateResponse, missing/non-terminal body mutations, and
        timeout. The channel is closed before return.
        """
        pb2 = self._modules.pb2
        responses = self._collect_responses(
            payload, path=path, request_id=request_id, timeout_s=timeout_s
        )
        return decode_epp_token_data(
            _extract_unique_terminal_request_body(pb2, responses)
        )

    def collect_route_observation(
        self,
        payload: dict[str, Any],
        *,
        path: str,
        request_id: str,
        timeout_s: float = EXT_PROC_TIMEOUT_S,
    ) -> RouteObservation:
        pb2 = self._modules.pb2
        responses = self._collect_responses(
            payload, path=path, request_id=request_id, timeout_s=timeout_s
        )
        token_ids = decode_epp_token_data(
            _extract_unique_terminal_request_body(pb2, responses)
        )
        headers = _extract_unique_routing_headers(responses)
        return RouteObservation(
            worker_id=headers["worker_id"],
            dp_rank=headers["dp_rank"],
            token_ids=token_ids,
            routing_mode=headers["routing_mode"],
            destination=headers["destination"],
        )

    def _collect_responses(
        self,
        payload: dict[str, Any],
        *,
        path: str,
        request_id: str,
        timeout_s: float,
    ) -> list[Any]:
        pb2 = self._modules.pb2
        stub_cls = self._modules.grpc.ExternalProcessorStub
        stream = _RequestStream()
        channel = _loopback_epp_channel(timeout_s)
        responses: list[Any] = []
        try:
            stub = stub_cls(channel)
            response_iter = stub.Process(stream, timeout=timeout_s)
            headers = [
                (":method", "POST"),
                (":path", path),
                (":authority", "127.0.0.1"),
                ("content-type", "application/json"),
                ("x-request-id", request_id),
            ]
            stream.send(
                pb2.ProcessingRequest(
                    protocol_config=pb2.ProtocolConfiguration(
                        request_body_mode=FULL_DUPLEX_STREAMED,
                        response_body_mode=FULL_DUPLEX_STREAMED,
                    ),
                    request_headers=pb2.HttpHeaders(
                        headers=_header_map(pb2, headers),
                        end_of_stream=False,
                    ),
                )
            )
            stream.send(
                pb2.ProcessingRequest(
                    request_body=pb2.HttpBody(
                        body=json.dumps(payload).encode("utf-8"),
                        end_of_stream=True,
                    )
                )
            )
            deadline = time.monotonic() + timeout_s
            saw_terminal_request_body = False
            while not saw_terminal_request_body:
                if time.monotonic() > deadline:
                    raise AssertionError(
                        "timed out waiting for EPP RequestBody mutation"
                    )
                try:
                    response = next(response_iter)
                except StopIteration as exc:
                    raise AssertionError(
                        "EPP stream ended before a token mutation"
                    ) from exc
                responses.append(response)
                if response.HasField("immediate_response"):
                    raise AssertionError(
                        "EPP returned ImmediateResponse: "
                        f"{response.immediate_response.details}"
                    )
                status = _request_body_mutation_status(response)
                if status is None:
                    continue
                if status != "terminal_streamed":
                    raise AssertionError(
                        "EPP first RequestBody mutation is not a terminal "
                        f"streamed_response: {status}"
                    )
                saw_terminal_request_body = True
            stream.send(
                pb2.ProcessingRequest(
                    response_headers=pb2.HttpHeaders(
                        headers=_header_map(
                            pb2,
                            [
                                (":status", "200"),
                                ("content-type", "text/event-stream"),
                            ],
                        ),
                        end_of_stream=False,
                    )
                )
            )
            stream.send(
                pb2.ProcessingRequest(
                    response_body=pb2.HttpBody(
                        body=SYNTHETIC_EXT_PROC_SSE.encode("utf-8"),
                        end_of_stream=True,
                    )
                )
            )
            stream.close()
            while True:
                try:
                    leftover = next(response_iter)
                except StopIteration:
                    break
                responses.append(leftover)
                if leftover.HasField("immediate_response"):
                    raise AssertionError(
                        "EPP returned ImmediateResponse while draining: "
                        f"{leftover.immediate_response.details}"
                    )
            return responses
        finally:
            stream.close()
            channel.close()


def _with_query_instance_id(payload: dict[str, Any]) -> dict[str, Any]:
    annotated = json.loads(json.dumps(payload))
    nvext = dict(annotated.get("nvext") or {})
    annotations = list(nvext.get("annotations") or [])
    if not any(
        isinstance(item, str) and item.startswith("query_instance_id:")
        for item in annotations
    ):
        annotations.append("query_instance_id:")
    nvext["annotations"] = annotations
    annotated["nvext"] = nvext
    return annotated


def _post_frontend_query(
    frontend_url: str,
    payload: dict[str, Any],
    *,
    path: str,
    request_id: str,
    timeout_s: float,
) -> str:
    annotated = _with_query_instance_id(payload)
    with requests.Session() as session:
        session.trust_env = False
        with session.post(
            f"{frontend_url}{path}",
            json=annotated,
            headers={"x-request-id": request_id},
            timeout=timeout_s,
            stream=True,
        ) as response:
            if response.status_code != 200:
                raise AssertionError(
                    f"Frontend query_instance_id returned HTTP {response.status_code}"
                )
            return response.text


def _frontend_sse_data_events(body: str) -> list[str]:
    """Return complete query_instance_id SSE data fields, including final [DONE]."""
    data_events: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data_events.append(line[5:].strip())
    if not data_events:
        raise AssertionError("Frontend query stream contained no SSE data fields")
    if data_events[-1] != "[DONE]":
        raise AssertionError("Frontend query stream missing final [DONE]")
    if any(event == "[DONE]" for event in data_events[:-1]):
        raise AssertionError("Frontend query stream has [DONE] before the final event")
    return data_events


def _require_u64(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AssertionError(f"{name} is not an integer: {value!r}")
    if value < 0 or value > U64_MAX:
        raise AssertionError(f"{name} is out of unsigned64 range: {value!r}")
    return value


def _require_u32(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AssertionError(f"{name} is not an integer: {value!r}")
    if value < 0 or value > U32_MAX:
        raise AssertionError(f"{name} is out of unsigned32 range: {value!r}")
    return value


def collect_frontend_query_token_ids(
    frontend_url: str,
    payload: dict[str, Any],
    *,
    path: str,
    request_id: str,
    timeout_s: float = FRONTEND_QUERY_TIMEOUT_S,
) -> list[int]:
    """POST query_instance_id to native Frontend HTTP and parse nvext.token_ids.

    Adds the query annotation when absent so Frontend does not generate. Session
    trust_env is off. Non-200, incomplete SSE, or missing tokens fail the test.
    """
    return _parse_frontend_query_sse(
        _post_frontend_query(
            frontend_url,
            payload,
            path=path,
            request_id=request_id,
            timeout_s=timeout_s,
        )
    )


def collect_frontend_query_route(
    frontend_url: str,
    payload: dict[str, Any],
    *,
    path: str,
    request_id: str,
    timeout_s: float = FRONTEND_QUERY_TIMEOUT_S,
) -> RouteObservation:
    return _parse_frontend_query_route(
        _post_frontend_query(
            frontend_url,
            payload,
            path=path,
            request_id=request_id,
            timeout_s=timeout_s,
        )
    )


def _parse_frontend_query_sse(body: str) -> list[int]:
    """Require a complete query_instance_id SSE stream before comparing tokens."""
    return _token_ids_from_sse_events(_frontend_sse_data_events(body))


def _token_ids_from_sse_events(data_events: list[str]) -> list[int]:
    observed: list[int] | None = None
    for event in data_events[:-1]:
        try:
            chunk = json.loads(event)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                "Frontend query stream has a non-JSON data field"
            ) from exc
        if not isinstance(chunk, dict):
            raise AssertionError("Frontend query stream data field is not an object")
        if "error" in chunk:
            raise AssertionError("Frontend query stream contained an error chunk")
        nvext_chunk = chunk.get("nvext")
        if not isinstance(nvext_chunk, dict) or "token_ids" not in nvext_chunk:
            continue
        tokens = nvext_chunk["token_ids"]
        if not isinstance(tokens, list) or not tokens:
            raise AssertionError("Frontend nvext.token_ids is absent or empty")
        if not all(
            isinstance(token, int) and not isinstance(token, bool) for token in tokens
        ):
            raise AssertionError("Frontend nvext.token_ids contains a non-integer")
        as_ints = [int(token) for token in tokens]
        if observed is None:
            observed = as_ints
        elif as_ints != observed:
            raise AssertionError(
                "Frontend query stream has ambiguous token_ids at distinct positions"
            )
    if observed is None:
        raise AssertionError("Frontend nvext.token_ids is absent")
    return observed


def _typed_worker_rank(info: Mapping[str, Any]) -> tuple[int, int]:
    """Require matching prefill/decode worker fields; rank follows the same role."""
    pairs: list[tuple[int, int]] = []
    for id_key, rank_key, role in (
        ("prefill_worker_id", "prefill_dp_rank", "prefill"),
        ("decode_worker_id", "decode_dp_rank", "decode"),
    ):
        worker = info.get(id_key)
        rank = info.get(rank_key)
        if worker is None and rank is None:
            continue
        if worker is None or rank is None:
            raise AssertionError(
                f"Frontend nvext.worker_id {role} fields are incomplete: {dict(info)}"
            )
        pairs.append(
            (
                _require_u64(f"Frontend nvext.worker_id.{id_key}", worker),
                _require_u32(f"Frontend nvext.worker_id.{rank_key}", rank),
            )
        )
    if not pairs:
        raise AssertionError("Frontend nvext.worker_id is missing a worker id")
    if len(pairs) != 2:
        raise AssertionError(
            "Frontend aggregated nvext.worker_id must populate both prefill "
            f"and decode fields: {dict(info)}"
        )
    if pairs[0] != pairs[1]:
        raise AssertionError(
            "Frontend nvext.worker_id prefill and decode attribution disagree: "
            f"{dict(info)}"
        )
    return pairs[0]


def _parse_frontend_query_route(body: str) -> RouteObservation:
    """Require a complete query_instance_id SSE stream with unique worker/rank."""
    data_events = _frontend_sse_data_events(body)
    token_ids = _token_ids_from_sse_events(data_events)
    observed: tuple[int, int] | None = None
    for event in data_events[:-1]:
        chunk = json.loads(event)
        if not isinstance(chunk, dict):
            continue
        nvext_chunk = chunk.get("nvext")
        if not isinstance(nvext_chunk, dict) or "worker_id" not in nvext_chunk:
            continue
        worker_info = nvext_chunk["worker_id"]
        if not isinstance(worker_info, dict):
            raise AssertionError("Frontend nvext.worker_id is not an object")
        parsed = _typed_worker_rank(worker_info)
        if observed is None:
            observed = parsed
        elif parsed != observed:
            raise AssertionError(
                "Frontend query stream has ambiguous worker_id at distinct positions"
            )
    if observed is None:
        raise AssertionError("Frontend nvext.worker_id is absent")
    return RouteObservation(
        worker_id=observed[0],
        dp_rank=observed[1],
        token_ids=token_ids,
        routing_mode=AGGREGATED_ROUTING_MODE,
    )


def wait_for_frontend_models(
    frontend_url: str, timeout_s: float = PROCESS_TIMEOUT_S
) -> None:
    """Poll Frontend /v1/models until a model appears or timeout_s elapses."""
    deadline = time.monotonic() + timeout_s
    last_error = "frontend /v1/models not checked"
    with requests.Session() as session:
        session.trust_env = False
        while time.monotonic() < deadline:
            try:
                response = session.get(f"{frontend_url}/v1/models", timeout=2)
                if check_http_ok(response):
                    payload = response.json()
                    models = payload.get("data") if isinstance(payload, dict) else None
                    if isinstance(models, list) and models:
                        return
                    last_error = "frontend /v1/models returned no models"
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
            time.sleep(0.25)
    raise TimeoutError(f"Frontend models were not ready: {last_error}")


class EppFrontendParityStack:
    """Own the loopback API, one mocker, native frontend HTTP, and real /epp."""

    def __init__(self, request, tmp_path: Path, model_name: str) -> None:
        self.request = request
        self.tmp_path = tmp_path
        self.model_name = model_name
        self.identity = resolve_pod_identity()
        self._stack = ExitStack()
        self._lock: FileLock | None = None
        self.kube_api: LoopbackKubeAPI | None = None
        self.mockers = None
        self.frontend_port: int | None = None
        self.driver: ExtProcParityDriver | None = None
        self._allocated_ports: list[int] = []

    @property
    def frontend_url(self) -> str:
        if self.frontend_port is None:
            raise RuntimeError("parity stack has not started")
        return f"http://127.0.0.1:{self.frontend_port}"

    def collect_frontend_tokens(
        self, payload: dict[str, Any], *, path: str, request_id: str
    ) -> list[int]:
        """Collect query_instance_id tokens from the started Frontend HTTP port."""
        return collect_frontend_query_token_ids(
            self.frontend_url, payload, path=path, request_id=request_id
        )

    def collect_epp_tokens(
        self, payload: dict[str, Any], *, path: str, request_id: str
    ) -> list[int]:
        """Collect token_data from the started /epp ext_proc listener."""
        if self.driver is None:
            raise RuntimeError("parity stack has not started")
        return self.driver.collect_token_data(payload, path=path, request_id=request_id)

    def __enter__(self) -> EppFrontendParityStack:
        """Acquire the session lock and start API, mocker, Frontend, and /epp."""
        try:
            return self._start(MockerProcess)
        except Exception:
            self._stack.close()
            raise

    def _start(self, mocker_cls) -> EppFrontendParityStack:
        """Launch children against the loopback API; fail if the fixture recorded errors."""
        self._lock = acquire_epp_session_lock()
        self._stack.callback(self._lock.release)
        assert_epp_ports_free()
        allocated = allocate_ports(3, DynamoPortRange.ROUTER.value)
        self._allocated_ports = allocated
        self.request.addfinalizer(lambda: deallocate_ports(list(allocated)))
        api_port, frontend_port, worker_http_port = allocated
        self.frontend_port = frontend_port

        kubeconfig = write_loopback_kubeconfig(
            self.tmp_path / "kubeconfig",
            "127.0.0.1",
            api_port,
            self.identity.pod_namespace,
        )
        self.kube_api = LoopbackKubeAPI(self.identity.pod_namespace, api_port)
        self.kube_api.start()
        self._stack.callback(self.kube_api.stop)

        proto_modules = _ExtProcModules(self.tmp_path / "ext_proc_pb")
        self._stack.callback(proto_modules.close)
        self.driver = ExtProcParityDriver(proto_modules)

        mocker_env = loopback_process_env(
            kubeconfig, self.identity, WORKER_CONTAINER_NAME
        )
        self.mockers = mocker_cls(
            self.request,
            mocker_args={"speedup_ratio": 10.0, "block_size": 16},
            num_mockers=1,
            store_backend="kubernetes",
            request_plane="tcp",
            extra_env=mocker_env,
        )
        self.kube_api.upsert_worker_pod(
            identity=self.identity,
            dynamo_namespace=self.mockers.namespace,
            container_name=WORKER_CONTAINER_NAME,
            http_port=worker_http_port,
        )
        self._stack.enter_context(self.mockers)

        frontend_env = loopback_process_env(
            kubeconfig, self.identity, FRONTEND_CONTAINER_NAME
        )
        self._stack.enter_context(
            FrontendRouterProcess(
                self.request,
                16,
                frontend_port,
                self.mockers.namespace,
                "kubernetes",
                request_plane="tcp",
                event_plane="zmq",
                router_mode="kv",
                min_initial_workers=1,
                extra_env=frontend_env,
            )
        )

        epp_env = loopback_process_env(kubeconfig, self.identity, EPP_CONTAINER_NAME)
        epp_env.update(
            {
                "DYN_NAMESPACE": self.mockers.namespace,
                "DYN_COMPONENT_NAME": "mocker",
            }
        )
        self._stack.enter_context(
            RuntimeEppProcess(
                self.request,
                binary=epp_binary_path(),
                grpc_port=EPP_GRPC_PORT,
                health_port=EPP_HEALTH_PORT,
                extra_env=epp_env,
                timeout=PROCESS_TIMEOUT_S,
                kubeconfig=kubeconfig,
            )
        )
        wait_for_frontend_models(self.frontend_url)
        if self.kube_api.failures:
            raise RuntimeError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(self.kube_api.failures)
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Tear down children, then fail on recorded fixture API errors if the body passed."""
        close_error = None
        try:
            self._stack.close()
        except Exception as err:
            close_error = err
        failures = list(self.kube_api.failures) if self.kube_api is not None else []
        if failures:
            api_error = AssertionError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(failures)
            )
            if exc_type is None:
                if close_error is not None:
                    raise api_error from close_error
                raise api_error
            return None
        if close_error is not None and exc_type is None:
            raise close_error
        return None


@contextmanager
def scoped_process_env(overlay: Mapping[str, str | None]) -> Iterator[None]:
    """Apply process env for one runtime construction, then restore the parent."""
    previous: dict[str, str | None] = {}
    try:
        for name, value in overlay.items():
            previous[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def provider_query_metric_name() -> str:
    from dynamo.prometheus_names import name_prefix, work_handler

    return f"{name_prefix.COMPONENT}_{work_handler.REQUESTS_TOTAL}"


def first_selection_router_env(*, use_remote_indexer: bool) -> dict[str, str | None]:
    """One cache-sensitive router overlay for the provider and both consumers.

    Explicit values pin temperature 0 and cache-sensitive credit/scale/tracking.
    ``None`` clears inherited selectors and deprecated overlap-weight aliases so
    ``KvRouterConfig``'s env parser cannot silently pick another scorer.
    """
    return {
        "DYN_ROUTER_TEMPERATURE": "0",
        "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT": "1",
        "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY": "0",
        "DYN_ROUTER_PREFILL_LOAD_SCALE": "1",
        "DYN_ROUTER_TRACK_PREFILL_TOKENS": "true",
        "DYN_ROUTER_TRACK_OUTPUT_BLOCKS": "false",
        "DYN_ROUTER_REPLICA_SYNC": "false",
        "DYN_ROUTER_TRACKING_HASH": "public-xxh3-v1",
        "DYN_SHARED_CACHE_TYPE": "none",
        "DYN_ROUTER_CONDITIONAL_DISAGG": "false",
        "DYN_ROUTER_PREFILL_LOAD_MODEL": "none",
        "DYN_ROUTER_USE_KV_EVENTS": "true",
        "DYN_USE_REMOTE_INDEXER": "true" if use_remote_indexer else "false",
        "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT": None,
        "DYN_OVERLAP_SCORE_WEIGHT": None,
        "DYN_ROUTER_POLICY_CONFIG": None,
        "DYN_ROUTER_WORKER_SELECTION_POLICY": None,
        "DYN_ROUTER_PREFILL_POLICY": None,
        "DYN_ROUTER_DECODE_POLICY": None,
        "DYN_ROUTER_QUEUE_THRESHOLD": None,
        "DYN_SHARED_CACHE_MULTIPLIER": None,
        "DYN_ROUTER_TRACKING_KEY_FILE": None,
        "DYN_ROUTER_TRACKING_KEY_ID": None,
        "DYN_USE_KV_EVENTS": None,
    }


def scrape_provider_query_count(
    metrics_url: str,
    *,
    namespace: str,
    component: str,
    provider_instance_id: int,
) -> float:
    from dynamo.prometheus_names import labels
    from tests.utils.prometheus import find_metric_samples

    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"{metrics_url}/metrics", timeout=2)
    if response.status_code != 200:
        raise AssertionError(f"provider /metrics returned HTTP {response.status_code}")
    metric_name = provider_query_metric_name()
    # Fixture MockerProcess uses test-namespace-{random lowercase}; native
    # metrics sanitize that NAMESPACE label by replacing '-' with '_'.
    namespace_label = namespace.replace("-", "_")
    samples = find_metric_samples(
        response.text,
        metric_name,
        {
            labels.NAMESPACE: namespace_label,
            labels.COMPONENT: component,
            labels.ENDPOINT: KV_INDEXER_QUERY_ENDPOINT,
            labels.WORKER_ID: format(provider_instance_id, "x"),
        },
    )
    if not samples:
        raise AssertionError(
            "provider did not export "
            f"{metric_name} with "
            f"{labels.NAMESPACE}={namespace_label!r} "
            f"(raw namespace={namespace!r}) "
            f"{labels.ENDPOINT}={KV_INDEXER_QUERY_ENDPOINT!r} "
            f"{labels.WORKER_ID}={format(provider_instance_id, 'x')!r}"
        )
    return sum(samples)


def catalog_named_endpoint_instances(
    metadata_objects: list[Mapping[str, Any]],
    endpoint_name: str,
    *,
    namespace: str,
    component: str,
) -> dict[int, dict[str, Any]]:
    """Map actual endpoint instance IDs from fixture DynamoWorkerMetadata."""
    catalog: dict[int, dict[str, Any]] = {}
    for obj in metadata_objects:
        spec = obj.get("spec") or {}
        data = spec.get("data") if isinstance(spec, dict) else None
        if not isinstance(data, dict):
            continue
        endpoints = data.get("endpoints")
        if not isinstance(endpoints, dict):
            continue
        cr_name = (obj.get("metadata") or {}).get("name")
        owner_uid = _owner_pod_uid(obj)
        for endpoint in endpoints.values():
            if not isinstance(endpoint, dict):
                continue
            if endpoint.get("endpoint") != endpoint_name:
                continue
            if endpoint.get("namespace") != namespace:
                continue
            if endpoint.get("component") != component:
                continue
            raw_id = endpoint.get("instance_id")
            if not isinstance(raw_id, int) or isinstance(raw_id, bool):
                raise AssertionError(
                    f"DynamoWorkerMetadata {endpoint_name} instance_id is "
                    f"malformed: {raw_id!r}"
                )
            if raw_id in catalog and catalog[raw_id]["cr_name"] != cr_name:
                raise AssertionError(
                    f"{endpoint_name} instance_id {raw_id} is registered on "
                    "multiple CRs"
                )
            catalog[raw_id] = {
                "cr_name": cr_name,
                "namespace": endpoint.get("namespace"),
                "component": endpoint.get("component"),
                "endpoint": endpoint_name,
                "owner_uid": owner_uid,
            }
    return catalog


def catalog_generate_instances(
    metadata_objects: list[Mapping[str, Any]],
    *,
    namespace: str,
    component: str,
) -> dict[int, dict[str, Any]]:
    """Map generate-endpoint instance IDs from the expected namespace/component."""
    return catalog_named_endpoint_instances(
        metadata_objects, "generate", namespace=namespace, component=component
    )


def _owner_pod_uid(obj: Mapping[str, Any]) -> str | None:
    owners = (obj.get("metadata") or {}).get("ownerReferences") or []
    if not isinstance(owners, list):
        return None
    uids: list[str] = []
    for owner in owners:
        if not isinstance(owner, dict):
            continue
        if owner.get("kind") != "Pod":
            continue
        uid = owner.get("uid")
        if isinstance(uid, str) and uid:
            uids.append(uid)
    if not uids:
        return None
    unique = set(uids)
    if len(unique) != 1:
        raise AssertionError(f"DynamoWorkerMetadata has ambiguous Pod owners: {uids}")
    return unique.pop()


def catalog_worker_model_configs(
    metadata_objects: list[Mapping[str, Any]],
    *,
    namespace: str,
    component: str,
) -> dict[int, dict[str, Any]]:
    """Read registered generate-worker model cards from fixture metadata."""
    configs: dict[int, dict[str, Any]] = {}
    for obj in metadata_objects:
        spec = obj.get("spec") or {}
        data = spec.get("data") if isinstance(spec, dict) else None
        if not isinstance(data, dict):
            continue
        cards = data.get("model_cards")
        if not isinstance(cards, dict):
            continue
        for card in cards.values():
            if not isinstance(card, dict):
                continue
            if card.get("namespace") != namespace or card.get("component") != component:
                continue
            raw_id = card.get("instance_id")
            if not isinstance(raw_id, int) or isinstance(raw_id, bool):
                raise AssertionError(
                    f"DynamoWorkerMetadata model card instance_id is malformed: {raw_id!r}"
                )
            card_json = card.get("card_json")
            if not isinstance(card_json, dict):
                raise AssertionError(
                    f"DynamoWorkerMetadata model card JSON is missing for {raw_id}"
                )
            runtime = card_json.get("runtime_config")
            if not isinstance(runtime, dict):
                raise AssertionError(
                    f"worker {raw_id} model card is missing runtime_config"
                )
            configs[raw_id] = {
                "kv_cache_block_size": card_json.get("kv_cache_block_size"),
                "data_parallel_size": runtime.get("data_parallel_size"),
                "kv_event_publishing_enabled": runtime.get(
                    "kv_event_publishing_enabled"
                ),
            }
    return configs


def ready_container_names(pod: Mapping[str, Any]) -> set[str]:
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    names: set[str] = set()
    if not isinstance(statuses, list):
        return names
    for status in statuses:
        if isinstance(status, dict) and status.get("ready") and status.get("name"):
            names.add(str(status["name"]))
    return names


_LOOPBACK_KUBE_API_EXPOSED = (
    "start",
    "stop",
    "add_ready_container",
    "listed_metadata",
    "listed_pods",
    "get_failures",
)
_SPAWN_CTX = multiprocessing.get_context("spawn")


# Routing hosts LoopbackKubeAPI in a spawned process: DistributedRuntime::new
# block_on holds the pytest GIL, so a same-process API thread cannot serve
# kube list/watch during that constructor. spawn is required; fork would
# copy an initialized Tokio runtime.
class LoopbackKubeAPIManager(BaseManager):
    """Spawned BaseManager host for the existing LoopbackKubeAPI fixture."""


LoopbackKubeAPIManager.register(
    "LoopbackKubeAPI",
    LoopbackKubeAPI,
    exposed=_LOOPBACK_KUBE_API_EXPOSED,
)


def _reap_owned_manager_process(process: Any) -> None:
    """Bounded join of the retained manager Process; kill only that child."""
    if process.is_alive():
        process.join(timeout=KUBE_API_MANAGER_SHUTDOWN_TIMEOUT_S)
    if process.is_alive():
        process.terminate()
        process.join(timeout=KUBE_API_MANAGER_SHUTDOWN_TIMEOUT_S)
    if process.is_alive():
        process.kill()
        process.join(timeout=KUBE_API_MANAGER_SHUTDOWN_TIMEOUT_S)
    if process.is_alive():
        raise RuntimeError(
            f"owned kube API manager child pid {process.pid} did not exit"
        )
    logger.info(
        "Loopback Kubernetes API manager child pid=%s exitcode=%s alive=%s",
        process.pid,
        process.exitcode,
        process.is_alive(),
    )


def _shutdown_unregistered_manager(
    manager: LoopbackKubeAPIManager, start_error: BaseException
) -> None:
    try:
        process = manager._process
    except AttributeError:
        return
    if process.pid is None:
        return
    try:
        _reap_owned_manager_process(process)
    except RuntimeError as join_exc:
        raise join_exc from start_error


class LoopbackKubeAPIOwner:
    """Typed parent owner for the spawned LoopbackKubeAPI proxy.

    Live ``.failures`` reads the true proxy. Cleanup snapshots after API.stop
    and before manager shutdown. After the manager exits, ``.failures`` returns
    a copy of that snapshot or the saved snapshot error; never a fake empty.
    """

    def __init__(self, proxy: Any) -> None:
        self._proxy = proxy
        self._failures_snapshot: list[str] | None = None
        self._snapshot_error: BaseException | None = None
        self._manager_exited = False

    def start(self) -> None:
        self._proxy.start()

    def snapshot_failures(self) -> None:
        """Record true failures after API.stop and before manager shutdown."""
        if self._failures_snapshot is not None:
            return
        try:
            self._failures_snapshot = list(self._proxy.get_failures())
            self._snapshot_error = None
        except BaseException as exc:
            self._snapshot_error = exc
            raise

    def mark_manager_exited(self) -> None:
        self._manager_exited = True

    def stop(self) -> None:
        self._proxy.stop()

    def add_ready_container(
        self,
        *,
        identity: PodIdentity,
        dynamo_namespace: str,
        container_name: str,
        http_port: int,
    ) -> None:
        self._proxy.add_ready_container(
            identity=identity,
            dynamo_namespace=dynamo_namespace,
            container_name=container_name,
            http_port=http_port,
        )

    def listed_metadata(self) -> list[dict[str, Any]]:
        return self._proxy.listed_metadata()

    def listed_pods(self) -> list[dict[str, Any]]:
        return self._proxy.listed_pods()

    def get_failures(self) -> list[str]:
        if self._failures_snapshot is not None:
            return list(self._failures_snapshot)
        if self._manager_exited:
            if self._snapshot_error is not None:
                raise self._snapshot_error
            raise RuntimeError(
                "Loopback Kubernetes API manager exited before failures were "
                "snapshotted"
            )
        return list(self._proxy.get_failures())

    @property
    def failures(self) -> list[str]:
        return self.get_failures()


class EppFrontendRoutingParityStack:
    """Two mockers, one served remote index, and fresh FE/EPP consumers."""

    def __init__(self, request, tmp_path: Path, model_name: str) -> None:
        self.request = request
        self.tmp_path = tmp_path
        self.model_name = model_name
        self.identity = resolve_pod_identity()
        self._stack = ExitStack()
        self._lock: FileLock | None = None
        self._kube_api_manager: LoopbackKubeAPIManager | None = None
        self._kube_api_manager_process = None
        self.kube_api: LoopbackKubeAPIOwner | None = None
        self.kubeconfig: Path | None = None
        self.namespace: str | None = None
        self.component_name = "mocker"
        self.mocker_a = None
        self.mocker_b = None
        self.frontend_port: int | None = None
        self.provider_metrics_port: int | None = None
        self.worker_http_ports: dict[str, int] = {}
        self.pod_destination: str | None = None
        self.worker_ids: dict[str, int] = {}
        self.worker_configs: dict[int, dict[str, Any]] = {}
        self.provider_instance_id: int | None = None
        self.provider_runtime = None
        self.provider_router = None
        self.provider_router_config = None
        self.policy_parameters: dict[str, float] = {}
        self.driver: ExtProcParityDriver | None = None
        self._loop = None
        self._allocated_ports: list[int] = []
        self._consumer_stack: ExitStack | None = None
        self._consumer_pair_seq = 0

    @property
    def frontend_url(self) -> str:
        if self.frontend_port is None:
            raise RuntimeError("routing parity stack has not started")
        return f"http://127.0.0.1:{self.frontend_port}"

    @property
    def provider_metrics_url(self) -> str:
        if self.provider_metrics_port is None:
            raise RuntimeError("routing parity stack has not started")
        return f"http://127.0.0.1:{self.provider_metrics_port}"

    def collect_frontend_route(
        self, payload: dict[str, Any], *, path: str, request_id: str
    ) -> RouteObservation:
        return collect_frontend_query_route(
            self.frontend_url, payload, path=path, request_id=request_id
        )

    def collect_epp_route(
        self, payload: dict[str, Any], *, path: str, request_id: str
    ) -> RouteObservation:
        if self.driver is None:
            raise RuntimeError("routing parity stack has not started")
        observation = self.driver.collect_route_observation(
            payload, path=path, request_id=request_id
        )
        if self.pod_destination is None:
            raise AssertionError("fixture pod destination was not recorded")
        if observation.destination != self.pod_destination:
            raise AssertionError(
                "EPP destination does not match the fixture Pod mapping: "
                f"{observation.destination!r} != {self.pod_destination!r}"
            )
        return observation

    def __enter__(self) -> EppFrontendRoutingParityStack:
        try:
            return self._start(MockerProcess)
        except BaseException:
            # PyO3 PanicException is a BaseException. Close the spawned
            # manager and any later stack resources before it leaves.
            self._stack.close()
            raise

    def _start(self, mocker_cls) -> EppFrontendRoutingParityStack:
        import asyncio

        from dynamo.llm import KvRouter, KvRouterConfig
        from tests.router.common import _create_kv_router_with_timeout
        from tests.router.helper import get_runtime, poll_for_worker_instances

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "EppFrontendRoutingParityStack is a synchronous fixture and "
                "cannot run inside an active asyncio event loop"
            )

        self._lock = acquire_epp_session_lock()
        self._stack.callback(self._lock.release)
        assert_epp_ports_free()
        allocated = allocate_ports(5, DynamoPortRange.ROUTER.value)
        self._allocated_ports = allocated
        self.request.addfinalizer(lambda: deallocate_ports(list(allocated)))
        (
            api_port,
            frontend_port,
            worker_a_http,
            worker_b_http,
            provider_metrics_port,
        ) = allocated
        self.frontend_port = frontend_port
        self.provider_metrics_port = provider_metrics_port
        self.worker_http_ports = {
            WORKER_A_CONTAINER_NAME: worker_a_http,
            WORKER_B_CONTAINER_NAME: worker_b_http,
        }
        self.pod_destination = f"127.0.0.1:{worker_a_http}"

        kubeconfig = write_loopback_kubeconfig(
            self.tmp_path / "kubeconfig",
            "127.0.0.1",
            api_port,
            self.identity.pod_namespace,
        )
        self.kubeconfig = kubeconfig
        manager = LoopbackKubeAPIManager(
            ctx=_SPAWN_CTX,
            shutdown_timeout=KUBE_API_MANAGER_SHUTDOWN_TIMEOUT_S,
        )
        try:
            manager.start()
        except BaseException as exc:
            _shutdown_unregistered_manager(manager, exc)
            raise
        process = manager._process
        self._kube_api_manager = manager
        self._kube_api_manager_process = process
        logger.info("Loopback Kubernetes API manager child pid=%s", process.pid)
        self._stack.callback(self._shutdown_kube_api_manager)

        self.kube_api = LoopbackKubeAPIOwner(
            manager.LoopbackKubeAPI(self.identity.pod_namespace, api_port)
        )
        self.kube_api.start()

        proto_modules = _ExtProcModules(self.tmp_path / "ext_proc_pb")
        self._stack.callback(proto_modules.close)
        self.driver = ExtProcParityDriver(proto_modules)

        mocker_a_env = loopback_process_env(
            kubeconfig, self.identity, WORKER_A_CONTAINER_NAME
        )
        worker_args = {
            "speedup_ratio": 10.0,
            "block_size": WORKER_BLOCK_SIZE,
            "dp_size": WORKER_DP_SIZE,
            "enable_prefix_caching": True,
        }
        self.mocker_a = mocker_cls(
            self.request,
            mocker_args=worker_args,
            num_mockers=1,
            store_backend="kubernetes",
            request_plane="tcp",
            extra_env=mocker_a_env,
            display_name="dynamo-mocker-a",
        )
        self.namespace = self.mocker_a.namespace
        mocker_b_env = loopback_process_env(
            kubeconfig, self.identity, WORKER_B_CONTAINER_NAME
        )
        self.mocker_b = mocker_cls(
            self.request,
            mocker_args=worker_args,
            num_mockers=1,
            store_backend="kubernetes",
            request_plane="tcp",
            extra_env=mocker_b_env,
            namespace=self.namespace,
            display_name="dynamo-mocker-b",
        )
        self.kube_api.add_ready_container(
            identity=self.identity,
            dynamo_namespace=self.namespace,
            container_name=WORKER_A_CONTAINER_NAME,
            http_port=worker_a_http,
        )
        self.kube_api.add_ready_container(
            identity=self.identity,
            dynamo_namespace=self.namespace,
            container_name=WORKER_B_CONTAINER_NAME,
            http_port=worker_b_http,
        )
        self.kube_api.add_ready_container(
            identity=self.identity,
            dynamo_namespace=self.namespace,
            container_name=PROVIDER_CONTAINER_NAME,
            http_port=provider_metrics_port,
        )
        self._stack.enter_context(self.mocker_a)
        self._stack.enter_context(self.mocker_b)

        provider_env = loopback_process_env(
            kubeconfig, self.identity, PROVIDER_CONTAINER_NAME
        )
        provider_env.update(first_selection_router_env(use_remote_indexer=False))
        provider_env["DYN_SYSTEM_PORT"] = str(provider_metrics_port)
        self._loop = asyncio.new_event_loop()
        self._stack.callback(self._close_loop)

        async def create_provider_runtime():
            return get_runtime(
                store_backend="kubernetes",
                request_plane="tcp",
                event_plane="zmq",
            )

        with scoped_process_env(provider_env):
            self.provider_runtime = self._loop.run_until_complete(
                create_provider_runtime()
            )
            self._stack.callback(self._shutdown_provider)
            generate_path = f"{self.namespace}.{self.component_name}.generate"
            endpoint = self.provider_runtime.endpoint(generate_path)
            self.provider_router_config = KvRouterConfig(
                use_kv_events=True,
                serve_indexer=True,
                use_remote_indexer=False,
                router_temperature=0.0,
                router_replica_sync=False,
                router_track_prefill_tokens=True,
                router_track_output_blocks=False,
                overlap_score_credit=1.0,
                overlap_score_credit_decay=0.0,
                prefill_load_scale=1.0,
                shared_cache_type="none",
                conditional_disagg_enabled=False,
                router_tracking_hash="public-xxh3-v1",
                router_policy_config=None,
                router_prefill_policy=None,
                router_decode_policy=None,
                router_queue_threshold=None,
                router_prefill_load_model="none",
            )
            self.policy_parameters = self.provider_router_config.policy_parameters(
                "aggregated"
            )
            self.provider_router = _create_kv_router_with_timeout(
                router_factory=lambda: KvRouter(
                    endpoint=endpoint,
                    block_size=WORKER_BLOCK_SIZE,
                    kv_router_config=self.provider_router_config,
                ),
                num_workers=2,
                engine_workers=self.mocker_a,
            )

        async def wait_for_catalog() -> dict[str, int]:
            generate_ids = await poll_for_worker_instances(
                endpoint, 2, max_wait_time=120
            )
            query_endpoint = self.provider_runtime.endpoint(
                f"{self.namespace}.{self.component_name}.{KV_INDEXER_QUERY_ENDPOINT}"
            )
            query_client = await query_endpoint.client()
            record_endpoint = self.provider_runtime.endpoint(
                f"{self.namespace}.{self.component_name}."
                "kv_indexer_record_routing_decision"
            )
            record_client = await record_endpoint.client()
            deadline = time.monotonic() + PROCESS_TIMEOUT_S
            catalog: dict[int, dict[str, Any]] = {}
            query_instances: list[Any] = []
            models: dict[int, dict[str, Any]] = {}
            while time.monotonic() < deadline:
                listed = self.kube_api.listed_metadata()
                catalog = catalog_generate_instances(
                    listed,
                    namespace=self.namespace,
                    component=self.component_name,
                )
                query_catalog = catalog_named_endpoint_instances(
                    listed,
                    KV_INDEXER_QUERY_ENDPOINT,
                    namespace=self.namespace,
                    component=self.component_name,
                )
                record_catalog = catalog_named_endpoint_instances(
                    listed,
                    "kv_indexer_record_routing_decision",
                    namespace=self.namespace,
                    component=self.component_name,
                )
                query_instances = list(query_client.instances())
                record_ids = set(record_client.instance_ids())
                models = catalog_worker_model_configs(
                    listed,
                    namespace=self.namespace,
                    component=self.component_name,
                )
                if (
                    len(catalog) == 2
                    and set(generate_ids) == set(catalog)
                    and len(query_instances) == 1
                    and len(query_catalog) == 1
                    and not record_ids
                    and not record_catalog
                    and set(models) >= set(catalog)
                    and self._ready_joins_complete(catalog, query_catalog)
                    and self._worker_models_match(models, catalog)
                ):
                    break
                await asyncio.sleep(0.25)
            else:
                raise TimeoutError(
                    "generate workers or served kv_indexer_query did not register: "
                    f"generate={generate_ids} catalog={sorted(catalog)} "
                    f"query={[inst.instance_id for inst in query_instances]} "
                    f"record={record_client.instance_ids()} models={models}"
                )
            if len(query_instances) != 1:
                raise AssertionError(
                    "served kv_indexer_query must be exactly one discovered "
                    f"instance, got {[inst.instance_id for inst in query_instances]}"
                )
            provider = query_instances[0]
            if (
                provider.namespace != self.namespace
                or provider.component != self.component_name
                or provider.endpoint != KV_INDEXER_QUERY_ENDPOINT
            ):
                raise AssertionError(
                    "kv_indexer_query instance metadata does not match the "
                    f"served provider: {provider}"
                )
            self.provider_instance_id = provider.instance_id
            if self.provider_instance_id in catalog:
                raise AssertionError(
                    "provider kv_indexer_query instance must not be a generate "
                    f"target: {self.provider_instance_id}"
                )
            self.worker_configs = {
                worker_id: models[worker_id] for worker_id in catalog
            }
            labeled = sorted(catalog)
            logger.info(
                "routing catalog ready workers=%s provider=%s policy=%s configs=%s",
                {"A": labeled[0], "B": labeled[1]},
                self.provider_instance_id,
                self.policy_parameters,
                self.worker_configs,
            )
            return {"A": labeled[0], "B": labeled[1]}

        self.worker_ids = self._loop.run_until_complete(wait_for_catalog())
        if self.kube_api.failures:
            raise RuntimeError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(self.kube_api.failures)
            )
        return self

    def run_until_complete(self, coro):
        if self._loop is None:
            raise RuntimeError("routing parity stack has not started")
        return self._loop.run_until_complete(coro)

    def open_consumer_pair(self, *, include_epp: bool = True) -> None:
        if (
            self.kubeconfig is None
            or self.namespace is None
            or self.frontend_port is None
        ):
            raise RuntimeError("routing parity stack has not started")
        if self._consumer_stack is not None:
            raise RuntimeError("a consumer pair is already open")
        if include_epp:
            assert_epp_ports_free()
        self._consumer_pair_seq += 1
        pair_label = f"pair-{self._consumer_pair_seq}"
        stack = ExitStack()
        try:
            frontend_env = loopback_process_env(
                self.kubeconfig, self.identity, FRONTEND_CONTAINER_NAME
            )
            frontend_env.update(first_selection_router_env(use_remote_indexer=True))
            frontend = FrontendRouterProcess(
                self.request,
                WORKER_BLOCK_SIZE,
                self.frontend_port,
                self.namespace,
                "kubernetes",
                request_plane="tcp",
                event_plane="zmq",
                router_mode="kv",
                min_initial_workers=2,
                use_remote_indexer=True,
                extra_env=frontend_env,
            )
            frontend.display_name = (
                f"dynamo-frontend-kv-{self.frontend_port}-{pair_label}"
            )
            frontend.log_dir = f"{self.request.node.name}/{pair_label}-frontend"
            stack.enter_context(frontend)
            if include_epp:
                epp_env = loopback_process_env(
                    self.kubeconfig, self.identity, EPP_CONTAINER_NAME
                )
                epp_env.update(first_selection_router_env(use_remote_indexer=True))
                epp_env.update(
                    {
                        "DYN_NAMESPACE": self.namespace,
                        "DYN_COMPONENT_NAME": self.component_name,
                    }
                )
                epp = RuntimeEppProcess(
                    self.request,
                    binary=epp_binary_path(),
                    grpc_port=EPP_GRPC_PORT,
                    health_port=EPP_HEALTH_PORT,
                    extra_env=epp_env,
                    timeout=PROCESS_TIMEOUT_S,
                    kubeconfig=self.kubeconfig,
                )
                epp.display_name = f"dynamo-runtime-epp-{pair_label}"
                epp.log_dir = f"{self.request.node.name}/{pair_label}-epp"
                stack.enter_context(epp)
            wait_for_frontend_models(self.frontend_url)
        except Exception:
            stack.close()
            raise
        self._consumer_stack = stack
        logger.info(
            "opened consumer %s include_epp=%s frontend_log=%s",
            pair_label,
            include_epp,
            frontend.log_dir,
        )

    def close_consumer_pair(self) -> None:
        stack = self._consumer_stack
        self._consumer_stack = None
        if stack is not None:
            stack.close()

    def provider_query_count(self) -> float:
        if self.namespace is None:
            raise RuntimeError("routing parity stack has not started")
        if self.provider_instance_id is None:
            raise RuntimeError(
                "routing parity stack has not discovered the kv_indexer_query instance"
            )
        return scrape_provider_query_count(
            self.provider_metrics_url,
            namespace=self.namespace,
            component=self.component_name,
            provider_instance_id=self.provider_instance_id,
        )

    def dump_provider_events(self) -> str:
        if self.provider_router is None:
            raise RuntimeError("routing parity stack has not started")

        async def _dump():
            return await self.provider_router.dump_events()

        return self.run_until_complete(_dump())

    def __exit__(self, exc_type, exc, tb) -> None:
        close_error = None
        try:
            self.close_consumer_pair()
        except Exception as err:
            close_error = err
        try:
            self._stack.close()
        except Exception as err:
            if close_error is None:
                close_error = err
        failures = list(self.kube_api.failures) if self.kube_api is not None else []
        if failures:
            api_error = AssertionError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(failures)
            )
            if exc_type is None:
                if close_error is not None:
                    raise api_error from close_error
                raise api_error
            return None
        if close_error is not None and exc_type is None:
            raise close_error
        return None

    def _shutdown_kube_api_manager(self) -> None:
        owner = self.kube_api
        manager = self._kube_api_manager
        process = self._kube_api_manager_process
        self._kube_api_manager = None
        self._kube_api_manager_process = None
        held: list[BaseException] = []

        def _run_cleanup() -> None:
            try:
                try:
                    if owner is not None:
                        try:
                            owner.stop()
                        finally:
                            owner.snapshot_failures()
                finally:
                    if manager is not None:
                        manager.shutdown()
            except BaseException as exc:
                held.append(exc)
            finally:
                if owner is not None:
                    owner.mark_manager_exited()

        thread = threading.Thread(
            target=_run_cleanup,
            name="epp-parity-kube-api-cleanup",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=KUBE_API_MANAGER_SHUTDOWN_TIMEOUT_S)
        timeout_error = None
        reap_error = None
        if thread.is_alive():
            timeout_error = TimeoutError(
                "Loopback Kubernetes API cleanup exceeded "
                f"{KUBE_API_MANAGER_SHUTDOWN_TIMEOUT_S}s"
            )
            if process is not None:
                try:
                    _reap_owned_manager_process(process)
                except BaseException as exc:
                    reap_error = exc
            thread.join(timeout=KUBE_API_MANAGER_SHUTDOWN_TIMEOUT_S)
        elif process is not None:
            try:
                _reap_owned_manager_process(process)
            except BaseException as exc:
                reap_error = exc
        if owner is not None:
            owner.mark_manager_exited()
        if timeout_error is not None:
            cause = held[0] if held else reap_error
            if cause is not None:
                raise timeout_error from cause
            raise timeout_error
        if held:
            raise held[0]
        if reap_error is not None:
            raise reap_error

    def _shutdown_provider(self) -> None:
        router = self.provider_router
        self.provider_router = None
        if router is not None:
            del router
        runtime = self.provider_runtime
        self.provider_runtime = None
        if runtime is not None:
            runtime.shutdown()

    def _close_loop(self) -> None:
        loop = self._loop
        self._loop = None
        if loop is not None and not loop.is_closed():
            loop.close()

    def _ready_joins_complete(
        self,
        generate_catalog: Mapping[int, Mapping[str, Any]],
        query_catalog: Mapping[int, Mapping[str, Any]],
    ) -> bool:
        pods = self.kube_api.listed_pods()
        if len(pods) != 1:
            return False
        pod = pods[0]
        pod_uid = (pod.get("metadata") or {}).get("uid")
        if pod_uid != self.identity.pod_uid:
            return False
        ready = ready_container_names(pod)
        expected = {
            WORKER_A_CONTAINER_NAME,
            WORKER_B_CONTAINER_NAME,
            PROVIDER_CONTAINER_NAME,
        }
        if not expected.issubset(ready):
            return False
        for info in generate_catalog.values():
            if info.get("owner_uid") != pod_uid:
                return False
        for info in query_catalog.values():
            if info.get("owner_uid") != pod_uid:
                return False
        return True

    def _worker_models_match(
        self,
        models: Mapping[int, Mapping[str, Any]],
        generate_catalog: Mapping[int, Mapping[str, Any]],
    ) -> bool:
        for worker_id in generate_catalog:
            config = models.get(worker_id)
            if config is None:
                return False
            if config.get("kv_cache_block_size") != WORKER_BLOCK_SIZE:
                return False
            if config.get("data_parallel_size") != WORKER_DP_SIZE:
                return False
            if config.get("kv_event_publishing_enabled") is not True:
                return False
        return True
