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
import os
import queue
import socket
import sys
import tempfile
import threading
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import parse_qs, urlparse

import grpc
from filelock import FileLock

from tests.router.helper import generate_random_suffix
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
FRONTEND_CONTAINER_NAME = "frontend"
EPP_CONTAINER_NAME = "epp"
FULL_DUPLEX_STREAMED = 4
PROCESS_TIMEOUT_S = 240.0
EXT_PROC_TIMEOUT_S = 45.0
FRONTEND_QUERY_TIMEOUT_S = 45.0
TLS_HANDSHAKE_TIMEOUT_S = 5.0
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

    def _next_rv(self) -> str:
        self._resource_version += 1
        return str(self._resource_version)

    def _store_pod(self, pod: dict[str, Any], event_type: str) -> dict[str, Any]:
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
        # Caller holds self._cv. Persist then fan out so LIST-to-WATCH cannot drop
        # a mutation that happened before the watcher registered.
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
    raw = query.get("timeoutSeconds", ["30"])[0]
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _matches_labels(labels: Mapping[str, str], query: dict[str, list[str]]) -> bool:
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
    data = json.dumps(payload).encode("utf-8") + b"\n"
    handler.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
    handler.wfile.flush()


def _end_chunks(handler: BaseHTTPRequestHandler) -> None:
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()


def compile_ext_proc_protos(output_dir: Path) -> None:
    """Compile vendored ext_proc protos into tmp_path. Never writes the source tree."""
    try:
        from grpc_tools import protoc
    except ImportError as exc:
        raise RuntimeError(
            "grpcio-tools is required to generate ext_proc stubs into tmp_path"
        ) from exc

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
            return decode_epp_token_data(
                _extract_unique_terminal_request_body(pb2, responses)
            )
        finally:
            stream.close()
            channel.close()


def collect_frontend_query_token_ids(
    frontend_url: str,
    payload: dict[str, Any],
    *,
    path: str,
    request_id: str,
    timeout_s: float = FRONTEND_QUERY_TIMEOUT_S,
) -> list[int]:
    import requests

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
            body = response.text
    return _parse_frontend_query_sse(body)


def _parse_frontend_query_sse(body: str) -> list[int]:
    """Require a complete query_instance_id SSE stream before comparing tokens."""
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
        return collect_frontend_query_token_ids(
            self.frontend_url, payload, path=path, request_id=request_id
        )

    def collect_epp_tokens(
        self, payload: dict[str, Any], *, path: str, request_id: str
    ) -> list[int]:
        if self.driver is None:
            raise RuntimeError("parity stack has not started")
        return self.driver.collect_token_data(payload, path=path, request_id=request_id)

    def __enter__(self) -> EppFrontendParityStack:
        from tests.router.mocker_process import MockerProcess

        try:
            return self._start(MockerProcess)
        except Exception:
            self._stack.close()
            raise

    def _start(self, mocker_cls) -> EppFrontendParityStack:
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
        self._wait_for_frontend_models()
        if self.kube_api.failures:
            raise RuntimeError(
                "Loopback Kubernetes API recorded unexpected requests: "
                + "; ".join(self.kube_api.failures)
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
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

    def _wait_for_frontend_models(self) -> None:
        import requests

        deadline = time.monotonic() + PROCESS_TIMEOUT_S
        last_error = "frontend /v1/models not checked"
        with requests.Session() as session:
            session.trust_env = False
            while time.monotonic() < deadline:
                try:
                    response = session.get(f"{self.frontend_url}/v1/models", timeout=2)
                    if check_http_ok(response):
                        payload = response.json()
                        models = (
                            payload.get("data") if isinstance(payload, dict) else None
                        )
                        if isinstance(models, list) and models:
                            return
                        last_error = "frontend /v1/models returned no models"
                except (requests.RequestException, ValueError) as exc:
                    last_error = str(exc)
                time.sleep(0.25)
        raise TimeoutError(f"Frontend models were not ready: {last_error}")


def unique_request_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"
