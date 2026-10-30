# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``contracts.RetuneKube`` over the official ``kubernetes`` client.

Every call is synchronous (callers on the event loop use
``asyncio.to_thread``), carries ``_request_timeout=(connect, read)`` and
raises the ``contracts.Kube*`` errors instead of client exceptions. The
error carries the HTTP status as ``error.status`` (``None`` for a network
error or timeout), so a caller can tell 403 from the rest.

Custom resources (the DGD at v1beta1, DGDR v1beta2 Requests, Runs and
Candidates) come back as the API server's JSON. Leases are read without the
client's model deserialization (``_preload_content=False``) so a Lease dict
round-trips unchanged: deserializing turns ``renewTime`` into a ``datetime``
whose ``isoformat`` drops a zero fraction, which a later replace would send
back in a form the ``MicroTime`` parser rejects.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, Optional, TypeVar

from kubernetes import client
from kubernetes.client import ApiException

from dynamo.planner.retune.live.contracts import (
    DGDR_CANDIDATES,
    DGDR_GROUP,
    DGDR_REQUESTS,
    DGDR_RUNS,
    DGDR_VERSION,
    KubeAlreadyExists,
    KubeConflict,
    KubeError,
    KubeInvalid,
    KubeNotFound,
    PodInfo,
)

logger = logging.getLogger(__name__)

DGD_VERSION = "v1beta1"
DGD_PLURAL = "dynamographdeployments"
# The frontend's container and HTTP port names (deploy/operator/internal/consts).
MAIN_CONTAINER_NAME = "main"
HTTP_PORT_NAME = "http"
FIELD_MANAGER = "dynamo-planner-retune"

_T = TypeVar("_T")
_Operation = Literal["read", "create", "update", "delete"]


def request_timeout(timeout_s: float) -> tuple[int, int]:
    """The client's ``(connect, read)`` timeout for ``timeout_s``, at least 1 s each."""

    if not math.isfinite(timeout_s) or timeout_s <= 0:
        seconds = 1
    else:
        seconds = max(1, int(math.ceil(timeout_s)))
    return (seconds, seconds)


def _kube_error(error: Exception, operation: _Operation, what: str) -> KubeError:
    """Map a client exception to the contracts error for ``operation``."""

    if isinstance(error, ApiException):
        status = error.status
        detail = _api_message(error)
        message = f"{operation} {what}: HTTP {status}: {detail}"
        mapped: KubeError
        if status == 404:
            mapped = KubeNotFound(message)
        elif status == 409 and operation == "create":
            mapped = KubeAlreadyExists(message)
        elif status == 409:
            mapped = KubeConflict(message)
        elif status in (400, 422):
            mapped = KubeInvalid(message)
        else:
            mapped = KubeError(message)
        mapped.status = status  # type: ignore[attr-defined]
        return mapped
    mapped = KubeError(f"{operation} {what}: {type(error).__name__}: {error}")
    mapped.status = None  # type: ignore[attr-defined]
    return mapped


def _api_message(error: ApiException) -> str:
    """The ``Status.message`` of an API error body, else its reason."""

    body = error.body
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str) and body:
        try:
            parsed = json.loads(body)
        except ValueError:
            return body[:500]
        if isinstance(parsed, dict) and parsed.get("message"):
            return str(parsed["message"])[:500]
    return str(error.reason or "")


def _pod_info(pod: Any) -> PodInfo:
    """``PodInfo`` of a ``V1Pod``."""

    metadata = pod.metadata
    status = pod.status
    ready = any(
        condition.type == "Ready" and condition.status == "True"
        for condition in (status.conditions if status is not None else None) or ()
    )
    return PodInfo(
        name=metadata.name,
        uid=metadata.uid,
        ip=status.pod_ip if status is not None else None,
        ready=ready,
        terminating=metadata.deletion_timestamp is not None,
        labels=dict(metadata.labels or {}),
        http_port=_http_port(pod.spec),
        phase=status.phase if status is not None else None,
    )


def _http_port(spec: Any) -> Optional[int]:
    """The ``http`` container port of the main container, else of any container."""

    containers = list((spec.containers if spec is not None else None) or ())
    containers.sort(key=lambda container: container.name != MAIN_CONTAINER_NAME)
    for container in containers:
        for port in container.ports or ():
            if port.name == HTTP_PORT_NAME:
                return int(port.container_port)
    return None


def _decode(response: Any) -> dict:
    """The JSON body of a ``_preload_content=False`` response."""

    data = response.data
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data) if data else {}


class KubernetesRetuneKube:
    """``contracts.RetuneKube`` for one namespace.

    ``api_client`` defaults to a new ``ApiClient`` on the default
    configuration, which ``KubernetesAPI`` loads (in-cluster or kubeconfig);
    ``from_kube_api`` reuses that connector's client instead.
    """

    def __init__(self, namespace: str, api_client: Optional[client.ApiClient] = None):
        if not namespace:
            raise ValueError("namespace must be set")
        self.namespace = namespace
        self.api_client = api_client if api_client is not None else client.ApiClient()
        self._custom = client.CustomObjectsApi(self.api_client)
        self._core = client.CoreV1Api(self.api_client)
        self._coordination = client.CoordinationV1Api(self.api_client)

    @classmethod
    def from_kube_api(cls, kube_api: Any) -> KubernetesRetuneKube:
        """Share a ``KubernetesAPI``'s client and namespace."""

        return cls(
            kube_api.current_namespace,
            api_client=kube_api.custom_api.api_client,
        )

    # ------------------------------------------------------------ helpers

    def _call(self, operation: _Operation, what: str, function: Callable[[], _T]) -> _T:
        try:
            return function()
        except KubeError:
            raise
        except Exception as error:  # ApiException, urllib3 and socket errors
            raise _kube_error(error, operation, what) from error

    def _get_custom(
        self, version: str, plural: str, name: str, timeout_s: float
    ) -> Optional[dict]:
        try:
            return self._call(
                "read",
                f"{plural}/{name}",
                lambda: self._custom.get_namespaced_custom_object(
                    group=DGDR_GROUP,
                    version=version,
                    namespace=self.namespace,
                    plural=plural,
                    name=name,
                    _request_timeout=request_timeout(timeout_s),
                ),
            )
        except KubeNotFound:
            return None

    def _list_custom(
        self, version: str, plural: str, label_selector: str, timeout_s: float
    ) -> list[dict]:
        result = self._call(
            "read",
            f"{plural}?labelSelector={label_selector}",
            lambda: self._custom.list_namespaced_custom_object(
                group=DGDR_GROUP,
                version=version,
                namespace=self.namespace,
                plural=plural,
                label_selector=label_selector,
                _request_timeout=request_timeout(timeout_s),
            ),
        )
        return list(result.get("items") or [])

    # ----------------------------------------------------------------- DGD

    def get_dgd(self, name: str, *, timeout_s: float = 10.0) -> dict:
        found = self._get_custom(DGD_VERSION, DGD_PLURAL, name, timeout_s)
        if found is None:
            error = KubeNotFound(
                f"DynamoGraphDeployment {self.namespace}/{name} not found"
            )
            error.status = 404  # type: ignore[attr-defined]
            raise error
        return found

    # ---------------------------------------------------------------- pods

    def list_pods(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[PodInfo]:
        result = self._call(
            "read",
            f"pods?labelSelector={label_selector}",
            lambda: self._core.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector,
                _request_timeout=request_timeout(timeout_s),
            ),
        )
        return [_pod_info(pod) for pod in result.items or ()]

    # ---------------------------------------------------- DGDR v1beta2 API

    def create_request(
        self,
        body: dict,
        *,
        dry_run: bool = False,
        timeout_s: float = 10.0,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "field_validation": "Strict",
            "field_manager": FIELD_MANAGER,
            "_request_timeout": request_timeout(timeout_s),
        }
        if dry_run:
            kwargs["dry_run"] = "All"
        name = (body.get("metadata") or {}).get("name", "")
        return self._call(
            "create",
            f"{DGDR_REQUESTS}/{name}",
            lambda: self._custom.create_namespaced_custom_object(
                group=DGDR_GROUP,
                version=DGDR_VERSION,
                namespace=self.namespace,
                plural=DGDR_REQUESTS,
                body=body,
                **kwargs,
            ),
        )

    def get_request(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        return self._get_custom(DGDR_VERSION, DGDR_REQUESTS, name, timeout_s)

    def list_requests(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[dict]:
        return self._list_custom(DGDR_VERSION, DGDR_REQUESTS, label_selector, timeout_s)

    def delete_request(
        self,
        name: str,
        *,
        propagation: Literal["Background", "Foreground"] = "Background",
        timeout_s: float = 10.0,
    ) -> None:
        try:
            self._call(
                "delete",
                f"{DGDR_REQUESTS}/{name}",
                lambda: self._custom.delete_namespaced_custom_object(
                    group=DGDR_GROUP,
                    version=DGDR_VERSION,
                    namespace=self.namespace,
                    plural=DGDR_REQUESTS,
                    name=name,
                    propagation_policy=propagation,
                    _request_timeout=request_timeout(timeout_s),
                ),
            )
        except KubeNotFound:
            pass

    def get_run(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        return self._get_custom(DGDR_VERSION, DGDR_RUNS, name, timeout_s)

    def list_candidates(
        self, label_selector: str, *, timeout_s: float = 10.0
    ) -> list[dict]:
        return self._list_custom(
            DGDR_VERSION, DGDR_CANDIDATES, label_selector, timeout_s
        )

    # -------------------------------------------------------------- Leases

    def get_lease(self, name: str, *, timeout_s: float = 10.0) -> Optional[dict]:
        try:
            return self._call(
                "read",
                f"leases/{name}",
                lambda: _decode(
                    self._coordination.read_namespaced_lease(
                        name=name,
                        namespace=self.namespace,
                        _preload_content=False,
                        _request_timeout=request_timeout(timeout_s),
                    )
                ),
            )
        except KubeNotFound:
            return None

    def create_lease(self, body: dict, *, timeout_s: float = 10.0) -> dict:
        name = (body.get("metadata") or {}).get("name", "")
        return self._call(
            "create",
            f"leases/{name}",
            lambda: _decode(
                self._coordination.create_namespaced_lease(
                    namespace=self.namespace,
                    body=_lease_body(body),
                    field_manager=FIELD_MANAGER,
                    _preload_content=False,
                    _request_timeout=request_timeout(timeout_s),
                )
            ),
        )

    def replace_lease(self, name: str, body: dict, *, timeout_s: float = 10.0) -> dict:
        return self._call(
            "update",
            f"leases/{name}",
            lambda: _decode(
                self._coordination.replace_namespaced_lease(
                    name=name,
                    namespace=self.namespace,
                    body=_lease_body(body),
                    field_manager=FIELD_MANAGER,
                    _preload_content=False,
                    _request_timeout=request_timeout(timeout_s),
                )
            ),
        )

    def delete_lease(self, name: str, *, timeout_s: float = 10.0) -> None:
        """Missing Leases are not an error."""

        try:
            self._call(
                "delete",
                f"leases/{name}",
                lambda: self._coordination.delete_namespaced_lease(
                    name=name,
                    namespace=self.namespace,
                    _request_timeout=request_timeout(timeout_s),
                ),
            )
        except KubeNotFound:
            pass


def _lease_body(body: Mapping[str, Any]) -> dict:
    """A Lease body with its type fields set (the API server requires them)."""

    lease = dict(body)
    lease.setdefault("apiVersion", "coordination.k8s.io/v1")
    lease.setdefault("kind", "Lease")
    return lease
