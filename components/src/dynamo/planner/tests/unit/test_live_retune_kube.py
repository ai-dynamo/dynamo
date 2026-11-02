# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``KubernetesRetuneKube``.

The tests run the real ``kubernetes`` API classes and replace only
the REST transport, so they check the URLs, query parameters, timeouts and
bodies the client library actually sends, and the error mapping.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from kubernetes import client
from kubernetes.client import ApiException

from dynamo.planner.retune.live.contracts import (
    KubeAlreadyExists,
    KubeError,
    KubeInvalid,
    PodInfo,
    RetuneKube,
)
from dynamo.planner.retune.live.kube import KubernetesRetuneKube, request_timeout

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

NS = "test-ns"


class FakeResponse:
    def __init__(self, body: Any, status: int = 200) -> None:
        self.status = status
        self.reason = "OK"
        self.data = body if isinstance(body, (str, bytes)) else json.dumps(body)

    def getheaders(self) -> dict:
        return {"content-type": "application/json"}

    def getheader(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self.getheaders().get(name.lower(), default)


class Transport:
    """Stands in for ``RESTClientObject.request``: records and replies."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies: list[Any] = []

    def reply(self, value: Any) -> None:
        self.replies.append(value)

    def __call__(
        self,
        method,
        url,
        query_params=None,
        headers=None,
        post_params=None,
        body=None,
        _preload_content=True,
        _request_timeout=None,
    ):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "query": dict(query_params or []),
                "body": body,
                "preload": _preload_content,
                "timeout": _request_timeout,
            }
        )
        value = self.replies.pop(0) if self.replies else {}
        if isinstance(value, Exception):
            raise value
        if isinstance(value, FakeResponse):
            return value
        return FakeResponse(value)


def api_error(status: int, message: str = "nope") -> ApiException:
    error = ApiException(status=status, reason="Error")
    error.body = json.dumps({"kind": "Status", "message": message})
    return error


@pytest.fixture
def kube():
    api_client = client.ApiClient(client.Configuration())
    transport = Transport()
    api_client.rest_client.request = transport
    retune_kube = KubernetesRetuneKube(NS, api_client=api_client)
    yield retune_kube, transport
    api_client.close()


def path(call: dict) -> str:
    return call["url"].split("://", 1)[-1].split("/", 1)[-1]


# ---------------------------------------------------------------- client


def test_request_timeout_is_a_tuple_of_whole_seconds():
    assert request_timeout(10.0) == (10, 10)
    assert request_timeout(2.5) == (3, 3)
    assert request_timeout(0.2) == (1, 1)
    assert request_timeout(0.0) == (1, 1)
    assert request_timeout(float("nan")) == (1, 1)


def test_satisfies_the_protocol(kube):
    retune_kube, _ = kube
    assert isinstance(retune_kube, KubernetesRetuneKube)
    _: RetuneKube = retune_kube
    assert retune_kube.namespace == NS
    with pytest.raises(ValueError):
        KubernetesRetuneKube("", api_client=retune_kube.api_client)


def test_create_request_is_strict_and_dry_run_on_request(kube):
    retune_kube, transport = kube
    body = {"metadata": {"name": "r1"}, "spec": {"model": {"name": "m"}}}
    transport.reply({**body, "metadata": {"name": "r1", "uid": "x"}})
    created = retune_kube.create_request(body, timeout_s=2.5)
    assert created["metadata"]["uid"] == "x"
    call = transport.calls[-1]
    assert call["method"] == "POST"
    assert (
        path(call)
        == f"apis/nvidia.com/v1beta2/namespaces/{NS}/dynamographdeploymentrequests"
    )
    assert call["query"]["fieldValidation"] == "Strict"
    assert call["query"]["fieldManager"] == "dynamo-planner-retune"
    assert "dryRun" not in call["query"]
    assert call["body"] == body
    assert call["timeout"] == (3, 3)

    transport.reply(body)
    retune_kube.create_request(body, dry_run=True)
    assert transport.calls[-1]["query"]["dryRun"] == "All"
    assert transport.calls[-1]["query"]["fieldValidation"] == "Strict"


@pytest.mark.parametrize(
    "status, error_type",
    [
        (409, KubeAlreadyExists),
        (422, KubeInvalid),
        (400, KubeInvalid),
        (403, KubeError),
        (500, KubeError),
    ],
)
def test_create_request_maps_api_errors(kube, status, error_type):
    retune_kube, transport = kube
    transport.reply(api_error(status, "strict decoding error: unknown field"))
    with pytest.raises(error_type) as raised:
        retune_kube.create_request({"metadata": {"name": "r1"}})
    assert raised.value.status == status
    assert f"HTTP {status}" in str(raised.value)
    assert "unknown field" in str(raised.value)
    if error_type is KubeError:
        assert type(raised.value) is KubeError


def test_network_errors_become_kube_errors(kube):
    retune_kube, transport = kube
    # Imported here: the marker-report hook collects tests without urllib3.
    from urllib3.exceptions import ReadTimeoutError

    transport.reply(ReadTimeoutError(None, "url", "timed out"))
    with pytest.raises(KubeError) as raised:
        retune_kube.get_request("r1")
    assert raised.value.status is None
    assert "ReadTimeoutError" in str(raised.value)


def test_requests_runs_and_candidates(kube):
    retune_kube, transport = kube
    transport.reply({"metadata": {"name": "r1"}})
    assert retune_kube.get_request("r1") == {"metadata": {"name": "r1"}}
    transport.reply(api_error(404))
    assert retune_kube.get_request("missing") is None

    transport.reply({"items": [{"metadata": {"name": "r1"}}]})
    assert retune_kube.list_requests("a=b,c=d") == [{"metadata": {"name": "r1"}}]
    call = transport.calls[-1]
    assert (
        path(call)
        == f"apis/nvidia.com/v1beta2/namespaces/{NS}/dynamographdeploymentrequests"
    )
    assert call["query"]["labelSelector"] == "a=b,c=d"

    transport.reply({"metadata": {"name": "r1-1"}})
    assert retune_kube.get_run("r1-1") == {"metadata": {"name": "r1-1"}}
    assert path(transport.calls[-1]).endswith("dynamographdeploymentruns/r1-1")
    transport.reply(api_error(404))
    assert retune_kube.get_run("gone") is None

    transport.reply({"items": []})
    assert retune_kube.list_candidates("nvidia.com/dgdr-run-uid=u") == []
    call = transport.calls[-1]
    assert path(call).endswith("dynamographdeploymentcandidates")
    assert call["query"]["labelSelector"] == "nvidia.com/dgdr-run-uid=u"

    transport.reply({})
    retune_kube.delete_request("r1", propagation="Foreground", timeout_s=7)
    call = transport.calls[-1]
    assert call["method"] == "DELETE"
    assert path(call).endswith("dynamographdeploymentrequests/r1")
    assert call["query"]["propagationPolicy"] == "Foreground"
    assert call["timeout"] == (7, 7)
    transport.reply(api_error(404))
    retune_kube.delete_request("gone")  # missing is not an error
    transport.reply(api_error(500))
    with pytest.raises(KubeError):
        retune_kube.delete_request("r1")


def _pod(
    name: str,
    *,
    ready: bool = True,
    deleting: bool = False,
    containers: Optional[list] = None,
    ip: Optional[str] = "10.0.0.5",
) -> dict:
    return {
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "labels": {"nvidia.com/dynamo-component-type": "frontend"},
            **({"deletionTimestamp": "2026-10-28T00:00:00Z"} if deleting else {}),
        },
        "spec": {
            "containers": containers
            if containers is not None
            else [
                {"name": "sidecar", "ports": [{"name": "http", "containerPort": 9999}]},
                {
                    "name": "main",
                    "ports": [
                        {"name": "metrics", "containerPort": 9090},
                        {"name": "http", "containerPort": 8000},
                    ],
                },
            ]
        },
        "status": {
            "podIP": ip,
            "conditions": [
                {"type": "PodScheduled", "status": "True"},
                {"type": "Ready", "status": "True" if ready else "False"},
            ],
        },
    }


def test_list_pods_builds_pod_info(kube):
    retune_kube, transport = kube
    transport.reply(
        {
            "kind": "PodList",
            "apiVersion": "v1",
            "metadata": {},
            "items": [
                _pod("a"),
                _pod("b", ready=False, deleting=True, ip=None),
                _pod(
                    "c",
                    containers=[
                        {
                            "name": "other",
                            "ports": [{"name": "http", "containerPort": 8080}],
                        }
                    ],
                ),
                _pod("d", containers=[{"name": "main"}]),
            ],
        }
    )
    pods = retune_kube.list_pods("x=y", timeout_s=3)
    call = transport.calls[-1]
    assert path(call) == f"api/v1/namespaces/{NS}/pods"
    assert call["query"]["labelSelector"] == "x=y"
    assert call["timeout"] == (3, 3)
    assert pods[0] == PodInfo(
        name="a",
        uid="uid-a",
        ip="10.0.0.5",
        ready=True,
        terminating=False,
        labels={"nvidia.com/dynamo-component-type": "frontend"},
        http_port=8000,  # the main container's port wins
    )
    assert (pods[1].ready, pods[1].terminating, pods[1].ip) == (False, True, None)
    assert pods[2].http_port == 8080
    assert pods[3].http_port is None


def test_from_kube_api_shares_the_connector_client():
    api_client = client.ApiClient(client.Configuration())
    kube_api = SimpleNamespace(
        current_namespace="ns2",
        custom_api=client.CustomObjectsApi(api_client),
    )
    retune_kube = KubernetesRetuneKube.from_kube_api(kube_api)
    assert retune_kube.namespace == "ns2"
    assert retune_kube.api_client is api_client
    api_client.close()
