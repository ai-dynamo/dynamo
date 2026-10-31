# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import aiohttp
import httpx
import kr8s
import pytest
import requests
import yaml
from kubernetes_asyncio import client

from tests.deploy import dgd_utils
from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def managed_deployment(tmp_path) -> ManagedDeployment:
    spec = SimpleNamespace(name="test-deployment", services=[], api_version="v1beta1")
    return ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,  # type: ignore[arg-type]
        namespace="default",
    )


def test_logging_config_reads_existing_v1beta1_env(tmp_path) -> None:
    """Recognize JSONL logging already declared in a v1beta1 manifest."""
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "logging-test"},
        "spec": {
            "components": [],
            "env": [{"name": "DYN_LOGGING_JSONL", "value": "1"}],
        },
    }
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    deployment_spec = DeploymentSpec(str(manifest_path))

    assert deployment_spec.get_logging_config()["jsonl_enabled"] is True


@pytest.mark.parametrize(
    "connection_error",
    [
        httpx.RemoteProtocolError("tunnel disconnected"),
        kr8s.APITimeoutError("vCluster request timed out"),
    ],
)
def test_get_pods_retries_transient_vcluster_disconnect(
    monkeypatch, tmp_path, connection_error
) -> None:
    deployment = managed_deployment(tmp_path)
    pod = MagicMock()
    get_pods = MagicMock(side_effect=[connection_error, [pod]])
    sleep = MagicMock()
    monkeypatch.setattr("tests.deploy.dgd_utils.kr8s.get", get_pods)
    monkeypatch.setattr("tests.deploy.vcluster_utils.time.sleep", sleep)

    result = deployment.get_pods(["Frontend"])

    assert result == {"Frontend": [pod]}
    assert get_pods.call_count == 2
    sleep.assert_called_once_with(5)


async def test_delete_deployment_retries_vcluster_connection_failure(
    monkeypatch, tmp_path
) -> None:
    deployment = managed_deployment(tmp_path)
    connection_key = MagicMock(host="127.0.0.1", port=8443, ssl=True)
    deployment._custom_api = MagicMock()
    deployment._custom_api.delete_namespaced_custom_object = AsyncMock(
        side_effect=[
            aiohttp.ClientConnectorError(
                connection_key,
                ConnectionRefusedError(111, "vCluster tunnel unavailable"),
            ),
            None,
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("tests.deploy.vcluster_utils.asyncio.sleep", sleep)

    await deployment._delete_deployment()

    assert deployment._custom_api.delete_namespaced_custom_object.await_count == 2
    sleep.assert_awaited_once_with(5)


@pytest.mark.parametrize(
    "cleanup_error",
    [
        httpx.ConnectError("tunnel unavailable"),
        kr8s.APITimeoutError("vCluster request timed out"),
    ],
)
async def test_context_exit_preserves_original_error_when_cleanup_fails(
    tmp_path, cleanup_error
) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._cleanup = AsyncMock(side_effect=cleanup_error)

    result = await deployment.__aexit__(ValueError, ValueError("test failed"), None)

    assert result is False


async def test_context_enter_preserves_setup_error_when_cleanup_times_out(
    tmp_path,
) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._init_kubernetes = AsyncMock(side_effect=ValueError("setup failed"))
    deployment._cleanup = AsyncMock(
        side_effect=kr8s.APITimeoutError("vCluster request timed out")
    )

    with pytest.raises(ValueError, match="setup failed"):
        await deployment.__aenter__()


async def test_context_exit_preserves_primary_error_for_cleanup_defect(
    tmp_path,
) -> None:
    deployment = managed_deployment(tmp_path)
    cleanup_error = RuntimeError("cleanup defect")
    deployment._cleanup = AsyncMock(side_effect=cleanup_error)

    result = await deployment.__aexit__(ValueError, ValueError("test failed"), None)

    assert result is False
    assert deployment.cleanup_errors == [cleanup_error]


@pytest.mark.parametrize(
    "transport_error",
    [
        requests.ConnectionError("forward dropped"),
        requests.Timeout("forward stalled"),
    ],
)
def test_request_rebuilds_port_forward_after_transport_failure(
    monkeypatch, tmp_path, transport_error
) -> None:
    deployment = managed_deployment(tmp_path)
    original_port_forward = MagicMock(local_port=31001)
    replacement_port_forward = MagicMock(local_port=31002)
    deployment.port_forward = MagicMock(return_value=replacement_port_forward)
    response = MagicMock(spec=requests.Response)
    request_sender = MagicMock(side_effect=[transport_error, response, response])
    sleep = MagicMock()
    monkeypatch.setattr("tests.deploy.dgd_utils.time.sleep", sleep)

    result, active_port_forward = deployment.send_request_with_port_forward_retry(
        pod=MagicMock(),
        remote_port=8000,
        endpoint="/v1/chat/completions",
        payload={"model": "test"},
        timeout=120,
        port_forward=original_port_forward,
        request_sender=request_sender,
    )

    assert result is response
    assert (
        request_sender.call_args_list[0].args[0].startswith("http://localhost:31001/")
    )
    assert (
        request_sender.call_args_list[1].args[0].startswith("http://localhost:31002/")
    )
    assert active_port_forward is replacement_port_forward
    next_result, next_port_forward = deployment.send_request_with_port_forward_retry(
        pod=MagicMock(),
        remote_port=8000,
        endpoint="/v1/chat/completions",
        payload={"model": "test"},
        timeout=120,
        port_forward=active_port_forward,
        request_sender=request_sender,
    )
    assert next_result is response
    assert next_port_forward is replacement_port_forward
    assert request_sender.call_count == 3
    assert (
        request_sender.call_args_list[2].args[0].startswith("http://localhost:31002/")
    )
    original_port_forward.stop.assert_called_once_with()
    deployment.port_forward.assert_called_once()
    sleep.assert_called_once_with(5)


def test_request_rebuilds_port_forward_after_stream_body_failure(
    monkeypatch, tmp_path
) -> None:
    deployment = managed_deployment(tmp_path)
    original_port_forward = MagicMock(local_port=31001)
    replacement_port_forward = MagicMock(local_port=31002)
    deployment.port_forward = MagicMock(return_value=replacement_port_forward)
    dropped_response = MagicMock(spec=requests.Response)
    type(dropped_response).content = PropertyMock(
        side_effect=requests.ConnectionError("stream dropped after headers")
    )
    response = MagicMock(spec=requests.Response)
    type(response).content = PropertyMock(return_value=b"data: [DONE]\n")
    request_sender = MagicMock(side_effect=[dropped_response, response])
    monkeypatch.setattr("tests.deploy.dgd_utils.time.sleep", MagicMock())

    result, active_port_forward = deployment.send_request_with_port_forward_retry(
        pod=MagicMock(),
        remote_port=8000,
        endpoint="/v1/chat/completions",
        payload={"model": "test", "stream": True},
        timeout=120,
        port_forward=original_port_forward,
        request_sender=request_sender,
    )

    assert result is response
    assert active_port_forward is replacement_port_forward
    dropped_response.close.assert_called_once_with()
    original_port_forward.stop.assert_called_once_with()
    deployment.port_forward.assert_called_once()


def test_request_reraises_unexpected_port_forward_stop_error(tmp_path) -> None:
    deployment = managed_deployment(tmp_path)
    original_port_forward = MagicMock(local_port=31001)
    original_port_forward.stop.side_effect = ValueError("stop defect")
    request_sender = MagicMock(side_effect=requests.ConnectionError("forward dropped"))

    with pytest.raises(ValueError, match="stop defect"):
        deployment.send_request_with_port_forward_retry(
            pod=MagicMock(),
            remote_port=8000,
            endpoint="/v1/chat/completions",
            payload={"model": "test"},
            timeout=120,
            port_forward=original_port_forward,
            request_sender=request_sender,
        )


async def test_in_flight_restart_preserves_bounded_previous_log(tmp_path) -> None:
    """Keep a bounded previous-instance log before Kubernetes rotates again."""
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=SimpleNamespace(name="test-dgd"),
        namespace="default",
    )
    terminated = SimpleNamespace(reason="Error", exit_code=1)
    container_status = SimpleNamespace(
        name="main",
        restart_count=1,
        last_state=SimpleNamespace(terminated=terminated),
    )
    pod = SimpleNamespace(
        metadata=SimpleNamespace(name="worker-0"),
        status=SimpleNamespace(container_statuses=[container_status]),
    )
    deployment._core_api = SimpleNamespace(
        list_namespaced_pod=AsyncMock(return_value=SimpleNamespace(items=[pod])),
        read_namespaced_pod_log=AsyncMock(
            return_value="first line\nsecond line\nthird line\n"
        ),
    )

    warnings = await deployment._dump_in_flight_restart_logs(prev_log_tail_lines=2)

    assert len(warnings) == 1
    assert "first line" not in warnings[0]
    assert "second line" in warnings[0]
    assert "third line" in warnings[0]
    preserved = tmp_path / "restarts" / "worker-0.main.restart-1.previous.log"
    assert preserved.read_text() == "first line\nsecond line\nthird line\n"
    deployment._core_api.read_namespaced_pod_log.assert_awaited_once_with(
        name="worker-0",
        namespace="default",
        container="main",
        previous=True,
        tail_lines=50000,
    )


@pytest.mark.parametrize(
    ("failed", "capture_behavior", "expected_events"),
    [
        (False, "complete", ["service-logs", "delete"]),
        (
            True,
            "complete",
            ["capture-start", "capture-done", "service-logs", "delete"],
        ),
        (True, "timeout", ["capture-start", "service-logs", "delete"]),
        (True, "cancel", ["capture-start", "service-logs", "delete"]),
    ],
    ids=["success", "failure", "capture-timeout", "capture-cancelled"],
)
async def test_discovery_capture_and_cleanup(
    monkeypatch, tmp_path, failed, capture_behavior, expected_events
):
    deployment = managed_deployment(tmp_path)
    events = []

    # Isolate teardown from Kubernetes startup.
    monkeypatch.setattr(
        ManagedDeployment, "__aenter__", AsyncMock(return_value=deployment)
    )
    deployment._get_service_logs = MagicMock(
        side_effect=lambda: events.append("service-logs")
    )
    snapshot_timeout = 1 if capture_behavior == "cancel" else 0.01
    monkeypatch.setattr(dgd_utils, "DISCOVERY_SNAPSHOT_TIMEOUT", snapshot_timeout)

    async def capture():
        events.append("capture-start")
        if capture_behavior in {"timeout", "cancel"}:
            await asyncio.Event().wait()
        await asyncio.sleep(0)
        events.append("capture-done")

    async def delete(*, fail_on_timeout=True):
        events.append("delete")

    deployment._capture_discovery_state = capture
    deployment._delete_deployment = delete

    if capture_behavior == "cancel":
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                async with deployment:
                    raise ValueError("inference failed")
    elif failed:
        with pytest.raises(ValueError, match="inference failed"):
            async with deployment:
                raise ValueError("inference failed")
    else:
        async with deployment:
            pass

    assert events == expected_events


class _FakeHttpResponse:
    """The parts of an aiohttp response that kubernetes_asyncio reads."""

    def __init__(self, payload, status=200, reason="OK"):
        self.status = status
        self.reason = reason
        self.headers = {"Content-Type": "application/json"}
        self._body = json.dumps(payload).encode()
        self.released = False

    async def read(self):
        return self._body

    async def text(self):
        return self._body.decode()

    def release(self):
        self.released = True


class _FakeHttpSession:
    def __init__(self, payloads):
        self.payloads = payloads
        self.responses = []

    async def request(self, method, url, **kwargs):
        resource = url.rsplit("/", 1)[-1].split("?", 1)[0]
        response = _FakeHttpResponse(*self.payloads[resource])
        self.responses.append(response)
        return response

    async def close(self):
        pass


async def _capture_discovery(tmp_path, payloads):
    deployment = managed_deployment(tmp_path)
    api_client = client.ApiClient(client.Configuration(host="http://kubernetes.test"))
    await api_client.rest_client.pool_manager.close()
    session = _FakeHttpSession(payloads)
    api_client.rest_client.pool_manager = session
    deployment._custom_api = client.CustomObjectsApi(api_client)
    deployment._core_api = client.CoreV1Api(api_client)

    try:
        await deployment._capture_discovery_state()
    finally:
        await api_client.close()

    assert len(session.responses) == 5
    assert all(response.released for response in session.responses)


_EMPTY_LIST = ({"apiVersion": "v1", "metadata": {}, "items": []},)


async def test_discovery_capture_records_endpointslice_without_endpoints(
    tmp_path,
):
    endpoint_slice = {
        "metadata": {
            "name": "frontend-abc12",
            "labels": {"kubernetes.io/service-name": "frontend"},
        },
        "addressType": "IPv4",
        "endpoints": None,
    }
    await _capture_discovery(
        tmp_path,
        {
            "dynamographdeployments": _EMPTY_LIST,
            "dynamoworkermetadatas": _EMPTY_LIST,
            "pods": _EMPTY_LIST,
            "services": _EMPTY_LIST,
            "endpointslices": (
                {
                    "apiVersion": "discovery.k8s.io/v1",
                    "kind": "EndpointSliceList",
                    "metadata": {},
                    "items": [endpoint_slice],
                },
            ),
        },
    )

    record = json.loads((tmp_path / "discovery" / "endpointslices.json").read_text())
    assert "error" not in record, record
    assert record["response"]["items"] == [endpoint_slice]


async def test_discovery_capture_records_api_error_body(tmp_path):
    forbidden = {
        "kind": "Status",
        "status": "Failure",
        "message": "pods is forbidden: cannot list resource pods",
        "code": 403,
    }

    await _capture_discovery(
        tmp_path,
        {
            "dynamographdeployments": _EMPTY_LIST,
            "dynamoworkermetadatas": _EMPTY_LIST,
            "pods": (forbidden, 403, "Forbidden"),
            "services": _EMPTY_LIST,
            "endpointslices": _EMPTY_LIST,
        },
    )

    record = json.loads((tmp_path / "discovery" / "pods.json").read_text())
    assert "response" not in record, record
    assert "403" in record["error"]
    assert forbidden["message"] in record["error"]


async def test_readiness_wait_deadline_covers_stalled_request(tmp_path, monkeypatch):
    deployment = managed_deployment(tmp_path)

    async def stalled_status(**kwargs):
        await asyncio.sleep(60)

    deployment._custom_api = SimpleNamespace(
        get_namespaced_custom_object=stalled_status
    )
    monkeypatch.setattr(
        deployment, "_get_pod_status_details", AsyncMock(return_value=[])
    )

    with pytest.raises(TimeoutError, match="failed to reach Ready=True"):
        async with asyncio.timeout(10):
            await deployment._wait_for_ready(timeout=1)
