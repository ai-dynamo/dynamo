# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


async def test_context_exit_reraises_unexpected_cleanup_error(tmp_path) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._cleanup = AsyncMock(side_effect=RuntimeError("cleanup defect"))

    with pytest.raises(RuntimeError, match="cleanup defect"):
        await deployment.__aexit__(ValueError, ValueError("test failed"), None)


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
    request_sender = MagicMock(side_effect=[transport_error, response])
    sleep = MagicMock()
    monkeypatch.setattr("tests.deploy.dgd_utils.time.sleep", sleep)

    result = deployment.send_request_with_port_forward_retry(
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
    original_port_forward.stop.assert_called_once_with()
    deployment.port_forward.assert_called_once()
    sleep.assert_called_once_with(5)


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


@pytest.mark.parametrize("stage", ["success", "setup", "call", "skip"])
async def test_discovery_capture_precedes_deletion_only_on_failure(tmp_path, stage):
    deployment = managed_deployment(tmp_path)
    events = []

    async def capture():
        events.append("capture")

    async def delete():
        events.append("delete")

    deployment._capture_discovery_state = capture
    deployment._delete_deployment = delete
    deployment._get_service_logs = MagicMock()
    if stage == "setup":
        original = ValueError("setup failed")
        deployment._init_kubernetes = AsyncMock(side_effect=original)
        with pytest.raises(ValueError) as raised:
            await deployment.__aenter__()
        assert raised.value is original
    elif stage == "call":
        original = ValueError("call failed")
        with pytest.raises(ValueError) as raised:
            try:
                raise original
            except ValueError as error:
                assert await deployment.__aexit__(type(error), error, None) is False
                raise
        assert raised.value is original
    elif stage == "skip":
        await deployment.__aexit__(
            pytest.skip.Exception, pytest.skip.Exception("skip"), None
        )
    else:
        await deployment.__aexit__(None, None, None)
    assert events == (
        ["capture", "delete"] if stage in ("setup", "call") else ["delete"]
    )


@pytest.mark.parametrize("fault", [None, "timeout", "write_error", "api_error"])
async def test_discovery_snapshot_keeps_partial_state_and_allows_cleanup(
    monkeypatch, tmp_path, fault
):
    deployment = managed_deployment(tmp_path)
    metadata = {
        "name": "worker",
        "resourceVersion": "123",
        "uid": "pod-uid",
        "labels": {"app": "worker"},
        "ownerReferences": [{"uid": "owner-uid"}],
    }
    body = {
        "items": [
            {
                "metadata": metadata,
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                "endpoints": [{"conditions": {"ready": True}}],
            }
        ]
    }

    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    async with client.ApiClient() as api_client:
        deployment._core_api = SimpleNamespace(
            api_client=api_client,
            list_namespaced_pod=AsyncMock(return_value=body),
            list_namespaced_service=AsyncMock(return_value=body),
        )
        deployment._custom_api = SimpleNamespace(
            list_namespaced_custom_object=AsyncMock(return_value=body)
        )
        endpoints = AsyncMock(return_value=body)
        monkeypatch.setattr(
            dgd_utils.client,
            "DiscoveryV1Api",
            lambda _: SimpleNamespace(list_namespaced_endpoint_slice=endpoints),
        )
        monkeypatch.setattr(dgd_utils, "DISCOVERY_RESOURCE_TIMEOUT", 0.01)
        if fault == "timeout":
            deployment._core_api.list_namespaced_pod = stalled
        elif fault == "api_error":
            deployment._core_api.list_namespaced_pod = AsyncMock(
                side_effect=RuntimeError("API unavailable")
            )
        elif fault == "write_error":
            directory = tmp_path / "discovery"
            directory.mkdir()
            (directory / "pods.json").mkdir()
        deployment._get_service_logs = MagicMock()
        deployment._delete_deployment = AsyncMock()
        original = ValueError("original test failure")
        assert await deployment.__aexit__(ValueError, original, None) is False
        deployment._delete_deployment.assert_awaited_once()
        for name in ("dgd", "dwm", "services", "endpointslices"):
            record = json.loads((tmp_path / "discovery" / f"{name}.json").read_text())
            assert record["response"] == body
            assert record["captured_at"]
        if fault in ("timeout", "api_error"):
            record = json.loads((tmp_path / "discovery" / "pods.json").read_text())
            assert "error" in record
        if fault is None:
            record = json.loads((tmp_path / "discovery" / "pods.json").read_text())
            assert record["response"] == body
        endpoints.assert_awaited_once_with("default", _request_timeout=0.01)
        calls = deployment._custom_api.list_namespaced_custom_object.call_args_list
        assert {call.args[3] for call in calls} == {
            "dynamographdeployments",
            "dynamoworkermetadatas",
        }


@pytest.mark.parametrize("fault", ["timeout", "directory_error"])
async def test_snapshot_failure_cannot_replace_original_error(
    monkeypatch, tmp_path, fault
):
    deployment = managed_deployment(tmp_path)
    deployment._get_service_logs = MagicMock()
    deployment._delete_deployment = AsyncMock()
    if fault == "timeout":

        async def stalled():
            await asyncio.Event().wait()

        deployment._capture_discovery_state = stalled
        monkeypatch.setattr(dgd_utils, "DISCOVERY_SNAPSHOT_TIMEOUT", 0.01)
    else:
        (tmp_path / "discovery").write_text("not a directory")
    original = ValueError("original")
    with pytest.raises(ValueError) as raised:
        try:
            raise original
        except ValueError:
            assert await deployment.__aexit__(ValueError, original, None) is False
            raise
    assert raised.value is original
    deployment._delete_deployment.assert_awaited_once()
