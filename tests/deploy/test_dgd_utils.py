# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx
import pytest
import requests
import yaml

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


def test_get_pods_retries_transient_vcluster_disconnect(monkeypatch, tmp_path) -> None:
    deployment = managed_deployment(tmp_path)
    pod = MagicMock()
    get_pods = MagicMock(
        side_effect=[httpx.RemoteProtocolError("tunnel disconnected"), [pod]]
    )
    sleep = MagicMock()
    monkeypatch.setattr("tests.deploy.dgd_utils.kr8s.get", get_pods)
    monkeypatch.setattr("tests.deploy.dgd_utils.time.sleep", sleep)

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
    monkeypatch.setattr("tests.deploy.dgd_utils.asyncio.sleep", sleep)

    await deployment._delete_deployment()

    assert deployment._custom_api.delete_namespaced_custom_object.await_count == 2
    sleep.assert_awaited_once_with(5)


async def test_context_exit_preserves_original_error_when_cleanup_fails(
    tmp_path,
) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._cleanup = AsyncMock(
        side_effect=httpx.ConnectError("tunnel unavailable")
    )

    result = await deployment.__aexit__(ValueError, ValueError("test failed"), None)

    assert result is False


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
