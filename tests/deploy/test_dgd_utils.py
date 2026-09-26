# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

import asyncio
import builtins
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import aiohttp
import httpx
import kr8s
import pytest
import requests
import yaml

from tests.deploy import dgd_utils
from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment, SnapshotJob
from tests.deploy.vcluster_utils import VCLUSTER_CONNECTION_RETRY_LIMIT

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


@pytest.mark.parametrize(
    ("failed", "capture_behavior", "expected_events"),
    [
        (False, "complete", ["checkpoint-logs", "service-logs", "delete"]),
        (
            True,
            "complete",
            [
                "capture-start",
                "capture-done",
                "checkpoint-logs",
                "service-logs",
                "delete",
            ],
        ),
        (
            True,
            "timeout",
            ["capture-start", "checkpoint-logs", "service-logs", "delete"],
        ),
        (
            True,
            "cancel",
            ["capture-start", "checkpoint-logs", "service-logs", "delete"],
        ),
    ],
    ids=["success", "failure", "capture-timeout", "capture-cancelled"],
)
@pytest.mark.timeout(5)
async def test_discovery_capture_and_cleanup(
    monkeypatch, tmp_path, failed, capture_behavior, expected_events
):
    deployment = managed_deployment(tmp_path)
    events = []

    # Isolate teardown from Kubernetes startup.
    monkeypatch.setattr(
        ManagedDeployment, "__aenter__", AsyncMock(return_value=deployment)
    )

    def checkpoint_logs():
        events.append("checkpoint-logs")
        return {"checkpoint-source"}

    deployment._get_checkpoint_pod_logs = Mock(side_effect=checkpoint_logs)
    deployment._get_service_logs = MagicMock(
        side_effect=lambda **kwargs: events.append("service-logs")
    )
    snapshot_timeout = 1 if capture_behavior == "cancel" else 0.01
    monkeypatch.setattr(dgd_utils, "DISCOVERY_SNAPSHOT_TIMEOUT", snapshot_timeout)

    async def capture():
        events.append("capture-start")
        if capture_behavior in {"timeout", "cancel"}:
            await asyncio.Event().wait()
        await asyncio.sleep(0)
        events.append("capture-done")

    async def delete():
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
    deployment._get_service_logs.assert_called_once_with(
        exclude_pods={"checkpoint-source"}
    )


@pytest.mark.parametrize(
    "has_dgd_labels, failure_stage",
    [
        (True, None),
        (False, None),
        (False, "jobs"),
        (False, "pods"),
        (True, "manifest"),
        (True, "logs"),
        (True, "write"),
        (True, "previous_logs"),
    ],
)
async def test_cleanup_preserves_checkpoint_pod_logs_before_deletion(
    tmp_path, monkeypatch, has_dgd_labels, failure_stage
) -> None:
    deployment = managed_deployment(tmp_path)
    deployment.deployment_spec.services = [SimpleNamespace(name="worker")]
    list_jobs = Mock(
        return_value=[
            SimpleNamespace(
                raw={"metadata": {"name": "checkpoint-test", "uid": "job-uid"}}
            )
        ]
    )
    pod = Mock()
    pod.name = "checkpoint-test-source"
    pod.raw = {
        "metadata": {
            "name": pod.name,
            "labels": {
                "nvidia.com/snapshot-job": "checkpoint-test",
                "nvidia.com/snapshot-job-uid": "job-uid",
            },
        },
        "spec": {"containers": [{"name": "main"}]},
    }
    if has_dgd_labels:
        pod.raw["metadata"]["labels"].update(
            {
                "nvidia.com/dynamo-graph-deployment-name": "test-deployment",
                "nvidia.com/dynamo-component": "worker",
            }
        )
    pod.to_yaml.return_value = yaml.safe_dump(pod.raw)

    def pod_logs(**kwargs):
        if kwargs.get("previous"):
            if failure_stage == "previous_logs":
                raise kr8s.ServerError("no previous terminated container")
            return ["previous instance"]
        if failure_stage == "logs" and pod.logs.call_count == 1:
            raise kr8s.ServerError("temporary log fetch failure")
        return ["source startup error"]

    pod.logs.side_effect = pod_logs
    if failure_stage == "manifest":
        pod.to_yaml.side_effect = [
            RuntimeError("temporary manifest failure"),
            pod.to_yaml.return_value,
        ]
    if failure_stage == "write":

        def open_artifact(path, *args, **kwargs):
            if path == str(tmp_path / "checkpoint" / f"{pod.name}.main.log"):
                raise OSError("checkpoint log write failed")
            return builtins.open(path, *args, **kwargs)

        monkeypatch.setattr("tests.deploy.dgd_utils.open", open_artifact, raising=False)

    monkeypatch.setattr("tests.deploy.vcluster_utils.time.sleep", Mock())
    if failure_stage == "jobs":
        list_jobs.side_effect = [
            httpx.ConnectError("tunnel dropped"),
            list_jobs.return_value,
        ]
    pod_attempts = 0

    def get_pods(kind, *, namespace, label_selector):
        nonlocal pod_attempts
        if kind is SnapshotJob:
            return list_jobs(namespace=namespace, label_selector=label_selector)
        assert kind == "pods"
        assert namespace == "default"
        if label_selector == (
            "nvidia.com/snapshot-job=checkpoint-test,"
            "nvidia.com/snapshot-job-uid=job-uid"
        ):
            pod_attempts += 1
            if failure_stage == "pods" and pod_attempts == 1:
                raise httpx.ConnectError("tunnel dropped")
            return [pod]
        if has_dgd_labels and label_selector == (
            "nvidia.com/dynamo-graph-deployment-name=test-deployment,"
            "nvidia.com/dynamo-component=worker"
        ):
            return [pod]
        return []

    monkeypatch.setattr("tests.deploy.dgd_utils.kr8s.get", get_pods)
    metrics = Mock()
    monkeypatch.setattr(deployment, "_get_pod_metrics", metrics)
    needs_fallback = failure_stage in {"manifest", "logs", "write"}
    directory = tmp_path / ("worker" if needs_fallback else "checkpoint")

    async def delete_deployment():
        assert (
            directory / f"{pod.name}.main.log"
        ).read_text() == "source startup error"
        previous_log = directory / f"{pod.name}.main.previous.log"
        if failure_stage == "previous_logs":
            assert not previous_log.exists()
        else:
            assert previous_log.read_text() == "previous instance"
        assert yaml.safe_load((directory / f"{pod.name}.yaml").read_text()) == pod.raw

    delete = AsyncMock(side_effect=delete_deployment)
    monkeypatch.setattr(deployment, "_delete_deployment", delete)
    result = await deployment.__aexit__(
        AssertionError, AssertionError("test failed"), None
    )
    assert result is False
    assert list_jobs.call_count == (2 if failure_stage == "jobs" else 1)
    assert pod_attempts == (2 if failure_stage == "pods" else 1)
    list_jobs.assert_called_with(
        namespace="default",
        label_selector="nvidia.com/dynamo-graph-deployment-name=test-deployment",
    )
    delete.assert_awaited_once()
    if needs_fallback:
        metrics.assert_called_once_with(pod, "worker", "")
    else:
        metrics.assert_not_called()
    assert pod.logs.call_count == (4 if needs_fallback else 2)


@pytest.mark.parametrize("status", [404, 403])
async def test_checkpoint_log_api_failure_does_not_prevent_cleanup(
    tmp_path, monkeypatch, status
) -> None:
    deployment = managed_deployment(tmp_path)
    list_jobs = Mock(
        side_effect=kr8s.ServerError("API error", response=httpx.Response(status))
    )
    monkeypatch.setattr("tests.deploy.dgd_utils.kr8s.get", list_jobs)
    delete = AsyncMock()
    monkeypatch.setattr(deployment, "_delete_deployment", delete)
    await deployment._cleanup()
    delete.assert_awaited_once()
    list_jobs.assert_called_once()


@pytest.mark.parametrize(
    "failure_stage, error",
    [
        ("jobs", httpx.ConnectError("tunnel dropped")),
        ("jobs", kr8s.APITimeoutError("API stalled")),
        ("pods", httpx.ConnectError("tunnel dropped")),
        ("pods", kr8s.APITimeoutError("API stalled")),
    ],
)
async def test_checkpoint_query_retry_exhaustion_preserves_cleanup(
    tmp_path, monkeypatch, failure_stage, error
) -> None:
    deployment = managed_deployment(tmp_path)
    list_jobs = Mock(
        return_value=[
            SimpleNamespace(raw={"metadata": {"name": "checkpoint", "uid": "uid"}})
        ]
    )
    monkeypatch.setattr("tests.deploy.vcluster_utils.time.sleep", Mock())
    list_pods = Mock(side_effect=error)

    def get_resources(kind, **kwargs):
        return list_jobs() if kind is SnapshotJob else list_pods()

    monkeypatch.setattr("tests.deploy.dgd_utils.kr8s.get", get_resources)
    if failure_stage == "jobs":
        list_jobs.side_effect = error
    collect = Mock()
    monkeypatch.setattr(deployment, "get_pod_manifest_logs_metrics", collect)
    service_logs = Mock()
    monkeypatch.setattr(deployment, "_get_service_logs", service_logs)
    delete = AsyncMock()
    monkeypatch.setattr(deployment, "_delete_deployment", delete)
    port_forward = Mock()
    deployment._active_port_forwards.append(port_forward)

    await deployment._cleanup()

    attempts = VCLUSTER_CONNECTION_RETRY_LIMIT + 1
    assert list_jobs.call_count == (attempts if failure_stage == "jobs" else 1)
    assert list_pods.call_count == (attempts if failure_stage == "pods" else 0)
    collect.assert_not_called()
    service_logs.assert_called_once_with(exclude_pods=set())
    port_forward.stop.assert_called_once()
    delete.assert_awaited_once()


def test_service_logs_excludes_only_collected_checkpoint_pods(tmp_path, monkeypatch):
    deployment = managed_deployment(tmp_path)
    source = SimpleNamespace(name="source")
    worker = SimpleNamespace(name="worker")
    monkeypatch.setattr(
        deployment, "get_pods", Mock(return_value={"worker": [source, worker]})
    )
    collect = Mock()
    monkeypatch.setattr(deployment, "get_pod_manifest_logs_metrics", collect)

    deployment._get_service_logs(exclude_pods={source.name})

    collect.assert_called_once_with("worker", worker, "")
