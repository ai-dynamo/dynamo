# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoGraphDeploymentRequest lifecycle helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kubernetes_asyncio import config

import tests.deploy.test_dgdr as dgdr_tests
from tests.deploy.dgdr_utils import (
    DGDRCleanupError,
    DGDRTestConfig,
    ManagedDGDR,
    parse_final_dgd,
    run_lifecycle,
    unique_name,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_all_dgdr_tests_have_ci_suite_prefix() -> None:
    prefixes = (
        "test_dgdr_validation_",
        "test_dgdr_profiling_",
        "test_dgdr_lifecycle_",
    )
    unassigned = sorted(
        name
        for name, value in vars(dgdr_tests).items()
        if name.startswith("test_dgdr_")
        and callable(value)
        and not name.startswith(prefixes)
    )
    assert not unassigned, f"DGDR tests missing a CI suite prefix: {unassigned}"


@pytest.mark.parametrize("name_prefix", ["", "ci-" + "x" * 60])
def test_unique_name_fits_profiler_and_grove_limits(name_prefix: str) -> None:
    config_ = DGDRTestConfig(
        namespace="dgdr-lifecycle-30007208578-1",
        image="test",
        name_prefix=name_prefix,
    )
    name = unique_name(config_, "lifecycle-ready")

    assert len(f"{name}-dgd") + len("TRTLLMPrefillWorker") <= 45
    assert len(f"{config_.namespace}-{name}-dgd") <= 63


async def test_init_propagates_unexpected_incluster_errors(monkeypatch) -> None:
    monkeypatch.delenv("KUBECONFIG", raising=False)
    error = RuntimeError("broken in-cluster configuration")
    monkeypatch.setattr(
        config,
        "load_incluster_config",
        MagicMock(side_effect=error),
    )

    with pytest.raises(RuntimeError, match="broken in-cluster configuration"):
        await ManagedDGDR(
            DGDRTestConfig(namespace="test-namespace", image="test")
        ).init()


async def test_init_falls_back_when_incluster_config_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.delenv("KUBECONFIG", raising=False)
    monkeypatch.setattr(
        config,
        "load_incluster_config",
        MagicMock(side_effect=config.ConfigException("not running in a cluster")),
    )
    load_kube_config = AsyncMock()
    monkeypatch.setattr(config, "load_kube_config", load_kube_config)
    manager = ManagedDGDR(DGDRTestConfig(namespace="test-namespace", image="test"))

    try:
        await manager.init()
        load_kube_config.assert_awaited_once_with()
    finally:
        await manager.close()


def initialized_manager() -> ManagedDGDR:
    manager = ManagedDGDR(DGDRTestConfig(namespace="test-namespace", image="test"))
    manager.custom = MagicMock()
    manager.core = MagicMock()
    manager.batch = MagicMock()
    manager.apiextensions = MagicMock()
    return manager


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "must contain at least one YAML document"),
        ("kind: ConfigMap", "must be a DynamoGraphDeployment"),
    ],
)
def test_parse_final_dgd_rejects_invalid_external_data(
    content: str, message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        parse_final_dgd(content)


async def test_lifecycle_preserves_combined_deployment_timeout() -> None:
    manager = ManagedDGDR(
        DGDRTestConfig(
            namespace="test-namespace",
            image="test",
            profiling_timeout=17,
            deploy_timeout=11,
        )
    )
    manager.create = AsyncMock()
    manager.wait_for_phase_at_least = AsyncMock(
        return_value={"status": {"profilingJobName": "profiling-job"}}
    )
    manager.wait_for_phase = AsyncMock(
        return_value={"status": {"dgdName": "deployment"}}
    )

    await run_lifecycle(
        manager,
        {"metadata": {"name": "request"}, "spec": {}},
        verify_configmap=False,
    )

    manager.wait_for_phase.assert_awaited_once_with("request", "Deployed", 28)


async def test_cleanup_reports_all_failures_and_retains_failed_names() -> None:
    manager = initialized_manager()
    manager._created_names = ["first", "second", "third"]
    calls = []

    async def cleanup_name(name: str, failed: bool) -> None:
        calls.append((name, failed))
        if name != "second":
            raise RuntimeError(f"could not delete {name}")

    manager._cleanup_name = AsyncMock(side_effect=cleanup_name)

    with pytest.raises(DGDRCleanupError) as error:
        await manager.cleanup(failed=True)

    assert calls == [("third", True), ("second", True), ("first", True)]
    assert [name for name, _ in error.value.failures] == ["third", "first"]
    assert manager._created_names == ["first", "third"]


async def test_profiling_failure_diagnostics_include_job_and_pod_logs() -> None:
    manager = initialized_manager()
    assert manager.batch is not None
    assert manager.core is not None

    job = MagicMock()
    job.to_str.return_value = "profiling job"
    manager.batch.read_namespaced_job = AsyncMock(return_value=job)

    pod = MagicMock()
    pod.metadata.name = "profiling-pod"
    pod.spec.init_containers = []
    pod.spec.containers = [SimpleNamespace(name="profiler")]
    pod.to_str.return_value = "profiling pod"
    manager.core.list_namespaced_pod = AsyncMock(
        return_value=SimpleNamespace(items=[pod])
    )
    manager.core.read_namespaced_pod_log = AsyncMock(return_value="profiler logs")

    await manager._log_diagnostics({"status": {"profilingJobName": "profiling-job"}})

    manager.batch.read_namespaced_job.assert_awaited_once_with(
        "profiling-job", "test-namespace"
    )
    manager.core.list_namespaced_pod.assert_awaited_once_with(
        "test-namespace", label_selector="job-name=profiling-job"
    )
    manager.core.read_namespaced_pod_log.assert_awaited_once_with(
        "profiling-pod",
        "test-namespace",
        container="profiler",
        tail_lines=300,
    )
