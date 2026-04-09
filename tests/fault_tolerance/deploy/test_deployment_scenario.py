# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fault tolerance scenario tests.

Uses the scenario framework to define and run tests with:
- events: Actions to execute (StartLoad, Wait, DeletePod, etc.)
- checks: Validations to run after events (ZeroErrors, MinRequests, etc.)
- reports: Optional report generation after checks pass
"""

import pytest

from tests.fault_tolerance.deploy.checks import (
    MaxErrors,
    MinRequests,
    WasCancelled,
    ZeroErrors,
)
from tests.fault_tolerance.deploy.events import (
    DeletePod,
    RollingUpgrade,
    StartLoad,
    StopLoad,
    TerminateProcess,
    Wait,
    WaitForLoadCompletion,
    WaitForRecovery,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_sanity_10_requests(
    namespace, image, skip_service_restart, storage_class
):
    """
    Sanity test: deploy, send 10 requests via aiperf, verify all succeed.

    This is the simplest possible end-to-end test. It validates:
    - Deployment comes up and becomes ready
    - aiperf can reach the frontend and get responses
    - All 10 requests complete without errors
    - Results are collected from PVC
    """
    await run_scenario(
        deployment_spec=DeploymentSpec.from_backend("trtllm", "agg"),
        events=[
            StartLoad(
                load_config=LoadConfig(
                    request_count=10,
                    concurrency=2,
                    input_tokens_mean=128,
                    output_tokens_mean=32,
                )
            ),
            WaitForLoadCompletion(),
        ],
        checks=[
            WasCancelled(expected=False),
            MinRequests(min_count=10),
            ZeroErrors(),
        ],
        namespace=namespace,
        image=image,
        test_name="test_sanity_10_requests",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_worker_pod_kill_recovery(
    namespace, image, skip_service_restart, storage_class
):
    """
    Test worker pod kill with recovery.

    Scenario:
    1. Start load
    2. Kill worker pod
    3. Wait for recovery
    4. Stop load
    5. Assert low errors
    """
    await run_scenario(
        deployment_spec=DeploymentSpec.from_backend("trtllm", "disagg"),
        events=[
            StartLoad(load_config=LoadConfig(duration_minutes=5, concurrency=8)),
            Wait(duration=30),
            DeletePod(services=["TRTLLMWorker"]),
            WaitForRecovery(timeout=300),
            Wait(duration=30),
            StopLoad(),
        ],
        checks=[
            MaxErrors(max_errors=20),
            MinRequests(min_count=50),
        ],
        namespace=namespace,
        image=image,
        test_name="test_worker_pod_kill_recovery",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_smoke_50_requests(namespace, image, skip_service_restart, storage_class):
    """
    Basic smoke test: 50 requests with no failures.
    """
    await run_scenario(
        deployment_spec=DeploymentSpec.from_backend("trtllm", "agg"),
        events=[
            StartLoad(load_config=LoadConfig(request_count=50, concurrency=4)),
            WaitForLoadCompletion(),
        ],
        checks=[
            WasCancelled(expected=False),
            MinRequests(min_count=50),
            ZeroErrors(),
        ],
        namespace=namespace,
        image=image,
        test_name="test_smoke_50_requests",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_rolling_upgrade_zero_errors(
    namespace, image, skip_service_restart, storage_class
):
    """
    Test rolling upgrade with zero errors.

    Scenario:
    1. Start continuous load
    2. Rolling upgrade decode workers
    3. Rolling upgrade prefill workers
    4. Stop load
    5. Assert zero errors
    """
    spec = DeploymentSpec.from_backend("trtllm", "disagg")
    spec.set_worker_replicas(2)

    await run_scenario(
        deployment_spec=spec,
        events=[
            StartLoad(load_config=LoadConfig(duration_minutes=15, concurrency=8)),
            Wait(duration=30),
            RollingUpgrade(services=["TRTLLMDecodeWorker"]),
            Wait(duration=30),
            RollingUpgrade(services=["TRTLLMPrefillWorker"]),
            Wait(duration=30),
            StopLoad(),
        ],
        checks=[
            ZeroErrors(),
            MinRequests(min_count=100),
        ],
        namespace=namespace,
        image=image,
        test_name="test_rolling_upgrade_zero_errors",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "backend,deployment_type,worker_service,process_name,replicas",
    [
        # TensorRT-LLM configs
        ("trtllm", "agg", "TRTLLMWorker", "dynamo.runtime", 1),
        ("trtllm", "agg", "TRTLLMWorker", "dynamo.runtime", 2),
        ("trtllm", "disagg", "TRTLLMDecodeWorker", "dynamo.runtime", 1),
        ("trtllm", "disagg", "TRTLLMDecodeWorker", "dynamo.runtime", 2),
        # vLLM configs
        ("vllm", "agg", "VllmWorker", "dynamo.vllm", 1),
        ("vllm", "agg", "VllmWorker", "dynamo.vllm", 2),
        ("vllm", "disagg", "VllmDecodeWorker", "dynamo.vllm", 1),
        ("vllm", "disagg", "VllmDecodeWorker", "dynamo.vllm", 2),
    ],
)
async def test_engine_process_termination(
    namespace,
    image,
    skip_service_restart,
    storage_class,
    backend,
    deployment_type,
    worker_service,
    process_name,
    replicas,
):
    """
    Test engine process termination with recovery.

    Terminates the main engine process (not the pod) and verifies recovery.
    """
    spec = DeploymentSpec.from_backend(backend, deployment_type)
    spec.set_worker_replicas(replicas)

    await run_scenario(
        deployment_spec=spec,
        events=[
            StartLoad(load_config=LoadConfig(duration_minutes=5, concurrency=8)),
            Wait(duration=30),
            TerminateProcess(services=[worker_service], process_name=process_name),
            WaitForRecovery(timeout=300),
            Wait(duration=30),
            StopLoad(),
        ],
        checks=[
            MaxErrors(max_errors=20),
            MinRequests(min_count=50),
        ],
        namespace=namespace,
        image=image,
        test_name=f"test_engine_process_termination[{backend}-{deployment_type}-{replicas}]",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "backend,deployment_type,replicas",
    [
        # TensorRT-LLM configs
        ("trtllm", "agg", 1),
        ("trtllm", "agg", 2),
        ("trtllm", "disagg", 1),
        ("trtllm", "disagg", 2),
        # vLLM configs
        ("vllm", "agg", 1),
        ("vllm", "agg", 2),
        ("vllm", "disagg", 1),
        ("vllm", "disagg", 2),
    ],
)
async def test_frontend_process_termination(
    namespace,
    image,
    skip_service_restart,
    storage_class,
    backend,
    deployment_type,
    replicas,
):
    """
    Test frontend process termination with recovery.

    Terminates the frontend process and verifies recovery.
    """
    spec = DeploymentSpec.from_backend(backend, deployment_type)
    spec.set_worker_replicas(replicas)

    await run_scenario(
        deployment_spec=spec,
        events=[
            StartLoad(load_config=LoadConfig(duration_minutes=5, concurrency=8)),
            Wait(duration=30),
            TerminateProcess(services=["Frontend"], process_name="python"),
            WaitForRecovery(timeout=300),
            Wait(duration=30),
            StopLoad(),
        ],
        checks=[
            MaxErrors(max_errors=20),
            MinRequests(min_count=50),
        ],
        namespace=namespace,
        image=image,
        test_name=f"test_frontend_process_termination[{backend}-{deployment_type}-{replicas}]",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
