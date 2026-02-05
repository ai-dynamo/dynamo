# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def get_deployment_spec(
    backend: str, deployment_type: str, worker_replicas: int = 1
) -> DeploymentSpec:
    """Get DeploymentSpec for a backend and deployment type.

    Args:
        backend: "vllm", "trtllm", or "sglang"
        deployment_type: "agg" or "disagg"
        worker_replicas: Number of worker replicas (1 or 2)

    Returns:
        DeploymentSpec configured for the backend/type combination
    """
    yaml_path = f"/workspace/examples/backends/{backend}/deploy/{deployment_type}.yaml"
    spec = DeploymentSpec(yaml_path)

    # Set worker replicas based on deployment type
    if deployment_type == "agg":
        # Aggregated: single worker service
        worker_service = {
            "trtllm": "TRTLLMWorker",
            "vllm": "VllmWorker",
            "sglang": "SglangWorker",
        }[backend]
        spec.set_service_replicas(worker_service, worker_replicas)
    else:
        # Disaggregated: prefill and decode workers
        prefill_service = {
            "trtllm": "TRTLLMPrefillWorker",
            "vllm": "VllmPrefillWorker",
            "sglang": "SglangPrefillWorker",
        }[backend]
        decode_service = {
            "trtllm": "TRTLLMDecodeWorker",
            "vllm": "VllmDecodeWorker",
            "sglang": "SglangDecodeWorker",
        }[backend]
        spec.set_service_replicas(prefill_service, worker_replicas)
        spec.set_service_replicas(decode_service, worker_replicas)

    return spec


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_worker_pod_kill_recovery(request):
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
        request=request,
        deployment_spec=DeploymentSpec(
            "/workspace/examples/backends/trtllm/deploy/disagg.yaml"
        ),
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
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_smoke_50_requests(request):
    """
    Basic smoke test: 50 requests with no failures.
    """
    await run_scenario(
        request=request,
        deployment_spec=DeploymentSpec(
            "/workspace/examples/backends/trtllm/deploy/agg.yaml"
        ),
        events=[
            StartLoad(load_config=LoadConfig(request_count=50, concurrency=4)),
            WaitForLoadCompletion(),
        ],
        checks=[
            WasCancelled(expected=False),
            MinRequests(min_count=50),
            ZeroErrors(),
        ],
    )


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_rolling_upgrade_zero_errors(request):
    """
    Test rolling upgrade with zero errors.

    Scenario:
    1. Start continuous load
    2. Rolling upgrade decode workers
    3. Rolling upgrade prefill workers
    4. Stop load
    5. Assert zero errors
    """
    deployment_spec = DeploymentSpec(
        "/workspace/examples/backends/trtllm/deploy/disagg.yaml"
    )
    deployment_spec.set_service_replicas("TRTLLMPrefillWorker", 2)
    deployment_spec.set_service_replicas("TRTLLMDecodeWorker", 2)

    await run_scenario(
        request=request,
        deployment_spec=deployment_spec,
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
    request, backend, deployment_type, worker_service, process_name, replicas
):
    """
    Test engine process termination with recovery.

    Terminates the main engine process (not the pod) and verifies recovery.
    """
    await run_scenario(
        request=request,
        deployment_spec=get_deployment_spec(backend, deployment_type, replicas),
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
async def test_frontend_process_termination(request, backend, deployment_type, replicas):
    """
    Test frontend process termination with recovery.

    Terminates the frontend process and verifies recovery.
    """
    await run_scenario(
        request=request,
        deployment_spec=get_deployment_spec(backend, deployment_type, replicas),
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
    )
