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
    Wait,
    WaitForLoadCompletion,
    WaitForRecovery,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


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
