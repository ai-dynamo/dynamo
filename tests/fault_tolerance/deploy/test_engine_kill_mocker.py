# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Mocker-backed engine-kill fault test. SIGKILL the mocker process so
# the kubelet restarts the same pod's container in place; the PVC tee
# should produce a distinct log file per incarnation. Companion to
# test_engine_kill_vllm.py.

import pytest

from tests.fault_tolerance.deploy.checks import LoadStopped, MaxErrors, MinRequests
from tests.fault_tolerance.deploy.events import (
    StartLoad,
    StopLoad,
    TerminateProcess,
    Wait,
    WaitForRecovery,
)
from tests.fault_tolerance.deploy.reports import FaultToleranceReport
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_engine_kill_mocker(
    namespace, image, skip_service_restart, storage_class
):
    """Kill the mocker engine process; pod stays, container restarts in place."""
    spec = DeploymentSpec.from_backend("mocker", "agg")
    spec["decode"].set_arg(
        "--planner-profile-data",
        "/workspace/tests/planner/profiling_results/H200_TP1P_TP1D",
    )
    spec["decode"].set_arg("--startup-time", "10")
    served_model = spec["decode"].model

    await run_scenario(
        deployment_spec=spec,
        events=[
            StartLoad(
                load_config=LoadConfig(
                    model_name=served_model,
                    tokenizer=served_model,
                    duration_minutes=2,
                    concurrency=4,
                    input_tokens_mean=128,
                    output_tokens_mean=32,
                )
            ),
            Wait(duration=15),
            TerminateProcess(
                services=["decode"],
                process_name="dynamo.mocker",
                signal="SIGKILL",
            ),
            WaitForRecovery(timeout=120),
            Wait(duration=20),
            StopLoad(),
        ],
        checks=[
            LoadStopped(),
            MinRequests(min_count=20),
            MaxErrors(max_errors=20),
        ],
        reports=[FaultToleranceReport()],
        namespace=namespace,
        image=image,
        test_name="test_engine_kill_mocker",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
