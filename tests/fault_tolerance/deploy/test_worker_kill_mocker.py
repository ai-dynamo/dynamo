# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Mocker-backed pod-delete fault test. Mid-load DeletePod on the decode
# worker; the framework's WaitForRecovery + FaultToleranceReport
# captures recovery time and the pre/post-fault request and latency
# delta. Companion to test_worker_kill_vllm.py.

import pytest

from tests.fault_tolerance.deploy.checks import LoadStopped, MaxErrors, MinRequests
from tests.fault_tolerance.deploy.events import (
    DeletePod,
    StartLoad,
    StopLoad,
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
async def test_worker_kill_mocker(
    namespace, image, skip_service_restart, storage_class
):
    """Inject a single decode-pod kill mid-load and produce the report."""
    spec = DeploymentSpec.from_backend("mocker", "agg")
    spec["decode"].set_arg(
        "--planner-profile-data",
        "/workspace/tests/planner/profiling_results/H200_TP1P_TP1D",
    )
    # Make the mocker simulate a slow worker boot so the post-fault
    # window is long enough for in-flight requests to actually fail
    # (rather than just stalling under the frontend's buffering).
    spec["decode"].set_arg("--startup-time", "30")
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
            DeletePod(services=["decode"]),
            WaitForRecovery(timeout=120),
            Wait(duration=20),
            StopLoad(),
        ],
        checks=[
            # StopLoad explicitly terminates the load mid-run, so cancelled
            # is the expected outcome here (the alternative is to let the
            # load run to its natural duration end which would slow the
            # test for no extra signal).
            LoadStopped(),
            MinRequests(min_count=20),
            MaxErrors(max_errors=20),
        ],
        reports=[FaultToleranceReport()],
        namespace=namespace,
        image=image,
        test_name="test_worker_kill_mocker",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
