# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Mocker-backed engine-stall fault test. SIGSTOP the mocker process so
# the worker stops servicing requests but the pod, container, and TCP
# connections all stay alive. After 20s the process is SIGCONT'd and
# normal service resumes. This exercises the "hung worker" path
# (frontend sees idle sockets, no crash, no reconnect) rather than the
# "crash + reload" path of TerminateProcess. Companion to
# test_engine_stall_vllm.py.

import pytest

from tests.fault_tolerance.deploy.checks import LoadStopped, MaxErrors, MinRequests
from tests.fault_tolerance.deploy.events import StallProcess, StartLoad, StopLoad, Wait
from tests.fault_tolerance.deploy.reports import (
    ErrorBreakdownReport,
    FaultToleranceReport,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_engine_stall_mocker(
    namespace, image, skip_service_restart, storage_class
):
    """Pause the mocker engine process for 20s; expect inflight requests
    to back up at the frontend, then resume cleanly when SIGCONT lands."""
    spec = DeploymentSpec.from_backend("mocker", "agg")
    spec["decode"].set_arg(
        "--planner-profile-data",
        "/workspace/tests/planner/profiling_results/H200_TP1P_TP1D",
    )
    # Tighten aiperf's request timeout so in-flight requests fail fast
    # while the worker is paused, instead of hanging the full scenario.
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
                    request_timeout_seconds=10,
                )
            ),
            Wait(duration=15),
            StallProcess(
                services=["decode"],
                process_name="dynamo.mocker",
                duration=20,
            ),
            Wait(duration=10),  # observe recovery after SIGCONT
            StopLoad(),
        ],
        checks=[
            LoadStopped(),
            MinRequests(min_count=20),
            # Cap permissive: the point is the FT report, not an SLA on errors.
            MaxErrors(max_errors=1_000_000),
        ],
        reports=[FaultToleranceReport(), ErrorBreakdownReport()],
        namespace=namespace,
        image=image,
        test_name="test_engine_stall_mocker",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
