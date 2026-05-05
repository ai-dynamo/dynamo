# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vllm pod-delete fault test. Mid-load DeletePod on the
# VllmDecodeWorker; the framework's WaitForRecovery + FaultToleranceReport
# captures recovery time (vllm reload dominates, ~80-90s) and the pre/
# post-fault request and latency delta on a real worker. Companion to
# test_worker_kill_mocker.py.

import pytest

from tests.fault_tolerance.deploy.checks import MaxErrors, MinRequests
from tests.fault_tolerance.deploy.events import (
    DeletePod,
    StartLoad,
    StopLoad,
    Wait,
    WaitForModelReady,
    WaitForRecovery,
)
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
async def test_worker_kill_vllm(namespace, image, skip_service_restart, storage_class):
    spec = DeploymentSpec.from_backend("vllm", "agg")
    served_model = spec["VllmDecodeWorker"].model

    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=600),
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
            Wait(duration=20),
            DeletePod(services=["VllmDecodeWorker"]),
            WaitForRecovery(timeout=300),
            Wait(duration=20),
            StopLoad(),
        ],
        checks=[
            # No LoadStopped check — vllm startup + recovery can exceed
            # the load duration, in which case the load completes
            # naturally rather than being stopped early.
            MinRequests(min_count=20),
            # vllm worker recovery can take >30s; aiperf retries the dead
            # endpoint at high rate so absolute error count is huge but
            # each is a 404. The point of this test is the report, not
            # the threshold. Keep the cap permissive so the test can
            # render its FT report.
            MaxErrors(max_errors=1_000_000),
        ],
        reports=[FaultToleranceReport(), ErrorBreakdownReport()],
        namespace=namespace,
        image=image,
        test_name="test_worker_kill_vllm",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
