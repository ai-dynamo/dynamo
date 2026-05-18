# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vllm engine-kill fault test. SIGKILL the dynamo.vllm process inside
# the worker pod so kubelet restarts the container in place; pod IP is
# preserved but vllm reload still dominates recovery (~90s). Companion
# to test_engine_kill_mocker.py.

import pytest

from tests.fault_tolerance.deploy.checks import MaxErrors, MinRequests
from tests.fault_tolerance.deploy.events import (
    StartLoad,
    StopLoad,
    TerminateProcess,
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
async def test_engine_kill_vllm(namespace, image, skip_service_restart, storage_class):
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
            TerminateProcess(
                services=["VllmDecodeWorker"],
                process_name="dynamo.vllm",
                signal="SIGKILL",
            ),
            WaitForRecovery(timeout=300),
            Wait(duration=20),
            StopLoad(),
        ],
        checks=[
            MinRequests(min_count=20),
            MaxErrors(max_errors=1_000_000),
        ],
        reports=[FaultToleranceReport(), ErrorBreakdownReport()],
        namespace=namespace,
        image=image,
        test_name="test_engine_kill_vllm",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
