# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vllm engine-stall fault test. SIGSTOP the dynamo.vllm process for
# 20s — the worker stops responding but the pod, container, and TCP
# request-plane sockets all stay alive. SIGCONT resumes normal
# service. Tests how the frontend handles a "hung" worker (idle
# sockets, no crash, no reconnect) — distinct from the crash + reload
# path of test_engine_kill_vllm. Companion to test_engine_stall_mocker.py.

import pytest

from tests.fault_tolerance.deploy.checks import LoadStopped, MaxErrors, MinRequests
from tests.fault_tolerance.deploy.events import (
    StallProcess,
    StartLoad,
    StopLoad,
    Wait,
    WaitForModelReady,
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
async def test_engine_stall_vllm(namespace, image, skip_service_restart, storage_class):
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
                    request_timeout_seconds=10,
                )
            ),
            Wait(duration=15),
            StallProcess(
                services=["VllmDecodeWorker"],
                process_name="dynamo.vllm",
                duration=20,
            ),
            Wait(duration=10),
            StopLoad(),
        ],
        checks=[
            LoadStopped(),
            MinRequests(min_count=20),
            MaxErrors(max_errors=1_000_000),
        ],
        reports=[FaultToleranceReport(), ErrorBreakdownReport()],
        namespace=namespace,
        image=image,
        test_name="test_engine_stall_vllm",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
