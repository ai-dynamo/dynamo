# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vllm network-partition fault test. NetworkPolicy + conntrack flush
# sever Frontend<->VllmDecodeWorker for 20s, then the partition lifts
# automatically. The worker keeps running but is unreachable from the
# frontend during the window. Companion to test_network_partition_mocker.py.

import pytest

from tests.fault_tolerance.deploy.checks import LoadStopped, MaxErrors, MinRequests
from tests.fault_tolerance.deploy.events import (
    NetworkPartition,
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
async def test_network_partition_vllm(
    namespace, image, skip_service_restart, storage_class
):
    spec = DeploymentSpec.from_backend("vllm", "agg")
    spec["Frontend"].set_env_var("DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS", "5")
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
                    # Force a fresh TCP connection per request so the
                    # NetworkPolicy applied mid-load actually rejects
                    # new flows instead of being bypassed by an existing
                    # conntrack-tracked socket.
                    connection_reuse_strategy="never",
                )
            ),
            Wait(duration=15),
            NetworkPartition(
                source="Frontend",
                target="VllmDecodeWorker",
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
        test_name="test_network_partition_vllm",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
