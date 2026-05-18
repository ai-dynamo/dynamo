# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vllm + Qwen3-0.6B sanity test. Runs the framework against a real
# vllm worker (instead of the mocker) so the baseline next to the
# vllm fault tests reflects realistic worker behaviour. 10 requests
# at concurrency=2; expects zero errors and load-completed.

import pytest

from tests.fault_tolerance.deploy.checks import LoadCompleted, MinRequests, ZeroErrors
from tests.fault_tolerance.deploy.events import (
    StartLoad,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import FaultToleranceReport
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_sanity_vllm(namespace, image, skip_service_restart, storage_class):
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
                    request_count=10,
                    concurrency=2,
                    input_tokens_mean=128,
                    output_tokens_mean=32,
                )
            ),
            WaitForLoadCompletion(),
        ],
        checks=[
            LoadCompleted(),
            MinRequests(min_count=10),
            ZeroErrors(),
        ],
        reports=[FaultToleranceReport()],
        namespace=namespace,
        image=image,
        test_name="test_sanity_vllm",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
