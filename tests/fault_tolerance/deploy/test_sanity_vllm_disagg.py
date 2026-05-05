# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vllm disagg sanity test (Qwen3-0.6B, 1 prefill + 1 decode). Exercises
# the prefill -> decode KV-transfer path so vllm:nixl_* counters are
# registered on the prefill worker; aiperf's --server-metrics auto-
# scrape (driven by StartLoad enumerating worker pod IPs at execution
# time) collects them alongside the frontend metrics. Useful as a
# baseline that proves the disagg topology is reachable + the worker-
# side prometheus passthrough is wired end-to-end.
#
# REQUIREMENTS: needs >= 2 GPUs on the node (1 for prefill + 1 for
# decode). On a single-GPU dev cluster the prefill pod will sit
# Pending with FailedScheduling; run on a multi-GPU node or skip.

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
async def test_sanity_vllm_disagg(
    namespace, image, skip_service_restart, storage_class
):
    spec = DeploymentSpec.from_backend("vllm", "disagg")
    served_model = spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=600),
            StartLoad(
                load_config=LoadConfig(
                    model_name=served_model,
                    tokenizer=served_model,
                    request_count=20,
                    concurrency=2,
                    input_tokens_mean=128,
                    output_tokens_mean=32,
                )
            ),
            WaitForLoadCompletion(),
        ],
        checks=[
            LoadCompleted(),
            MinRequests(min_count=20),
            ZeroErrors(),
        ],
        reports=[FaultToleranceReport()],
        namespace=namespace,
        image=image,
        test_name="test_sanity_vllm_disagg",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
