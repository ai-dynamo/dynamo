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
# Uses tests/fault_tolerance/deploy/templates/vllm/disagg_same_gpu.yaml
# (k8s mirror of examples/backends/vllm/launch/disagg_same_gpu.sh):
# both workers fit on one GPU via tiny --gpu-memory-utilization,
# capped --kv-cache-memory-bytes, and --enforce-eager. Requires the
# cluster to have either >=2 GPUs OR nvidia-device-plugin
# time-slicing enabled (1 physical GPU exposed as N logical via
# sharing.timeSlicing.resources) so both pods can claim
# nvidia.com/gpu: 1.

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
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/disagg_same_gpu.yaml"
    )
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
