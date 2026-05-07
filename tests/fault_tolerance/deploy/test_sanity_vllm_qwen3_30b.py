# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sanity test for the 1F + 1P (TP=2) + 2D (TP=2) vllm disagg topology
# on Qwen3-30B-A3B. Verifies: PVC binding, all 4 pods reach Ready,
# the prefill→decode NIXL KV-transfer path comes up clean, the model
# registers on the frontend, and a small request burst returns with
# zero errors. Baseline for the disagg failure-mode repros that
# follow (cudagraph cliff, KV-cleanup retention, rank wedge).
#
# REQUIREMENTS:
#   - >= 6 GPUs reachable (1 prefill TP=2 + 2 decode TP=2 = 6).
#     Fits one 8xH100-80GB node; on smaller hosts use the
#     templates/vllm/disagg_same_gpu.yaml + Qwen3-0.6B path instead.
#   - dynamo v1.0.1 vllm-runtime image
#     (nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1).
#   - hf-token-secret in the test namespace (Qwen3-30B-A3B is gated
#     on HF; the secret is loaded via envFromSecret on workers).

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
async def test_sanity_vllm_qwen3_30b(
    namespace, image, skip_service_restart, storage_class, log_pvc, model_pvc
):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_1p2d.yaml"
    )
    served_model = spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=spec,
        events=[
            # Model load on Qwen3-30B is the long pole here — pull,
            # init, KV-cache warmup. Generous timeout.
            WaitForModelReady(timeout=1200),
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
        test_name="test_sanity_vllm_qwen3_30b",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
        log_pvc=log_pvc,
        model_pvc=model_pvc,
    )
