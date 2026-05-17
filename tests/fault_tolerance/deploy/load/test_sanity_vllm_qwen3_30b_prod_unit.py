# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sanity for the prod-mirror FP8 unit deployment. One "unit" is
# 1 Frontend + 2 Prefill (TP=2) + 1 Decode (TP=2) = 6 GPUs.
#
# Scale with --units N (declared in ``load/conftest.py``, scoped to this
# directory so the flag is invisible to tests outside ``load/``):
#   pytest --units 1 ...   # default — 6 GPUs
#   pytest --units 2 ...   # 12 GPUs (2 FE / 4 PF / 2 DE)
#   pytest --units 3 ...   # 18 GPUs
# Same FP8 args / env vars across all unit counts.

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


def _scale_to_units(spec: DeploymentSpec, units: int) -> None:
    """Multiply each service's replica count by `units`.

    Template baseline = 1 unit (1 FE / 2 PF / 1 DE). At N units this
    yields N FE / 2N PF / N DE, preserving the prod 2:1 PF:DE ratio.
    """
    assert units >= 1, f"--units must be >= 1, got {units}"
    for service_name in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[service_name].replicas = spec[service_name].replicas * units


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_sanity_vllm_qwen3_30b_prod_unit(runtime_env, units, request):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_unit_prod.yaml"
    )
    _scale_to_units(spec, units)
    served_model = spec["VllmDecodeWorker"].model
    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=1800),
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
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
