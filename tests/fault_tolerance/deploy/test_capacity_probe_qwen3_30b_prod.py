# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Capacity probe against the prod-matching FP8 4P:2D:2F deployment
# (mirrors the 2026-05-03 disagg cascade deployment as closely as a 12-GPU
# H100-80GB cluster allows; see disagg_qwen3_30b_4p2d_2f_prod.yaml
# header for delta vs prod).
#
# With FP8 weights (~15 GB / GPU at TP=2 + expert-parallel) inside a
# ~36 GB / GPU vLLM budget, KV headroom is ~20 GB / GPU vs ~10 GB / GPU
# in the BF16-40GB emulation — so the KV-saturation cliff lands at
# ~2× the concurrency of the BF16 variant. Rungs span that range.

import pytest

from tests.fault_tolerance.deploy.checks import LoadCompleted
from tests.fault_tolerance.deploy.events import (
    StartLoad,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import (
    FaultToleranceReport,
    PerWorkerLatencyReport,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig

_RUNGS = [32, 64, 96, 128, 192, 256]
_RUNG_DURATION_S = 180
_ISL_MEAN = 7000
_ISL_STDDEV = 600
_OSL_MEAN = 100
_OSL_STDDEV = 40


def _load_config(model: str, concurrency: int) -> LoadConfig:
    return LoadConfig(
        model_name=model,
        tokenizer=model,
        input_tokens_mean=_ISL_MEAN,
        input_tokens_stddev=_ISL_STDDEV,
        output_tokens_mean=_OSL_MEAN,
        output_tokens_stddev=_OSL_STDDEV,
        concurrency=concurrency,
        duration_minutes=_RUNG_DURATION_S / 60.0,
        request_timeout_seconds=300,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_capacity_probe_qwen3_30b_prod(runtime_env):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_4p2d_2f_prod.yaml"
    )
    served_model = spec["VllmDecodeWorker"].model

    events = [WaitForModelReady(timeout=1800)]
    for concurrency in _RUNGS:
        events += [
            StartLoad(
                load_config=_load_config(served_model, concurrency),
                name=f"c{concurrency}",
            ),
            WaitForLoadCompletion(name=f"c{concurrency}"),
        ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=[LoadCompleted(name=f"c{_RUNGS[0]}")],
        reports=[FaultToleranceReport(), PerWorkerLatencyReport()],
        test_name="test_capacity_probe_qwen3_30b_prod",
        runtime_env=runtime_env,
    )
