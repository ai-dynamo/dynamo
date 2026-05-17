# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Capacity probe (A100-40GB emulation) — push the 4P:2D:2F deployment until
# vllm:kv_cache_usage_perc hits 100 % and preemption begins.
#
# Same workload as test_capacity_probe_qwen3_30b.py but against the
# disagg_qwen3_30b_4p2d_2f_40gb template (gpu-memory-utilization=0.5, capping
# vllm to 40 GB / GPU). On H100-80GB this leaves ~10 GB / GPU for KV after
# Qwen3-30B-A3B bf16 weights, making 100 % KV reachable at moderate
# concurrency.
#
# Rungs extend further than the 80 GB sweep to capture preemption /
# crash-loop signatures past saturation.

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

# Start at c32 — the 80 GB baseline already showed sustained decode
# queueing + NIXL tail at every rung from c8 up. With KV per GPU ~¼ of
# the baseline at gpu-memory-utilization=0.5, c32 is the smallest rung
# expected to drive kv_cache_usage_perc to 100 % within 3 min, and c128
# is where we expect crash-loops / OOM.
_RUNGS = [32, 64, 96, 128]
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
async def test_capacity_probe_qwen3_30b_40gb(runtime_env):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_4p2d_2f_40gb.yaml"
    )
    served_model = spec["VllmDecodeWorker"].model

    events = [WaitForModelReady(timeout=1500)]
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
        test_name="test_capacity_probe_qwen3_30b_40gb",
        runtime_env=runtime_env,
    )
