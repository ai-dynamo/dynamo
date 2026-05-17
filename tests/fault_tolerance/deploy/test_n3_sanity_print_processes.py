# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sanity test: stand up the N=3 prod-mirror DGD and dump the full process
# tree of every prefill and decode pod. Used to verify which PID is the
# launcher and which are the per-rank (TP) workers before designing
# rank-targeted faults.

import pytest

from tests.fault_tolerance.deploy.backend_processes import VLLM
from tests.fault_tolerance.deploy.checks import RankProcessCount
from tests.fault_tolerance.deploy.events import (
    PrintProcessTree,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec


def _scale_to_units(spec: DeploymentSpec, units: int) -> None:
    assert units >= 1
    for service_name in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[service_name].replicas = spec[service_name].replicas * units


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_n3_sanity_print_processes(runtime_env, request):
    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_unit_prod.yaml"
    )
    _scale_to_units(spec, units=3)

    events = [
        WaitForModelReady(timeout=1800),
        PrintProcessTree(services=["VllmPrefillWorker", "VllmDecodeWorker"]),
    ]

    # Both VllmPrefillWorker and VllmDecodeWorker run TP=2 per the template.
    # The vLLM process tree per pod is one EngineCore (the coordinator) and
    # TP=2 worker subprocesses (VLLM::Worker_TP0_EP0 / TP1_EP1). We assert
    # both counts so a missing rank or missing engine is caught loudly,
    # before any fault scenario tries to target the wrong PID.
    services = ["VllmPrefillWorker", "VllmDecodeWorker"]
    checks = [
        RankProcessCount(
            services=services, process_name=VLLM.engine_core, expected=1
        ),
        RankProcessCount(
            services=services, process_name=VLLM.worker, expected=2
        ),
    ]

    await run_scenario(
        deployment_spec=spec,
        events=events,
        checks=checks,
        reports=[],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
