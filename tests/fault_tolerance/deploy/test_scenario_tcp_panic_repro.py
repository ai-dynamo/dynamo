# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# E2E DGD repro for the PR #8254 TCP stream-read panic. v1.0.1 of
# vllm-runtime contains the bug; under TCP RST on an established socket
# the `handle_reader` arm at lib/runtime/src/pipeline/network/tcp/client.rs
# panics on Some(Err(_)).
#
# Steady c=24 closed-loop, 3 min warm-up, then iptables FORWARD REJECT
# --reject-with tcp-reset on decode pod-0 for 60 s. We expect to see
# `panicked at .*tcp/(client|server).rs` lines in Frontend + worker
# logs and possibly a kubelet liveness-driven restart on the affected
# pods. Run against a fixed image (commit eec0d5432+) to validate
# absence.

import pytest

from tests.fault_tolerance.deploy.checks import (
    LoadCompleted,
    RestartCountIncreased,
    ServiceLogPatternRate,
    WorkerPanics,
)
from tests.fault_tolerance.deploy.events import (
    RstInjection,
    StartLoad,
    Wait,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import (
    ErrorBreakdownReport,
    ErrorTrackingReport,
    FaultToleranceReport,
    PerWorkerLatencyReport,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


_TEMPLATE = (
    "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
    "disagg_qwen3_30b_unit_prod.yaml"
)


def _scale_to_units(spec, units):
    for service in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[service].replicas = spec[service].replicas * units


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_tcp_panic_repro_e2e(runtime_env, request):
    spec = DeploymentSpec(_TEMPLATE)
    _scale_to_units(spec, units=3)
    served_model = spec["VllmDecodeWorker"].model

    cfg = LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=7000, input_tokens_stddev=0,
        output_tokens_mean=100, output_tokens_stddev=0,
        concurrency=24,
        duration_minutes=10.0,  # 3 min warmup + 60 s fault + ~5.5 min observation
        request_timeout_seconds=120,
        streaming=True, ignore_eos=True, warmup_requests=0,
        connection_reuse_strategy="never",
    )

    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=1800),
            StartLoad(load_config=cfg, name="load"),
            Wait(duration=180),  # warm-up so streams settle
            RstInjection(
                service="VllmDecodeWorker",
                pod_indices=[0],
                duration=60.0,
                name="rst_inject_decode_0",
            ),
            WaitForLoadCompletion(name="load"),
        ],
        checks=[
            LoadCompleted(name="load"),
            # WorkerPanics: floor=1.0/sec is conservative (prod was ~2.2)
            # On image with PR #8254 fix this should drop to zero (run separately).
            WorkerPanics(
                services=["Frontend", "VllmPrefillWorker", "VllmDecodeWorker"],
                acceptable=True,  # report-only; explicit floor next:
            ),
            ServiceLogPatternRate(
                services=["Frontend", "VllmPrefillWorker", "VllmDecodeWorker"],
                pattern=r"panicked at .*tcp/(client|server)\.rs",
                min_rate_per_sec=0.01,  # 1 panic / 100s = passes
                # NOTE: when running against PR #8254 fix image, flip to
                # expect_zero=True via a parametrize over image. Today we
                # only run vs the bug-image so floor>0 confirms the bug.
            ),
            # Did the panic flood actually drive a kubelet probe restart?
            # Informational — recorded as report-only.
            RestartCountIncreased(
                services=["Frontend", "VllmDecodeWorker"],
                expect_min_increment=1,
                expect_zero=False,  # negative result is informative
            ),
        ],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
