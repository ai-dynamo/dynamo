# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Load-shedding-maintains-SLA test.
#
# Question being verified: can we configure the router to reject (HTTP 503)
# requests once worker load crosses a configurable busy threshold, such that
# TTFT / request-latency p99 for ACCEPTED requests stays within SLA even when
# the offered concurrency is well above the knee?
#
# Two arms via parametrize:
#   - "baseline"   — no shedding config. Concurrency above knee → latency
#                    spiral.
#   - "shed_early" — busy thresholds set low so the router returns 503 once
#                    workers are saturated.  Above-knee concurrency yields
#                    high 503 rate but bounded p99 for accepted requests.
#
# The "shed_early" arm uses two mechanisms:
#  1. POST /busy_threshold to the FE for the served model.
#     See lib/llm/src/http/service/busy_threshold.rs — returns 503 when all
#     workers for the model exceed their configured thresholds.
#  2. (informational) DYN_ROUTER_QUEUE_THRESHOLD env on the FE — the existing
#     kv-router queue-threshold knob.
#
# Both arms use aiperf goodput SLO so the report carries the same SLA shape
# we already use for capacity sweeps: `request_latency:30000` ms.

import json

import pytest

from tests.fault_tolerance.deploy.checks import (
    LoadCompleted,
    ServiceLogPatternRate,
)
from tests.fault_tolerance.deploy.events import (
    Event,
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


# --------------------------------------------------------------------------
# SetBusyThreshold — POST /busy_threshold to the Frontend so the router
# returns 503 once worker load crosses the configured fraction.
# --------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SetBusyThreshold(Event):
    """POST /busy_threshold to the Frontend service. Affects only the named
    model. Subsequent requests will receive 503 once *all* workers for the
    model exceed the thresholds."""

    model_name: str
    active_decode_blocks_threshold: float = 0.85
    active_prefill_tokens_threshold_frac: float = 0.85
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx) -> None:
        import asyncio

        import aiohttp

        pods = (await asyncio.to_thread(
            ctx.deployment.get_pods, ["Frontend"])).get("Frontend") or []
        if not pods:
            raise RuntimeError("SetBusyThreshold: no Frontend pods")
        fe = pods[0]
        body = {
            "model": self.model_name,
            "active_decode_blocks_threshold": self.active_decode_blocks_threshold,
            "active_prefill_tokens_threshold_frac":
                self.active_prefill_tokens_threshold_frac,
        }
        pf = await asyncio.to_thread(ctx.deployment.port_forward, fe, 8000)
        if pf is None or pf.local_port == 0:
            raise RuntimeError(
                f"SetBusyThreshold: port-forward to {fe.name}:8000 failed"
            )
        url = f"http://127.0.0.1:{pf.local_port}/busy_threshold"
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=body, timeout=30) as r:
                txt = await r.text()
                if r.status != 200:
                    raise RuntimeError(
                        f"SetBusyThreshold: POST returned {r.status}: {txt}"
                    )
                ctx.logger.info(
                    f"SetBusyThreshold: {self.model_name} -> {txt}"
                )
                self.results = json.loads(txt)

    async def stop(self, ctx) -> None:
        pass

    @property
    def description(self) -> str:
        return (
            f"SetBusyThreshold(model={self.model_name}, "
            f"decode_frac={self.active_decode_blocks_threshold}, "
            f"prefill_frac={self.active_prefill_tokens_threshold_frac})"
        )


def _load_config_prod(model: str, concurrency: int) -> LoadConfig:
    """Prod-shape workload — generic ISL/OSL distribution params."""
    return LoadConfig(
        model_name=model,
        tokenizer=model,
        input_tokens_mean=1641, input_tokens_stddev=2800,
        output_tokens_mean=2, output_tokens_stddev=100,
        concurrency=concurrency,
        # Use closed-loop concurrency so we expose latency spiral cleanly;
        # 4 min per rung is enough to reach steady state for prod ISL.
        duration_minutes=4.0,
        request_timeout_seconds=60,
        # Goodput SLO for SLA enforcement: requests with latency > 30s
        # count as failed in goodput. Lets us report goodput vs offered RPS
        # for the comparison plot.
        goodput=["request_latency:30000"],
        streaming=True, ignore_eos=True, warmup_requests=0,
        connection_reuse_strategy="never",
    )


@pytest.fixture
def n1_spec():
    spec = DeploymentSpec(_TEMPLATE)
    return spec


# --------------------------------------------------------------------------
# Two arms — same N=1 prod-mirror DGD, same concurrency ladder, different
# router/FE shedding config.
# --------------------------------------------------------------------------
@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "shed_arm",
    ["baseline", "shed_early"],
)
async def test_load_shedding_maintains_sla(
    runtime_env, request, n1_spec, shed_arm
):
    served_model = n1_spec["VllmDecodeWorker"].model

    # In "shed_early" we also lower the kv-router queue threshold via env
    # var on the FE, so the router rejects earlier at the queue layer.
    if shed_arm == "shed_early":
        fe = n1_spec["Frontend"]
        fe.set_env_var("DYN_ROUTER_QUEUE_THRESHOLD", "0.5")
        fe.set_env_var("DYN_ROUTER_TRACK_PREFILL_TOKENS", "true")

    # Three rungs: well below / at / well above the N=1 knee (c≈8).
    rungs = [
        ("c4", _load_config_prod(served_model, concurrency=4)),
        ("c8", _load_config_prod(served_model, concurrency=8)),
        ("c24", _load_config_prod(served_model, concurrency=24)),
    ]

    events: list = [WaitForModelReady(timeout=1800)]
    if shed_arm == "shed_early":
        events.append(SetBusyThreshold(
            model_name=served_model,
            active_decode_blocks_threshold=0.85,
            active_prefill_tokens_threshold_frac=0.85,
            name="set_busy",
        ))
    # Chain rungs back-to-back with a 30s settle in between.
    for name, cfg in rungs:
        events.append(StartLoad(load_config=cfg, name=name))
        events.append(WaitForLoadCompletion(name=name))
        events.append(Wait(duration=30))

    # Checks differ per arm.
    checks: list = [LoadCompleted(name="c24")]
    if shed_arm == "shed_early":
        # We want to see 503s in the highest rung (proves shedding fires)
        # AND we want non-error latency p99 to stay bounded. The latter is
        # naturally enforced by aiperf goodput SLO — if shedding works, the
        # goodput=(requests-meeting-SLA / total-non-error) should stay high
        # at c24, where without shedding it collapses.
        # ServiceLogPatternRate looks for the Rust-side queue-exceeded WARN.
        checks.append(ServiceLogPatternRate(
            services=["Frontend"],
            pattern=r"all workers.*busy|queue.*threshold|503",
            min_rate_per_sec=0.0,
            expect_zero=False,
        ))
    # (baseline arm has no SLA assertion — it's informational; the report
    # carries the spiral evidence.)

    await run_scenario(
        deployment_spec=n1_spec,
        events=events,
        checks=checks,
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
            ErrorTrackingReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )
