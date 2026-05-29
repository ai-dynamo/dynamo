# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Saturation discovery test — finds C_max for an EXISTING deployment under
# a latency SLA. Does NOT deploy/teardown; runs aiperf against the FE
# service in the chosen namespace.
#
# Sweeps a concurrency ladder, measures p95 latency on each rung, and reports
# the highest concurrency where the SLA still held. Roll-our-own substitute for
# aiperf's `max_concurrency_under_sla` recipe (not in the pinned aiperf version).
#
# Example invocation (must be run AFTER a deployment is up in <namespace>):
#
#   uv run pytest test_saturation_discovery.py \
#       --namespace neelays-e1-arm5 \
#       --sat-served-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
#       --sat-concurrency-ladder 200,400,800,1200,1600,2000,2400 \
#       --sat-rung-seconds 90 \
#       --sat-sla-ms 5000 \
#       --sat-sla-quantile p95 \
#       --storage-class dgxc-enterprise-file --log-pvc dynamo-ft-logs \
#       --skip-service-restart -s -v
#
# For mocker deployments running at --speedup-ratio 5.0, an SLA of 5000 ms
# (=30000 / 5x speedup) is the canonical "production-equivalent" target.
# For real-vLLM workloads the SLA is the production p95 latency budget
# (e.g., 30000 ms).

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.utils.managed_load import LoadConfig, ManagedLoad

_LOG = logging.getLogger(__name__)


def _normalize_quantile(q: str) -> str:
    """Map p50/p95/p99/avg/min/max → the key in aiperf summary JSON."""
    if q in ("avg", "p50", "p95", "p99", "min", "max"):
        return q
    raise ValueError(f"unknown SLA quantile: {q}")


def _extract_quantile(summary: dict, metric: str, quantile: str) -> float | None:
    """Read summary[metric][quantile] from the aiperf JSON shape."""
    m = summary.get(metric, {})
    return m.get(quantile)


def add_cli_options(parser):
    """Pytest CLI options scoped to saturation discovery."""
    g = parser.getgroup("saturation_discovery")
    g.addoption(
        "--sat-served-model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        help="Served model name passed to aiperf --model + --tokenizer.",
    )
    g.addoption(
        "--sat-concurrency-ladder",
        default="200,400,800,1200,1600,2000,2400",
        help="Comma-separated concurrency levels to sweep (ascending).",
    )
    g.addoption(
        "--sat-rung-seconds",
        type=int,
        default=90,
        help="Seconds of steady load per rung. Default 90.",
    )
    g.addoption(
        "--sat-warmup-seconds",
        type=int,
        default=15,
        help="Warmup seconds at start of each rung (aiperf request_count "
        "stabilization). Counts against rung-seconds.",
    )
    g.addoption(
        "--sat-sla-metric",
        default="request_latency",
        choices=["request_latency", "time_to_first_token", "inter_token_latency"],
        help="Which aiperf metric to gate on.",
    )
    g.addoption(
        "--sat-sla-quantile",
        default="p95",
        choices=["avg", "p50", "p95", "p99"],
        help="Which quantile of the SLA metric to gate on.",
    )
    g.addoption(
        "--sat-sla-ms",
        type=float,
        default=5000.0,
        help="SLA threshold in milliseconds. Below = pass. Above = saturated. "
        "Default 5000 ms — appropriate for mocker --speedup-ratio=5.0 "
        "(matches prod 30s budget / 5x).",
    )
    g.addoption(
        "--sat-min-success-rate",
        type=float,
        default=0.95,
        help="Minimum aiperf request success rate per rung (1 - errors/total). "
        "Below this counts as saturated.",
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_saturation_discovery(runtime_env, request, namespace, image):
    """Sweep concurrency, find the highest rung where p95 latency stays under SLA.

    Does not deploy. Assumes a deployment with a Frontend Service is already
    running in the chosen namespace. Each rung is an independent aiperf job
    submitted via ManagedLoad; results are collected per-rung and the test
    reports the saturation boundary.
    """
    cfg = request.config
    model = cfg.getoption("--sat-served-model")
    ladder = [
        int(c)
        for c in cfg.getoption("--sat-concurrency-ladder").split(",")
        if c.strip()
    ]
    rung_seconds = cfg.getoption("--sat-rung-seconds")
    warmup_seconds = cfg.getoption("--sat-warmup-seconds")
    sla_metric = cfg.getoption("--sat-sla-metric")
    sla_quantile = _normalize_quantile(cfg.getoption("--sat-sla-quantile"))
    sla_ms = cfg.getoption("--sat-sla-ms")
    min_success = cfg.getoption("--sat-min-success-rate")

    log_dir = Path(f"/workspace/test_outputs/test_saturation_discovery_{namespace}")
    log_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info(
        "Saturation sweep: namespace=%s model=%s ladder=%s sla=%s/%s/%dms rung=%ds",
        namespace,
        model,
        ladder,
        sla_metric,
        sla_quantile,
        sla_ms,
        rung_seconds,
    )

    results: list[dict] = []
    c_max = 0
    saturated_at: int | None = None

    for level in ladder:
        _LOG.info(f"=== rung concurrency={level} ===")

        load_config = LoadConfig(
            model_name=model,
            tokenizer=model,
            concurrency=level,
            benchmark_duration_seconds=rung_seconds,
            warmup_duration_seconds=warmup_seconds,
            # Match the production seq_dist shape, same as E1 arms.
            seq_dist=(
                "100,20:1;100,80:2;100,130:2;100,180:1;100,200:1;"
                "500,20:2;500,80:5;500,130:5;500,180:2;500,200:1;"
                "1000,20:3;1000,80:5;1000,130:5;1000,180:3;1000,200:2;"
                "1600,20:5;1600,80:8;1600,130:8;1600,180:5;1600,200:3;"
                "3400,20:3;3400,80:6;3400,130:6;3400,180:3;3400,200:2;"
                "7000,20:2;7000,80:3;7000,130:3;7000,180:2;7000,200:1"
            ),
            num_prefix_prompts=5,
            prefix_prompt_length=740,
            streaming=True,
            ignore_eos=True,
            request_timeout_seconds=30,
            connection_reuse_strategy="never",
        )

        rung_name = f"sat-c{level}"
        ml = ManagedLoad(
            log_dir=str(log_dir),
            load_config=load_config,
            namespace=namespace,
            name=rung_name,
            target_service_name="Frontend",
        )
        async with ml:
            await ml.run(wait_for_completion=True)

        # aiperf summary lives at <log_dir>/load/<job-name>/profile_export_aiperf.json
        # ManagedLoad's run() returns a dict; we also can read the file directly.
        try:
            summary_path = next(
                (log_dir / "load").glob(f"{rung_name}-*/profile_export_aiperf.json")
            )
            with open(summary_path) as fh:
                summary = json.load(fh)
        except (StopIteration, FileNotFoundError) as e:
            _LOG.error(f"rung c={level}: aiperf summary missing ({e})")
            saturated_at = level
            results.append({"concurrency": level, "error": "no aiperf summary"})
            break

        latency = _extract_quantile(summary, sla_metric, sla_quantile)
        total_count = (summary.get("request_count", {}) or {}).get("avg")
        err_count = (summary.get("error_request_count", {}) or {}).get("avg") or 0
        success_rate = (total_count - err_count) / total_count if total_count else 0.0

        passed_sla = latency is not None and latency <= sla_ms
        passed_success = success_rate >= min_success
        passed = passed_sla and passed_success

        _LOG.info(
            f"  c={level}: {sla_metric}/{sla_quantile}={latency:.0f}ms "
            f"(sla={sla_ms:.0f}) success_rate={success_rate:.3f} "
            f"(min={min_success}) → {'PASS' if passed else 'SATURATED'}"
        )

        results.append(
            {
                "concurrency": level,
                f"{sla_metric}_{sla_quantile}_ms": latency,
                "success_rate": success_rate,
                "total_requests": total_count,
                "error_requests": err_count,
                "passed_sla": passed_sla,
                "passed_success": passed_success,
                "passed": passed,
            }
        )

        if passed:
            c_max = level
        else:
            saturated_at = level
            _LOG.info(
                f"=== Saturation boundary found: C_max={c_max}, first failure at c={level} ==="
            )
            break

    # Write a summary CSV that's easy to plot / share
    summary_path = log_dir / "saturation_sweep_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "namespace": namespace,
                "image": image,
                "served_model": model,
                "sla_metric": sla_metric,
                "sla_quantile": sla_quantile,
                "sla_ms": sla_ms,
                "min_success_rate": min_success,
                "rung_seconds": rung_seconds,
                "ladder": ladder,
                "c_max_under_sla": c_max,
                "saturated_at": saturated_at,
                "rungs": results,
            },
            fh,
            indent=2,
        )
    _LOG.info(f"Summary written: {summary_path}")
    _LOG.info(f"FINAL: C_max under SLA={c_max}, saturation observed at={saturated_at}")
