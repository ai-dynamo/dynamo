#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-GPU end-to-end coverage for ``InstrumentedScheduler`` self-benchmark
mode (``DYN_BENCHMARK_MODE``).

These tests catch regressions that pure-Python unit tests can't see -- in
particular two recently-fixed cases that only manifest on a live worker
running the model on a GPU:

1. ``kv_connector_metadata`` must be attached to every benchmark-built
   ``SchedulerOutput``. Otherwise vLLM's worker-side
   ``_get_kv_connector_output`` asserts and EngineCore dies before the
   first decode batch in any disagg config (any worker with a KV
   connector configured -- NixlConnector, FlexKVConnectorV1, etc.).

2. The synthetic decode prompt must be padded to ``ctx_len + 1``.
   Otherwise the async-scheduler ``-1`` placeholder write at
   ``token_ids_cpu[req_idx, ctx_len]`` collides with the read slot the
   next benchmark batch's request reads as its decode input -- the
   embedding lookup OOBs and EngineCore dies somewhere on the second
   decode point (first batch_size > 1 sweep).

The test launches a real ``python -m dynamo.vllm`` worker with
``--benchmark-mode {agg,decode}`` and waits for the benchmark to write
its JSON result. We then validate the file structure (right mode, right
number of points, every point has at least one FPM with positive
wall_time). The worker is terminated before it can register with the
runtime so we don't depend on frontend handshake.

Coverage matrix:
  * agg benchmark, no connector -- the simplest path; catches both bugs
    if either re-regresses (the prompt-padding bug fires on agg too).
  * decode benchmark, disagg + NixlConnector kv_both -- the original
    user-reported configuration; catches the connector-metadata bug.

Both runs use Qwen3-0.6B with tight benchmark granularity (``2``) and
``--max-model-len 1024`` so each test takes ~30s of GPU time, well
within a single-GPU pre-merge slot.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import pytest

from tests.utils.constants import QWEN
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.pre_merge,
    pytest.mark.model(QWEN),
]


# Tight benchmark sweeps: 2 points per axis covers
# (batch=1, batch=max) x (ctx=block_size, ctx=max) -- enough to exercise
# both batch=1 and batch>1 code paths and both ctx=block_size_multiple
# and ctx=non-multiple cases. Keeps the wall time short.
_BENCH_GRANULARITY_FAST = "2"
_BENCH_WARMUP_ITERATIONS = "2"
_MAX_MODEL_LEN = "1024"
_GPU_MEMORY_UTIL = "0.4"


def _wait_for_benchmark_json(output_path: Path, expected_mode: str):
    """Health-check function for ManagedProcess: returns True once the
    benchmark JSON file is on disk and parses cleanly."""

    def _check(_remaining_timeout: float = 0.0) -> bool:
        if not output_path.exists():
            return False
        # File can appear before write completes; tolerate JSONDecodeError.
        try:
            data = json.loads(output_path.read_text())
        except (json.JSONDecodeError, ValueError):
            return False
        if data.get("config", {}).get("mode") != expected_mode:
            logger.warning(
                "Benchmark output mode mismatch: got %s expected %s",
                data.get("config", {}).get("mode"),
                expected_mode,
            )
            return False
        if not data.get("results"):
            return False
        return True

    return _check


def _validate_benchmark_results(output_path: Path, expected_mode: str) -> dict:
    """Parse the benchmark JSON and assert that every point reported a
    valid FPM with positive wall_time. Returns the parsed dict for
    further per-test checks."""
    assert output_path.exists(), f"benchmark JSON missing at {output_path}"
    data = json.loads(output_path.read_text())

    assert data["config"]["mode"] == expected_mode, (
        f"benchmark mode in JSON ({data['config']['mode']}) "
        f"does not match expected ({expected_mode})"
    )
    results = data["results"]
    assert len(results) > 0, "benchmark JSON has no result points"

    for r in results:
        point = r["point"]
        fpms = r["fpms"]
        assert len(fpms) > 0, (
            f"point {point} produced no FPMs -- the model didn't actually "
            f"execute that batch (regression in _bench_inject_fake_decode "
            f"or the empty-frame schedule branch)"
        )
        for fpm in fpms:
            wall_time = fpm.get("wall_time", 0.0)
            assert wall_time > 0, (
                f"point {point} FPM has non-positive wall_time={wall_time}; "
                f"the benchmark recorded a heartbeat instead of a real "
                f"forward-pass measurement"
            )
    return data


def _bench_command(
    *,
    mode: str,
    output_path: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the ``python -m dynamo.vllm`` argv for a benchmark run."""
    cmd = [
        "python3",
        "-m",
        "dynamo.vllm",
        "--model",
        QWEN,
        "--enforce-eager",
        "--max-model-len",
        _MAX_MODEL_LEN,
        "--gpu-memory-utilization",
        _GPU_MEMORY_UTIL,
        "--benchmark-mode",
        mode,
        "--benchmark-prefill-granularity",
        _BENCH_GRANULARITY_FAST,
        "--benchmark-decode-length-granularity",
        _BENCH_GRANULARITY_FAST,
        "--benchmark-decode-batch-granularity",
        _BENCH_GRANULARITY_FAST,
        "--benchmark-warmup-iterations",
        _BENCH_WARMUP_ITERATIONS,
        "--benchmark-output-path",
        output_path,
        # Cap the worker's benchmark wait at ~3min; the result file is
        # what we actually wait on, this is just a safety net for the
        # worker side.
        "--benchmark-timeout",
        "180",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _run_benchmark(
    request: pytest.FixtureRequest,
    *,
    mode: str,
    extra_args: list[str] | None,
    tmp_path: Path,
    log_label: str,
) -> Path:
    """Spawn a vllm worker in benchmark mode, wait for the result JSON,
    then terminate. Returns the path to the produced JSON."""
    output_path = tmp_path / f"bench_{log_label}.json"

    log_dir = f"{request.node.name}_{log_label}"
    try:
        shutil.rmtree(log_dir)
    except FileNotFoundError:
        pass

    env = os.environ.copy()
    # Verbose logging makes triage tractable when CI catches a regression.
    env.setdefault("DYN_LOG", "info")

    proc = ManagedProcess(
        command=_bench_command(
            mode=mode, output_path=str(output_path), extra_args=extra_args
        ),
        env=env,
        # Wait on the JSON file rather than a frontend port: the benchmark
        # finishes before the worker registers, and we tear down the
        # worker before its serving loop matters.
        health_check_funcs=[_wait_for_benchmark_json(output_path, mode)],
        timeout=420,  # cold start (~30s) + warmup (~5s) + sweep (~30s) + buffer
        display_output=True,
        terminate_all_matching_process_names=False,
        stragglers=["VLLM::EngineCore"],
        straggler_commands=["-m dynamo.vllm"],
        log_dir=log_dir,
    )

    with proc:
        # ManagedProcess __enter__ has already polled
        # _wait_for_benchmark_json until it returned True -- the JSON
        # file is on disk and parses. Do the structural validation
        # while the process is still alive so any error log is captured
        # in the test's captured output.
        _validate_benchmark_results(output_path, mode)

    return output_path


@pytest.mark.timeout(600)
def test_self_benchmark_agg_mode(request, runtime_services, tmp_path):
    """Aggregated worker (no kv-transfer-config) running
    ``--benchmark-mode agg``.

    Catches the prompt-padding bug: prefill sweep runs through normal
    scheduling (unaffected), but the decode sweep within agg mode
    hits ``_bench_inject_fake_decode`` and would CUDA-OOB at
    batch>1 if the prompt isn't padded.
    """
    output = _run_benchmark(
        request,
        mode="agg",
        extra_args=None,
        tmp_path=tmp_path,
        log_label="agg",
    )
    data = json.loads(output.read_text())

    # Sanity: agg sweep produces both prefill and decode points.
    point_types = {r["point"]["point_type"] for r in data["results"]}
    assert (
        "prefill" in point_types
    ), f"agg benchmark missing prefill points; got types={point_types}"
    assert (
        "decode" in point_types
    ), f"agg benchmark missing decode points; got types={point_types}"

    # Sanity: the decode sweep covers batch_size > 1 (the case that
    # regressed). With granularity=2 we expect at least one decode point
    # at batch_size > 1.
    decode_batches = sorted(
        r["point"]["batch_size"]
        for r in data["results"]
        if r["point"]["point_type"] == "decode"
    )
    assert any(b > 1 for b in decode_batches), (
        f"agg decode sweep only ran batch=1; the prompt-padding regression "
        f"would not be caught by this run. batch_sizes={decode_batches}"
    )


@pytest.mark.timeout(600)
def test_self_benchmark_disagg_decode_with_nixl_connector(
    request, runtime_services, tmp_path
):
    """Disagg decode worker with NixlConnector ``kv_both`` running
    ``--benchmark-mode decode``.

    This is the configuration users hit in production -- it catches
    BOTH:
      * connector-metadata regression (fails on the very first
        synthetic decode batch),
      * prompt-padding regression (fails on the second decode point,
        first batch>1).
    """
    output = _run_benchmark(
        request,
        mode="decode",
        extra_args=[
            "--disaggregation-mode",
            "decode",
            "--kv-transfer-config",
            '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
        ],
        tmp_path=tmp_path,
        log_label="disagg_decode_nixl",
    )
    data = json.loads(output.read_text())

    # All result points must be decode-mode.
    for r in data["results"]:
        assert (
            r["point"]["point_type"] == "decode"
        ), f"--benchmark-mode decode produced a non-decode point: {r['point']}"

    # Sanity: at least one batch>1 point ran. If it didn't, the
    # connector-metadata fix passed but the prompt-padding fix isn't
    # exercised -- a future regression in the latter would be silent.
    decode_batches = sorted(r["point"]["batch_size"] for r in data["results"])
    assert any(b > 1 for b in decode_batches), (
        f"disagg decode sweep only ran batch=1; the prompt-padding "
        f"regression would not be caught. batch_sizes={decode_batches}"
    )
