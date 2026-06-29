# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang self-benchmark adapter wiring."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamo.sglang.args import parse_args
from dynamo.sglang.benchmark import benchmark_config, wait_and_load_benchmark
from dynamo.sglang.tests.conftest import make_cli_args_fixture

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.filterwarnings("ignore:.*torch.jit.script_method.*:DeprecationWarning"),
]

mock_sglang_cli = make_cli_args_fixture("dynamo.sglang")


@pytest.mark.asyncio
async def test_benchmark_mode_enabled_from_env(monkeypatch, mock_sglang_cli):
    """Dynamo should map benchmark env vars onto native SGLang args."""
    monkeypatch.setenv("DYN_BENCHMARK_MODE", "prefill")
    monkeypatch.setenv("DYN_BENCHMARK_PREFILL_GRANULARITY", "32")
    monkeypatch.setenv("DYN_BENCHMARK_PREFILL_KV_READ_GRANULARITY", "4")
    monkeypatch.setenv("DYN_BENCHMARK_TIMEOUT", "17")
    mock_sglang_cli("--model", "Qwen/Qwen3-0.6B")

    config = await parse_args(sys.argv[1:])
    assert config.server_args.benchmark_mode == "prefill"
    assert config.server_args.enable_forward_pass_metrics is True
    assert config.server_args.benchmark_prefill_granularity == 32
    assert config.server_args.benchmark_prefill_kv_read_granularity == 4
    assert config.server_args.benchmark_timeout == 17
    assert benchmark_config(config.server_args)["prefill_kv_read_granularity"] == 4


@pytest.mark.asyncio
async def test_benchmark_cli_overrides_env(monkeypatch, mock_sglang_cli):
    """Explicit SGLang benchmark CLI flags should win over Dynamo env defaults."""
    monkeypatch.setenv("DYN_BENCHMARK_MODE", "prefill")
    monkeypatch.setenv("DYN_BENCHMARK_PREFILL_KV_READ_GRANULARITY", "4")
    monkeypatch.setenv("DYN_BENCHMARK_TIMEOUT", "17")
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--benchmark-mode",
        "decode",
        "--benchmark-prefill-kv-read-granularity",
        "9",
        "--benchmark-timeout",
        "23",
    )

    config = await parse_args(sys.argv[1:])
    assert config.server_args.benchmark_mode == "decode"
    assert config.server_args.benchmark_prefill_kv_read_granularity == 9
    assert config.server_args.benchmark_timeout == 23


# ---- result handoff consumer (wait_and_load_benchmark) ----
#
# The producer (SGLang scheduler) rewrites the result file to an invalid
# ``status: running`` sentinel at startup and atomically replaces it with the
# completed results when the sweep finishes. The consumer must wait for
# ``status == "complete"`` rather than mere file existence, or it would accept
# the in-progress sentinel (or a stale prior-run file) and mark the worker ready
# prematurely.


def _server_args(
    output_path: Path, *, timeout: int = 5, model: str = "test-model", **overrides
):
    args = SimpleNamespace(
        benchmark_mode="agg",
        benchmark_output_path=str(output_path),
        benchmark_timeout=timeout,
        model_path=model,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _complete(run_id="run-1", model="test-model", results=None):
    return {
        "schema_version": 1,
        "scope": "local_diagnostics",
        "status": "complete",
        "valid": True,
        "run_id": run_id,
        "identity": {"model_path": model},
        "results": results
        if results is not None
        else [{"point": {"point_type": "prefill", "isl": 8}, "fpms": []}],
    }


def _sentinel(run_id="run-1", model="test-model"):
    return {
        "schema_version": 1,
        "scope": "local_diagnostics",
        "status": "running",
        "valid": False,
        "run_id": run_id,
        "identity": {"model_path": model},
        "message": "Self-benchmark is running; previous results are invalid.",
    }


def _write(path: Path, payload: dict):
    path.write_text(json.dumps(payload))


@pytest.mark.asyncio
async def test_returns_error_when_benchmark_mode_disabled(tmp_path):
    server_args = _server_args(tmp_path / "results.json")
    server_args.benchmark_mode = None

    result = await wait_and_load_benchmark(server_args)

    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_waits_past_running_sentinel_until_complete(tmp_path):
    output_path = tmp_path / "results.json"
    # The file already exists as the in-progress sentinel before the sweep ends.
    _write(output_path, _sentinel(run_id="run-7"))
    server_args = _server_args(output_path, timeout=5)

    async def finish_after_delay():
        await asyncio.sleep(0.05)
        _write(output_path, _complete(run_id="run-7"))

    writer = asyncio.create_task(finish_after_delay())
    merged = await wait_and_load_benchmark(server_args)
    await writer

    assert merged["status"] == "complete"
    assert merged["run_id"] == "run-7"
    assert len(merged["results"]) == 1


@pytest.mark.asyncio
async def test_running_sentinel_alone_times_out(tmp_path):
    # A benchmark that never completes (e.g. a multi-rank finish-sync deadlock)
    # leaves the file stuck at status=running; existence must NOT satisfy us.
    output_path = tmp_path / "results.json"
    _write(output_path, _sentinel())
    server_args = _server_args(output_path, timeout=0)

    with pytest.raises(TimeoutError):
        await wait_and_load_benchmark(server_args)


@pytest.mark.asyncio
async def test_missing_file_times_out(tmp_path):
    server_args = _server_args(tmp_path / "results.json", timeout=0)

    with pytest.raises(TimeoutError):
        await wait_and_load_benchmark(server_args)


@pytest.mark.asyncio
async def test_warns_on_model_mismatch_but_still_loads(tmp_path, caplog):
    output_path = tmp_path / "results.json"
    _write(output_path, _complete(model="some-other-model"))
    server_args = _server_args(output_path, model="test-model")

    with caplog.at_level(logging.WARNING, logger="dynamo.sglang.benchmark"):
        merged = await wait_and_load_benchmark(server_args)

    assert merged["status"] == "complete"
    assert "some-other-model" in caplog.text


@pytest.mark.asyncio
async def test_backward_compat_file_without_status_field(tmp_path):
    # Producers predating the status field write only results; still accept them.
    output_path = tmp_path / "results.json"
    _write(output_path, {"results": [{"point": {"point_type": "decode"}, "fpms": []}]})
    server_args = _server_args(output_path)

    merged = await wait_and_load_benchmark(server_args)

    assert len(merged["results"]) == 1


@pytest.mark.asyncio
async def test_merges_results_across_dp_ranks(tmp_path):
    base = tmp_path / "results.json"
    dp1 = tmp_path / "results_dp1.json"
    _write(
        base,
        _complete(run_id="same", results=[{"point": {"point_type": "prefill"}, "fpms": []}]),
    )
    _write(
        dp1,
        _complete(run_id="same", results=[{"point": {"point_type": "decode"}, "fpms": []}]),
    )
    server_args = _server_args(
        base, dp_size=2, enable_dp_attention=True, nnodes=1, node_rank=0
    )

    merged = await wait_and_load_benchmark(server_args)

    assert len(merged["results"]) == 2
    dp_ranks = {r["point"]["dp_rank"] for r in merged["results"]}
    assert dp_ranks == {0, 1}


@pytest.mark.asyncio
async def test_warns_on_run_id_mismatch_across_dp_ranks(tmp_path, caplog):
    # A leftover dp-rank file from a previous run (different run_id) must surface
    # as a warning rather than being silently merged in as valid data.
    base = tmp_path / "results.json"
    dp1 = tmp_path / "results_dp1.json"
    _write(base, _complete(run_id="run-A"))
    _write(dp1, _complete(run_id="run-B"))
    server_args = _server_args(
        base, dp_size=2, enable_dp_attention=True, nnodes=1, node_rank=0
    )

    with caplog.at_level(logging.WARNING, logger="dynamo.sglang.benchmark"):
        merged = await wait_and_load_benchmark(server_args)

    assert len(merged["results"]) == 2
    assert "mismatched run_ids" in caplog.text
