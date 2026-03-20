import argparse
import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "components" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "bindings" / "python" / "src"))

_fake_kv_cache = types.ModuleType("dynamo.mocker.utils.kv_cache")
_fake_kv_cache.DEFAULT_KV_TRANSFER_BANDWIDTH_GBPS = 64.0
_fake_kv_cache.compute_kv_bytes_per_token = lambda *args, **kwargs: 0
sys.modules.setdefault("dynamo.mocker.utils.kv_cache", _fake_kv_cache)

_fake_profiler_converter = types.ModuleType(
    "dynamo.mocker.utils.planner_profiler_perf_data_converter"
)
_fake_profiler_converter.convert_profile_results_to_npz = lambda *args, **kwargs: None
_fake_profiler_converter.is_mocker_format_npz = lambda *args, **kwargs: False
_fake_profiler_converter.is_profile_results_dir = lambda *args, **kwargs: False
sys.modules.setdefault(
    "dynamo.mocker.utils.planner_profiler_perf_data_converter",
    _fake_profiler_converter,
)

from dynamo.mocker import args as mocker_args


def test_parse_args_rejects_non_aggregated_replay(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dynamo.mocker",
            "--trace-file",
            "trace.jsonl",
            "--disaggregation-mode",
            "decode",
        ],
    )

    with pytest.raises(
        ValueError, match="--trace-file only supports aggregated replay"
    ):
        mocker_args.parse_args()


def test_worker_replay_writes_json_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    mocker_main = importlib.import_module("dynamo.mocker.main")
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text('{"timestamp":1.0,"input_length":4,"output_length":2,"hash_ids":[1]}\n')
    extra_engine_args = tmp_path / "engine_args.json"
    extra_engine_args.write_text("{}")
    output_file = tmp_path / "report.json"

    args = argparse.Namespace(
        planner_profile_data=None,
        extra_engine_args=extra_engine_args,
        num_workers=1,
        model_path=None,
        kv_bytes_per_token=None,
        kv_cache_dtype="auto",
        trace_file=trace_file,
        output_file=output_file,
    )
    monkeypatch.setattr(mocker_main, "parse_args", lambda: args)
    monkeypatch.setattr(
        mocker_main,
        "run_mocker_trace_replay",
        lambda trace_file, extra_engine_args=None: {
            "num_requests": 1,
            "completed_requests": 1,
            "total_input_tokens": 4,
            "total_output_tokens": 2,
            "duration_ms": 12.5,
            "wall_time_ms": 1.5,
            "request_throughput_rps": 80.0,
            "input_throughput_tok_s": 320.0,
            "output_throughput_tok_s": 160.0,
            "total_throughput_tok_s": 480.0,
            "prefix_cache_reused_ratio": 0.25,
            "mean_queue_ms": 0.5,
            "mean_ttft_ms": 4.0,
            "median_ttft_ms": 4.0,
            "p95_ttft_ms": 4.0,
            "p99_ttft_ms": 4.0,
            "mean_tpot_ms": 2.0,
            "median_tpot_ms": 2.0,
            "p95_tpot_ms": 2.0,
            "p99_tpot_ms": 2.0,
            "mean_itl_ms": 2.0,
            "median_itl_ms": 2.0,
            "p95_itl_ms": 2.0,
            "p99_itl_ms": 2.0,
            "max_itl_ms": 2.0,
            "mean_e2e_latency_ms": 6.0,
            "median_e2e_latency_ms": 6.0,
            "p95_e2e_latency_ms": 6.0,
            "p99_e2e_latency_ms": 6.0,
        },
    )

    asyncio.run(mocker_main.worker())

    report = json.loads(output_file.read_text())
    stdout = capsys.readouterr().out

    assert report["completed_requests"] == 1
    assert "Replay Summary" in stdout
    assert "Completed requests: 1/1" in stdout
    assert str(output_file) in stdout
