# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

import dynamo.mocker as mocker_module
import dynamo.planner.offline.replay_adapter as replay_adapter_module
import dynamo.replay.main as replay_main

from .replay_utils import (
    _assert_basic_report_counts,
    _assert_basic_report_metrics,
    _assert_replay_cli_outputs,
    _planner_profile_data_dir_path,
    _run_replay_cli,
    _write_cli_smoke_trace,
    _write_multiturn_trace,
    _write_planner_profile_data_npz,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_replay_cli_aic_perf_config_includes_moe_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_aic_perf_config(**kwargs):
        captured_kwargs.update(kwargs)
        return kwargs

    monkeypatch.setattr(replay_main, "AicPerfConfig", fake_aic_perf_config)

    config = replay_main._load_aic_perf_config(
        SimpleNamespace(
            aic_backend="vllm",
            aic_system="h200_sxm",
            aic_model_path="moonshotai/Kimi-K2-Instruct",
            aic_backend_version=None,
            aic_tp_size=2,
            aic_moe_tp_size=2,
            aic_moe_ep_size=1,
            aic_attention_dp_size=1,
            aic_nextn=None,
            aic_nextn_accept_rates=None,
            aic_gemm_dtype=None,
            aic_moe_dtype=None,
            aic_fmha_dtype=None,
            aic_kv_cache_dtype=None,
            aic_comm_dtype=None,
        )
    )

    assert config == captured_kwargs
    assert captured_kwargs == {
        "aic_backend": "vllm",
        "aic_system": "h200_sxm",
        "aic_model_path": "moonshotai/Kimi-K2-Instruct",
        "aic_tp_size": 2,
        "aic_backend_version": None,
        "aic_moe_tp_size": 2,
        "aic_moe_ep_size": 1,
        "aic_attention_dp_size": 1,
        "aic_nextn": None,
        "aic_nextn_accept_rates": None,
        "aic_gemm_dtype": None,
        "aic_moe_dtype": None,
        "aic_fmha_dtype": None,
        "aic_kv_cache_dtype": None,
        "aic_comm_dtype": None,
    }


def test_replay_policy_config_flag_overrides_router_json(monkeypatch):
    captured = []

    class FakeKvRouterConfig:
        @staticmethod
        def from_json(value):
            captured.append(value)
            return value

    monkeypatch.setattr(replay_main, "KvRouterConfig", FakeKvRouterConfig)

    config = replay_main._load_router_config(
        '{"router_queue_policy":"wspt","router_policy_config":"embedded.yaml"}',
        "explicit.yaml",
    )

    assert config == captured[0]
    assert json.loads(captured[0]) == {
        "router_queue_policy": "wspt",
        "router_policy_config": "explicit.yaml",
    }


def test_planner_replay_passes_configured_dynamo_warmup_observations(
    monkeypatch, tmp_path
):
    warmup_path = tmp_path / "warmup.jsonl"
    records = []
    for index, received_ms in enumerate((1_000, 11_000)):
        records.append(
            {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": received_ms + 10,
                "request": {
                    "request_id": f"request-{index}",
                    "request_received_ms": received_ms,
                    "output_tokens": 4 + index,
                    "replay": {
                        "trace_block_size": 64,
                        "input_length": 64 * (index + 1),
                        "input_sequence_hashes": [101 + index],
                    },
                },
            }
        )
    warmup_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"timestamp":0,"input_length":64,"output_length":2,"hash_ids":[101]}\n',
        encoding="utf-8",
    )

    captured = {}

    class FakeBridge:
        @staticmethod
        def from_trace_files(**kwargs):
            captured["trace_files"] = kwargs["trace_files"]
            return FakeBridge()

        def run(self, _adapter):
            return {}

    class FakeAdapter:
        def __init__(self, *, warmup_observations, **_kwargs):
            captured["warmup_observations"] = warmup_observations

        def _is_easy_mode(self):
            return True

        def finalize(self, trace_report):
            return SimpleNamespace(trace_report=trace_report)

        def close(self):
            pass

    monkeypatch.setattr(mocker_module, "PlannerReplayBridge", FakeBridge)
    monkeypatch.setattr(replay_adapter_module, "ReplayPlannerAdapter", FakeAdapter)

    replay_main._run_planner_replay(
        trace_files=[trace_path],
        trace_format="mooncake",
        extra_engine_args=replay_main.MockEngineArgs(
            block_size=64, speedup_ratio=1000.0
        ),
        prefill_engine_args=None,
        decode_engine_args=None,
        router_config=None,
        num_workers=1,
        num_prefill_workers=1,
        num_decode_workers=1,
        router_mode="round_robin",
        arrival_speedup_ratio=1.0,
        trace_block_size=64,
        planner_config_arg=json.dumps(
            {
                "mode": "agg",
                "optimization_target": "throughput",
                "load_predictor_warmup_trace": str(warmup_path),
                "throughput_adjustment_interval_seconds": 10,
                "report_interval_hours": None,
                "live_dashboard_port": 0,
            }
        ),
    )

    observations = captured["warmup_observations"]
    assert captured["trace_files"] == [trace_path]
    assert [(item.num_req, item.isl, item.osl) for item in observations] == [
        (1.0, 64.0, 4.0),
        (1.0, 128.0, 5.0),
    ]


@pytest.mark.timeout(30)
def test_replay_cli_subprocess_synthetic_smoke(tmp_path):
    report_path = tmp_path / "synthetic_report.json"

    completed = _run_replay_cli(
        tmp_path,
        "--input-tokens",
        "250",
        "--output-tokens",
        "25",
        "--request-count",
        "10",
        "--num-workers",
        "4",
        "--replay-concurrency",
        "4",
        "--report-json",
        str(report_path),
        "--extra-engine-args",
        '{"block_size":64,"speedup_ratio":1000.0}',
    )

    report = _assert_replay_cli_outputs(completed, report_path)
    _assert_basic_report_counts(
        report,
        num_requests=10,
        input_tokens=250,
        output_tokens=25,
    )
    _assert_basic_report_metrics(report)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("planner_profile_data_kind", ["dir", "npz"])
def test_replay_cli_subprocess_synthetic_smoke_accepts_planner_profile_data(
    tmp_path, planner_profile_data_kind
):
    report_path = tmp_path / f"synthetic_report_{planner_profile_data_kind}.json"
    planner_profile_data = (
        _planner_profile_data_dir_path()
        if planner_profile_data_kind == "dir"
        else _write_planner_profile_data_npz(tmp_path)
    )

    completed = _run_replay_cli(
        tmp_path,
        "--input-tokens",
        "250",
        "--output-tokens",
        "25",
        "--request-count",
        "10",
        "--num-workers",
        "4",
        "--replay-concurrency",
        "4",
        "--report-json",
        str(report_path),
        "--extra-engine-args",
        json.dumps(
            {
                "block_size": 64,
                "speedup_ratio": 1000.0,
                "planner_profile_data": str(planner_profile_data),
            }
        ),
    )

    report = _assert_replay_cli_outputs(completed, report_path)
    _assert_basic_report_counts(
        report,
        num_requests=10,
        input_tokens=250,
        output_tokens=25,
    )
    _assert_basic_report_metrics(report)


@pytest.mark.timeout(30)
def test_replay_cli_subprocess_synthetic_multiturn_smoke(tmp_path):
    report_path = tmp_path / "synthetic_multiturn_report.json"

    completed = _run_replay_cli(
        tmp_path,
        "--input-tokens",
        "64",
        "--output-tokens",
        "4",
        "--request-count",
        "3",
        "--request-rate",
        "10",
        "--arrival-seed",
        "17",
        "--turns-per-session",
        "2",
        "--shared-prefix-ratio",
        "0.5",
        "--num-prefix-groups",
        "2",
        "--inter-turn-delay-ms",
        "5.0",
        "--num-workers",
        "2",
        "--report-json",
        str(report_path),
        "--extra-engine-args",
        '{"block_size":64,"speedup_ratio":1000.0}',
    )

    report = _assert_replay_cli_outputs(completed, report_path)
    _assert_basic_report_counts(
        report,
        num_requests=6,
        input_tokens=64,
        output_tokens=4,
    )
    _assert_basic_report_metrics(report)


@pytest.mark.timeout(30)
def test_replay_cli_subprocess_trace_smoke(tmp_path):
    trace_path = _write_cli_smoke_trace(tmp_path)
    report_path = tmp_path / "trace_report.json"

    completed = _run_replay_cli(
        tmp_path,
        str(trace_path),
        "--replay-mode",
        "offline",
        "--router-mode",
        "kv_router",
        "--num-workers",
        "4",
        "--report-json",
        str(report_path),
        "--extra-engine-args",
        '{"block_size":64,"speedup_ratio":1000.0}',
    )

    report = _assert_replay_cli_outputs(completed, report_path)
    _assert_basic_report_counts(
        report,
        num_requests=10,
        input_tokens=250,
        output_tokens=25,
    )
    _assert_basic_report_metrics(report)


@pytest.mark.timeout(30)
def test_replay_cli_subprocess_trace_disagg_smoke(tmp_path):
    trace_path = _write_cli_smoke_trace(tmp_path)
    report_path = tmp_path / "trace_disagg_report.json"

    completed = _run_replay_cli(
        tmp_path,
        str(trace_path),
        "--replay-mode",
        "offline",
        "--router-mode",
        "kv_router",
        "--num-prefill-workers",
        "2",
        "--num-decode-workers",
        "2",
        "--report-json",
        str(report_path),
        "--prefill-engine-args",
        '{"block_size":64,"speedup_ratio":1000.0,"worker_type":"prefill"}',
        "--decode-engine-args",
        '{"block_size":64,"speedup_ratio":1000.0,"worker_type":"decode"}',
    )

    report = _assert_replay_cli_outputs(completed, report_path)
    _assert_basic_report_counts(
        report,
        num_requests=10,
        input_tokens=250,
        output_tokens=25,
    )
    _assert_basic_report_metrics(report)


@pytest.mark.timeout(30)
def test_replay_cli_subprocess_multiturn_trace_smoke(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)
    report_path = tmp_path / "multiturn_trace_report.json"

    completed = _run_replay_cli(
        tmp_path,
        str(trace_path),
        "--replay-mode",
        "online",
        "--router-mode",
        "kv_router",
        "--num-workers",
        "2",
        "--report-json",
        str(report_path),
        "--extra-engine-args",
        '{"block_size":64,"speedup_ratio":1000.0}',
    )

    report = _assert_replay_cli_outputs(completed, report_path)
    _assert_basic_report_counts(
        report,
        num_requests=4,
        input_tokens=64,
        output_tokens=2,
    )
    _assert_basic_report_metrics(report)
