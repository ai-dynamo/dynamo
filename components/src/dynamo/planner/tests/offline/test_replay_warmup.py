# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for planner replay warmup wiring."""

import json
from types import SimpleNamespace

import pytest

import dynamo.mocker as mocker_module
import dynamo.planner.offline.replay_adapter as replay_adapter_module
import dynamo.replay.main as replay_main

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


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
