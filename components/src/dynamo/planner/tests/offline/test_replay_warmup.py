# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before replay CLI imports.
# ruff: noqa: E402

"""Regression tests for planner replay warmup wiring."""

import json

import pytest

pytest.importorskip(
    "aisimulate.replay",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

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
    captured = {}

    class FakeAdapter:
        def _is_easy_mode(self):
            return True

        def set_bootstrap_metadata(self, metadata):
            captured["bootstrap_metadata"] = metadata

    adapter = FakeAdapter()

    def fake_create_replay_planner_adapter(*, warmup_observations, **_kwargs):
        captured["warmup_observations"] = warmup_observations
        return adapter

    monkeypatch.setattr(
        replay_adapter_module,
        "create_replay_planner_adapter",
        fake_create_replay_planner_adapter,
    )

    result = replay_main._prepare_planner_replay(
        extra_engine_args=replay_main.MockEngineArgs(
            block_size=64, speedup_ratio=1000.0
        ),
        prefill_engine_args=None,
        decode_engine_args=None,
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

    assert result is adapter
    observations = captured["warmup_observations"]
    assert [(item.num_req, item.isl, item.osl) for item in observations] == [
        (1.0, 64.0, 4.0),
        (1.0, 128.0, 5.0),
    ]
    assert captured["bootstrap_metadata"] == {"status": "not_required"}
