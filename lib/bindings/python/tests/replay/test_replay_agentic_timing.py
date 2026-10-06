# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.replay import run_trace_replay

from .replay_utils import _report_summary, _vllm_args

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.timeout(120),
]


def _write_timing_trace(path, *, start_ms, root_gap_ms, dependency_delay_ms):
    """Original fixture with roots and two turns sharing a completion trigger."""
    records = [
        {
            "schema": "dynamo.agentic_mooncake",
            "version": 2,
            "block_size": 64,
            "hash_id_scope": "local",
            "source": {"format": "self-authored-test", "digest": "finite-timing"},
        }
    ]
    for index, (request_id, ready_at_ms, dependencies) in enumerate(
        [
            ("root", start_ms, []),
            ("independent", start_ms + root_gap_ms, []),
            (
                "anchor",
                start_ms,
                [
                    {
                        "request_id": "root",
                        "trigger": "completion",
                        "delay_ms": 0.0,
                        "relation": "sequence",
                    }
                ],
            ),
            (
                "dependent",
                start_ms,
                [
                    {
                        "request_id": "root",
                        "trigger": "completion",
                        "delay_ms": dependency_delay_ms,
                        "relation": "sequence",
                    }
                ],
            ),
        ],
        start=1,
    ):
        records.append(
            {
                "request_id": request_id,
                "play_id": "play",
                "session_id": request_id,
                "model": "fixture-model",
                "input_length": 64,
                "output_length": 2,
                "hash_ids": [index],
                "not_before_ms": ready_at_ms,
                "dependencies": dependencies,
            }
        )
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return path


@pytest.mark.parametrize("replay_mode", ["offline", "online"])
@pytest.mark.parametrize("agentic_lanes", [None, 1])
def test_agentic_mooncake_finite_binding_preserves_normalized_scaled_timing(
    tmp_path, replay_mode, agentic_lanes
):
    source = _write_timing_trace(
        tmp_path / "source.jsonl",
        start_ms=100.0,
        root_gap_ms=48.0,
        dependency_delay_ms=32.0,
    )
    # The former file-loading path normalized starts before scaling timers.
    # Author that result independently as a control for the binding dispatch.
    prepared = _write_timing_trace(
        tmp_path / "prepared.jsonl",
        start_ms=0.0,
        root_gap_ms=12.0,
        dependency_delay_ms=8.0,
    )
    timing = []
    for trace, speedup in [(source, 4.0), (prepared, 1.0)]:
        report_path = tmp_path / f"{trace.stem}-report.jsonl"
        report = run_trace_replay(
            trace,
            extra_engine_args=_vllm_args(),
            replay_mode=replay_mode,
            trace_format="agentic_mooncake",
            execution_model="fixture-model",
            agentic_lanes=agentic_lanes,
            arrival_speedup_ratio=speedup,
            report_jsonl_path=report_path,
        )
        assert _report_summary(report)["completed_requests"] == 4
        records = [json.loads(line) for line in report_path.read_text().splitlines()]
        by_id = {record["request_id"]: record for record in records}
        root, independent, anchor, dependent = (
            by_id[name] for name in ("root", "independent", "anchor", "dependent")
        )
        # Online completion notification can lag terminal delivery. Both turns
        # share that notification, so their scheduled arrival gap isolates the
        # authored delay without any wall-clock tolerance.
        timing.append(
            (
                root["arrival_time_ms"],
                independent["arrival_time_ms"],
                dependent["arrival_time_ms"] - anchor["arrival_time_ms"],
            )
        )
        assert timing[-1] == pytest.approx((0.0, 12.0, 8.0))
    assert timing[0] == pytest.approx(timing[1])
