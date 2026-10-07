# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for standalone interactive Autoscaling Arena reports."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import pytest
from autoscaling_arena.html_report import (
    _downsample,
    _frontend_report_data,
    build_match_report_data,
    render_match_report,
    trace_arrival_series,
)


def _write_trace(
    tmp_path: Path,
    timestamps_ms: list[float],
    *,
    name: str = "workload.jsonl",
) -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "input_length": 64,
                    "output_length": 32,
                    "hash_ids": [index],
                }
            )
            + "\n"
            for index, timestamp in enumerate(timestamps_ms)
        )
    )
    return path


def _timeline(
    *,
    timestamp_s: float = 10.0,
    window_start_s: float = 6.0,
    active_prefill: int = 2,
    active_decode: int = 4,
    provisioned_prefill: int = 3,
    provisioned_decode: int = 5,
    requested_prefill: int | None = 4,
    requested_decode: int | None = 6,
    total_queued: int | None = None,
    queued_prefill: int | None = None,
    queued_decode: int | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "timestamp_s": timestamp_s,
            "window_start_s": window_start_s,
            "completed_requests": 7,
            "total_queued_requests": total_queued,
            "queued_prefill_requests": queued_prefill,
            "queued_decode_requests": queued_decode,
            "ttft_sample_count": 5,
            "itl_sample_count": 6,
            "mean_ttft_ms": 41.5,
            "mean_tpot_ms": 8.25,
            "active_prefill_replicas": active_prefill,
            "active_decode_replicas": active_decode,
            "provisioned_prefill_replicas": provisioned_prefill,
            "provisioned_decode_replicas": provisioned_decode,
            "scaling_decision": (
                requested_prefill is not None or requested_decode is not None
            ),
            "requested_prefill_replicas": requested_prefill,
            "requested_decode_replicas": requested_decode,
        }
    ]


def _result(
    autoscaler: str,
    trace: Path,
    *,
    rank_by: str = "goodput_rps",
    rank_value: float | None = 10.0,
    workload: str = "burst",
    sla: str = "interactive",
    repetition: int = 0,
    status: str = "ok",
    timeline: list[dict[str, Any]] | None = None,
    arrival_speedup: float = 1.0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "run_id": f"{autoscaler}-{workload}-{sla}-{repetition}",
        "backend": "sim",
        "autoscaler": autoscaler,
        "workload": workload,
        "sla": sla,
        "repetition": repetition,
        "seed": 17 + repetition,
        "status": status,
        "metrics": {
            rank_by: rank_value,
            "duration_s": 12.0,
            "p99_ttft_ms": (rank_value if rank_by == "p99_ttft_ms" else 75.0),
        },
        "evaluation": {
            "workload": workload,
            "seed": 17 + repetition,
            "repetition": repetition,
            "arrival_speedup": arrival_speedup,
            "trace": {
                "path": str(trace),
                "sha256": "fixture-sha",
                "size_bytes": trace.stat().st_size,
            },
            "sla": {
                "ttft_ms": 100.0 if sla == "interactive" else 250.0,
                "itl_ms": 15.0,
                "e2e_ms": 1_000.0,
            },
        },
        "runtime": {
            "topology": "disagg",
            "engines": {
                "prefill": {"config": {"num_gpus": 2}},
                "decode": {"config": {"num_gpus": 4}},
            },
        },
        "timeline": _timeline() if timeline is None else timeline,
    }
    if status != "ok":
        result["error"] = {
            "type": "AutoscalerCrashed",
            "message": f"{autoscaler} failed during replay",
        }
    return result


def _report(
    results: list[dict[str, Any]],
    *,
    rank_by: str = "goodput_rps",
    concurrency: int | None = None,
    title: str = "Arena fixture",
    description: str = "Standalone report contract",
) -> dict[str, Any]:
    failed = sum(result["status"] != "ok" for result in results)
    return {
        "match": {
            "name": title,
            "description": description,
            "backend": "sim",
        },
        "summary": {
            "status": "ok" if not failed else "failed",
            "planned_runs": len(results),
            "executed_runs": len(results),
            "succeeded_runs": len(results) - failed,
            "failed_runs": failed,
            "skipped_runs": 0,
            "rank_by": rank_by,
            "metrics": [rank_by, "duration_s", "p99_ttft_ms"],
        },
        "provenance": {
            "finished_at": "2026-07-29T12:00:00+00:00",
            "git_commit": "1234567890abcdef",
            "config_sha256": "d" * 64,
            "replay_config_sha256": "c" * 64,
            "session_id": "fixture-session",
            "config_file": "fixture.match.yaml",
            "replay": {
                "kind": "match_config",
                "config_path": "configs/fixture.match.yaml",
            },
        },
        "resolved_config": {
            "backend": {
                "type": "sim",
                "topology": "disagg",
                "model": {"name": "fixture/model"},
                "replay": {
                    "concurrency": concurrency,
                    "telemetry_sample_interval_s": 5.0,
                },
                "engines": {
                    "prefill": {
                        "backend": "vllm",
                        "system": "h200_sxm",
                        "num_gpus": 2,
                        "runtime": {"cold_start_delay_s": 30.0},
                    },
                    "decode": {
                        "backend": "vllm",
                        "system": "h200_sxm",
                        "num_gpus": 4,
                        "runtime": {"cold_start_delay_s": 45.0},
                    },
                },
            }
        },
        "results": results,
    }


def _scope(
    data: dict[str, Any],
    *,
    configuration_id: str = "interactive::r0",
    workload: str = "burst",
) -> dict[str, Any]:
    return next(
        scope
        for scope in data["scopes"]
        if scope["configuration_id"] == configuration_id
        and scope["workload"] == workload
    )


def test_report_data_groups_configuration_and_workload_and_keeps_failures(
    tmp_path: Path,
):
    burst = _write_trace(tmp_path, [1_000, 1_250, 2_000], name="burst.jsonl")
    steady = _write_trace(tmp_path, [1_000, 2_000], name="steady.jsonl")
    report = _report(
        [
            _result("planner", burst, rank_value=12.0),
            _result("reactive", burst, status="failed", rank_value=None),
            _result("planner", steady, workload="steady", rank_value=8.0),
            _result(
                "planner",
                burst,
                workload="burst",
                sla="relaxed",
                repetition=1,
                rank_value=14.0,
            ),
        ]
    )

    data = build_match_report_data(report)

    assert [
        (configuration["id"], configuration["repetition"])
        for configuration in data["configurations"]
    ] == [("interactive::r0", 0), ("relaxed::r1", 1)]
    assert data["workloads"] == ["burst", "steady"]
    assert {
        (scope["configuration_id"], scope["workload"]) for scope in data["scopes"]
    } == {
        ("interactive::r0", "burst"),
        ("interactive::r0", "steady"),
        ("relaxed::r1", "burst"),
    }
    assert [item["name"] for item in data["autoscalers"]] == [
        "planner",
        "reactive",
    ]
    assert data["topology"] == "disagg"
    assert data["telemetry_sample_interval_s"] == 5.0
    assert data["cold_start_label"] == "P 30 s · D 45 s"

    burst_scope = _scope(data)
    assert [result["autoscaler"] for result in burst_scope["results"]] == [
        "planner",
        "reactive",
    ]
    assert burst_scope["results"][0]["rank"] == 1
    assert burst_scope["results"][0]["replay_command"] == (
        "python scripts/run_match_config.py configs/fixture.match.yaml "
        f"--expect-config-sha256 {'c' * 64} "
        "--run-id planner-burst-interactive-0 --no-publish"
    )
    failure = burst_scope["results"][1]
    assert failure["rank"] is None
    assert failure["status"] == "failed"
    assert failure["error"] == {
        "type": "AutoscalerCrashed",
        "message": "reactive failed during replay",
    }


@pytest.mark.parametrize(
    ("rank_by", "best", "worse", "direction"),
    [
        ("goodput_rps", 20.0, 10.0, "higher"),
        ("p99_ttft_ms", 20.0, 80.0, "lower"),
    ],
)
def test_report_ranking_honors_metric_direction_and_sorts_unranked_last(
    tmp_path: Path,
    rank_by: str,
    best: float,
    worse: float,
    direction: str,
):
    trace = _write_trace(tmp_path, [0, 1_000])
    report = _report(
        [
            _result(
                "worse",
                trace,
                rank_by=rank_by,
                rank_value=worse,
            ),
            _result(
                "missing",
                trace,
                rank_by=rank_by,
                rank_value=None,
            ),
            _result(
                "crashed",
                trace,
                rank_by=rank_by,
                rank_value=best,
                status="failed",
            ),
            _result(
                "best",
                trace,
                rank_by=rank_by,
                rank_value=best,
            ),
        ],
        rank_by=rank_by,
    )

    data = build_match_report_data(report)
    ranked = _scope(data)["results"]

    assert data["rank_direction"] == direction
    assert [result["autoscaler"] for result in ranked] == [
        "best",
        "worse",
        "missing",
        "crashed",
    ]
    assert [result["rank"] for result in ranked] == [1, 2, None, None]


def test_timeline_preserves_latency_and_uses_asymmetric_role_gpu_widths(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [0, 1_000])
    data = build_match_report_data(
        _report(
            [
                _result(
                    "planner",
                    trace,
                    timeline=_timeline(
                        total_queued=17,
                        queued_prefill=3,
                        queued_decode=5,
                    ),
                )
            ]
        )
    )

    point = _scope(data)["results"][0]["timeline"][0]
    assert point["time_s"] == 10.0
    assert point["latency_time_s"] == 8.0
    assert point["arriving_requests"] is None
    assert point["offered_requests"] is None
    assert point["completed_requests"] == 7
    # The role details deliberately do not sum to the saved total: the report
    # must display the authoritative total and never reconstruct one.
    assert point["total_queued_requests"] == 17
    assert point["queued_prefill_requests"] == 3
    assert point["queued_decode_requests"] == 5
    assert point["ttft_samples"] == 5
    assert point["tpot_samples"] == 6
    assert point["ttft_ms"] == 41.5
    assert point["tpot_ms"] == 8.25
    assert point["active_replicas"] == 6
    assert (point["active_prefill"], point["active_decode"]) == (2, 4)
    assert point["provisioned_replicas"] == 8
    assert point["provisioned_gpus"] == 3 * 2 + 5 * 4
    assert point["requested_replicas"] == 10
    assert (point["requested_prefill"], point["requested_decode"]) == (4, 6)
    assert point["requested_gpus"] == 4 * 2 + 6 * 4
    assert (
        point["prefill_gpus_per_replica"],
        point["decode_gpus_per_replica"],
    ) == (2, 4)


def test_decision_only_rows_are_identified_for_capacity_filtering(
    tmp_path: Path,
) -> None:
    trace = _write_trace(tmp_path, [0, 1_000])
    telemetry = _timeline(
        timestamp_s=5.0,
        requested_prefill=None,
        requested_decode=None,
    )[0]
    decision = {
        "timestamp_s": 7.5,
        "decision_only": True,
        "scaling_decision": True,
        "requested_prefill_replicas": 2,
        "requested_decode_replicas": 3,
    }
    data = build_match_report_data(
        _report([_result("planner", trace, timeline=[telemetry, decision])])
    )

    normalized = _scope(data)["results"][0]["timeline"]
    frontend = _scope(_frontend_report_data(data))["results"][0]["timeline"]

    assert normalized[1]["decision_only"] is True
    assert frontend[1]["decision_only"] is True


def test_open_loop_arrivals_are_reconstructed_and_speedup_adjusted(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [5_000, 5_500, 6_000])
    report = _report([_result("planner", trace, arrival_speedup=2.0, timeline=[])])

    arrivals = _scope(build_match_report_data(report))["arrivals"]

    assert arrivals["status"] == "ok"
    assert arrivals["bucket_width_s"] == 0.1
    assert sum(point["count"] for point in arrivals["points"]) == 3
    assert arrivals["points"][0] == {"time_s": 0.0, "count": 1, "rps": 10.0}
    assert arrivals["points"][-1]["time_s"] == 0.5
    assert arrivals["points"][-1]["count"] == 1


def test_saved_offered_requests_are_preferred_and_completions_stay_distinct(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [0, 1_000, 2_000])
    timeline = _timeline(timestamp_s=10.0, window_start_s=5.0)
    timeline[0]["offered_requests"] = 25
    timeline[0]["completed_requests"] = 7
    result = _result("planner", trace, timeline=timeline)
    result["timeline_semantics"] = "offered_and_completed_v2"

    scope = _scope(build_match_report_data(_report([result])))

    point = scope["results"][0]["timeline"][0]
    assert point["offered_requests"] == 25
    assert point["completed_requests"] == 7
    assert scope["arrivals"]["source"] == "saved_offered_requests"
    assert scope["arrivals"]["points"] == [
        {
            "time_s": 5.0,
            "count": 25,
            "rps": 5.0,
            "window_width_s": 5.0,
        }
    ]


def test_saved_arriving_requests_are_primary_over_legacy_offered_alias(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [0, 1_000, 2_000])
    timeline = _timeline(timestamp_s=10.0, window_start_s=5.0)
    timeline[0]["arriving_requests"] = 31
    timeline[0]["offered_requests"] = 25
    timeline[0]["completed_requests"] = 7
    result = _result("planner", trace, timeline=timeline)
    result["timeline_semantics"] = "arriving_and_completed_v2"

    scope = _scope(build_match_report_data(_report([result])))

    point = scope["results"][0]["timeline"][0]
    assert point["arriving_requests"] == 31
    assert point["offered_requests"] == 31
    assert point["completed_requests"] == 7
    assert scope["arrivals"]["source"] == "saved_arriving_requests"
    assert scope["arrivals"]["points"][0]["count"] == 31


def test_completed_requests_become_offered_only_with_explicit_legacy_marker(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [0, 1_000])
    unmarked = _result("unmarked", trace, timeline=_timeline())
    marked = _result("marked", trace, timeline=_timeline())
    marked["timeline_semantics"] = "legacy_num_req_as_completed"
    marked["evaluation"]["trace"]["request_rows"] = 10

    unmarked_data = _scope(build_match_report_data(_report([unmarked])))
    marked_data = _scope(build_match_report_data(_report([marked])))

    unmarked_point = unmarked_data["results"][0]["timeline"][0]
    assert unmarked_point["offered_requests"] is None
    assert unmarked_point["completed_requests"] == 7
    assert unmarked_data["arrivals"].get("source") is None
    marked_point = marked_data["results"][0]["timeline"][0]
    assert marked_point["offered_requests"] == 7
    assert marked_point["completed_requests"] is None
    assert marked_data["arrivals"]["source"] == "legacy_marked_offered_requests"
    assert "mislabeled" in marked_data["arrivals"]["note"]
    assert "cover 7 of 10 arriving requests" in marked_data["arrivals"]["note"]


def test_queue_depth_is_never_inferred_from_role_details(tmp_path: Path):
    trace = _write_trace(tmp_path, [0, 1_000])
    timeline = _timeline(queued_prefill=8, queued_decode=13)

    point = _scope(
        build_match_report_data(_report([_result("planner", trace, timeline=timeline)]))
    )["results"][0]["timeline"][0]

    assert point["total_queued_requests"] is None
    assert point["queued_prefill_requests"] == 8
    assert point["queued_decode_requests"] == 13
    assert point["scheduler_waiting_requests"] is None
    assert point["router_pending_requests"] is None
    assert point["active_kv_cache_utilization"] is None
    assert point["scheduler_cache_reuse"] is None


def test_timeline_normalizes_queue_layers_kv_metrics_and_raw_rows(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [0, 1_000])
    timeline = _timeline(total_queued=17, queued_prefill=7, queued_decode=10)
    raw_prefill = {
        "worker_id": "prefill-1",
        "dp_rank": 0,
        "active_blocks": 10,
        "inactive_blocks": 5,
        "total_blocks": 100,
    }
    raw_decode = {
        "worker_id": "decode-1",
        "dp_rank": 0,
        "active_blocks": 20,
        "inactive_blocks": 10,
        "total_blocks": 200,
    }
    timeline[0].update(
        {
            "scheduler_waiting_requests": 12,
            "scheduler_waiting_prefill_requests": 5,
            "scheduler_waiting_decode_requests": 7,
            "router_pending_requests": 5,
            "router_pending_prefill_requests": 2,
            "router_pending_decode_requests": 3,
            "queue_telemetry_semantics": ("scheduler_waiting_plus_router_pending"),
            "active_kv_blocks": 30,
            "inactive_kv_blocks": 15,
            "total_kv_blocks": 300,
            "active_kv_cache_utilization": 0.1,
            "physical_kv_cache_utilization": 0.15,
            "prefill_active_kv_cache_utilization": 0.1,
            "decode_active_kv_cache_utilization": 0.1,
            "prefill_physical_kv_cache_utilization": 0.15,
            "decode_physical_kv_cache_utilization": 0.15,
            "scheduler_cache_hit_tokens": 75,
            "scheduler_cache_total_tokens": 300,
            "scheduler_cache_reuse": 0.25,
            "mean_router_kv_hit_rate": 0.4,
            "router_kv_hit_sample_count": 11,
            "prefill_scheduler_cache_reuse": 0.2,
            "decode_scheduler_cache_reuse": 0.275,
            "total_running_requests": 9,
            "preemptions_total": 4,
            "prefill_scheduler_metrics": [raw_prefill],
            "decode_scheduler_metrics": [raw_decode],
            "scheduler_metrics_payload_available": True,
            "scheduler_metrics_available": True,
        }
    )

    data = build_match_report_data(
        _report([_result("planner", trace, timeline=timeline)])
    )
    point = _scope(data)["results"][0]["timeline"][0]

    assert point["total_queued_requests"] == 17
    assert point["scheduler_waiting_requests"] == 12
    assert point["router_pending_requests"] == 5
    assert point["active_kv_cache_utilization"] == pytest.approx(0.1)
    assert point["physical_kv_cache_utilization"] == pytest.approx(0.15)
    assert point["scheduler_cache_reuse"] == pytest.approx(0.25)
    assert point["router_kv_hit_rate"] == pytest.approx(0.4)
    assert point["router_kv_hit_samples"] == 11
    assert point["active_kv_blocks"] == 30
    assert point["scheduler_cache_hit_tokens"] == 75
    assert point["total_running_requests"] == 9
    assert point["preemptions_total"] == 4
    assert point["preemptions"] == 4
    assert point["prefill_scheduler_metrics"] == [raw_prefill]
    assert point["decode_scheduler_metrics"] == [raw_decode]
    assert point["scheduler_metrics_available"] is True
    assert point["scheduler_metrics_payload_available"] is True

    frontend_point = _scope(_frontend_report_data(data))["results"][0]["timeline"][0]
    assert frontend_point["total_queued_requests"] == 17
    assert "prefill_scheduler_metrics" not in frontend_point
    assert "decode_scheduler_metrics" not in frontend_point
    assert "scheduler_metrics_payload_available" not in frontend_point


def test_timeline_downsampling_preserves_lane_peaks_and_scaling_decision(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [0, 1_000])
    timeline: list[dict[str, Any]] = []
    for index in range(1_605):
        sample = _timeline(
            timestamp_s=float(index),
            window_start_s=float(max(0, index - 1)),
            total_queued=0,
            queued_prefill=0,
            queued_decode=0,
        )[0]
        sample["scaling_decision"] = False
        sample["requested_prefill_replicas"] = None
        sample["requested_decode_replicas"] = None
        timeline.append(sample)
    timeline[731]["total_queued_requests"] = 999
    timeline[731]["queued_prefill_requests"] = 700
    timeline[731]["queued_decode_requests"] = 299
    timeline[732]["active_kv_cache_utilization"] = 0.97
    timeline[733]["physical_kv_cache_utilization"] = 0.99
    timeline[734]["scheduler_cache_reuse"] = 0.93
    timeline[735]["mean_ttft_ms"] = 9_999.0
    timeline[736]["mean_tpot_ms"] = 999.0
    timeline[743]["scaling_decision"] = True
    timeline[743]["requested_prefill_replicas"] = 7
    timeline[743]["requested_decode_replicas"] = 9

    normalized = _scope(
        build_match_report_data(_report([_result("planner", trace, timeline=timeline)]))
    )["results"][0]["timeline"]

    assert len(normalized) <= 1_500
    assert any(point["total_queued_requests"] == 999 for point in normalized)
    assert any(
        point["active_kv_cache_utilization"] == pytest.approx(0.97)
        for point in normalized
    )
    assert any(
        point["physical_kv_cache_utilization"] == pytest.approx(0.99)
        for point in normalized
    )
    assert any(
        point["scheduler_cache_reuse"] == pytest.approx(0.93) for point in normalized
    )
    assert any(point["ttft_ms"] == 9_999.0 for point in normalized)
    assert any(point["tpot_ms"] == 999.0 for point in normalized)
    assert any(
        point["time_s"] == 743.0 and point["requested_replicas"] == 16
        for point in normalized
    )


def test_native_dynamo_arrivals_parse_nested_timestamps_across_gzip_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    shard_paths = [tmp_path / "part-1.jsonl.gz", tmp_path / "part-2.jsonl.gz"]
    records = [
        [
            {
                "timestamp": 999_000,
                "event": {"request": {"request_received_ms": 2_000}},
            },
            {"event": {"request": {"request_received_ms": 3_000}}},
        ],
        [{"event": {"request": {"request_received_ms": 1_000}}}],
    ]
    for path, shard_records in zip(shard_paths, records, strict=True):
        with gzip.open(path, "wt") as handle:
            for record in shard_records:
                handle.write(json.dumps(record) + "\n")

    original_gzip_open = gzip.open
    opened: list[Path] = []

    def counted_gzip_open(path, *args, **kwargs):
        opened.append(Path(path))
        return original_gzip_open(path, *args, **kwargs)

    monkeypatch.setattr("autoscaling_arena.html_report.gzip.open", counted_gzip_open)

    arrivals = trace_arrival_series(shard_paths, trace_format="dynamo", speedup=2.0)

    assert arrivals["status"] == "ok"
    assert sum(point["count"] for point in arrivals["points"]) == 3
    assert arrivals["points"][0]["time_s"] == 0.0
    assert arrivals["points"][-1]["time_s"] == 1.0
    assert opened == shard_paths


def test_embedded_arrivals_do_not_require_a_trace_path(tmp_path: Path):
    trace = _write_trace(tmp_path, [0])
    result = _result("planner", trace, timeline=[])
    result["evaluation"]["trace"].pop("path")
    result["evaluation"]["arrival_series"] = {
        "status": "ok",
        "note": "Embedded arriving traffic.",
        "bucket_width_s": 1.0,
        "points": [{"time_s": 0.0, "count": 3, "rps": 3.0}],
    }

    arrivals = _scope(build_match_report_data(_report([result])))["arrivals"]

    assert arrivals["status"] == "ok"
    assert arrivals["points"] == [{"time_s": 0.0, "count": 3, "rps": 3.0}]


def test_closed_loop_arrivals_are_explicitly_unavailable(tmp_path: Path):
    trace = _write_trace(tmp_path, [0, 1_000, 2_000])
    report = _report(
        [_result("planner", trace)],
        concurrency=8,
    )

    arrivals = _scope(build_match_report_data(report))["arrivals"]

    assert arrivals["status"] == "unavailable"
    assert arrivals["points"] == []
    assert arrivals["bucket_width_s"] is None
    assert "closed-loop" in arrivals["note"].lower()


def test_cache_reuse_keeps_token_weighted_timeline_and_aggregate(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [0, 1_000])
    result = _result("planner", trace)
    result["cache"] = {
        "prefix_cache_reused_ratio": 0.3863,
        "first_admission_prefix_cache_reused_ratio": 0.25,
        "timeline_available": True,
        "timeline_source": "per_request_admission_windows",
        "timeline": [
            {
                "time_s": 5.0,
                "window_start_s": 0.0,
                "input_tokens": 100,
                "reused_input_tokens": 30,
                "completed_requests": 4,
                # The reporter must trust the saved token totals, not a stale
                # precomputed ratio.
                "prefix_cache_reused_ratio": 0.99,
            },
            {
                "time_s": 10.0,
                "window_start_s": 5.0,
                "input_tokens": 300,
                "reused_input_tokens": 150,
                "completed_requests": 8,
                "prefix_cache_reused_ratio": 0.01,
            },
        ],
    }

    cache = _scope(build_match_report_data(_report([result])))["results"][0]["cache"]

    assert cache["status"] == "ok"
    assert cache["prefix_cache_reused_ratio"] == 0.3863
    assert cache["first_admission_prefix_cache_reused_ratio"] == 0.25
    assert cache["timeline_available"] is True
    assert [point["prefix_cache_reused_ratio"] for point in cache["timeline"]] == [
        0.3,
        0.5,
    ]


def test_cache_reuse_aggregate_only_is_explicit_and_never_invents_points(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [0, 1_000])
    result = _result("planner", trace)
    result["cache"] = {
        "prefix_cache_reused_ratio": 0.3863,
        "timeline_available": False,
        "timeline_source": "not_captured",
        "timeline": [],
    }

    cache = _scope(build_match_report_data(_report([result])))["results"][0]["cache"]

    assert cache["status"] == "aggregate_only"
    assert cache["timeline"] == []
    assert "exact time series is unavailable" in cache["note"]


def test_downsample_preserves_router_kv_hit_rate_peak() -> None:
    rows = [
        {
            "requested_replicas": None,
            "total_queued_requests": 0,
            "router_kv_hit_rate": 1.0 if index == 809 else 0.0,
        }
        for index in range(1_605)
    ]

    sampled = _downsample(rows)

    assert any(row["router_kv_hit_rate"] == 1.0 for row in sampled)


def test_downsample_preserves_constant_total_prefill_decode_swap() -> None:
    rows = []
    for index in range(1_605):
        prefill, decode = (1, 3) if index < 14 else (2, 2)
        rows.append(
            {
                "time_s": index,
                "requested_replicas": None,
                "total_queued_requests": 0,
                "active_replicas": 4,
                "active_prefill": prefill,
                "active_decode": decode,
                "provisioned_replicas": 4,
                "provisioned_prefill": prefill,
                "provisioned_decode": decode,
                "provisioned_gpus": 4,
            }
        )

    sampled = _downsample(rows)
    sampled_times = {row["time_s"] for row in sampled}

    # The generic queue sampler drops t=14 for this input. Both sides of the
    # role-only transition must nevertheless remain for an exact step trace.
    assert {13, 14} <= sampled_times


def test_downsample_stays_bounded_under_pathological_capacity_churn() -> None:
    rows = []
    for index in range(5_000):
        prefill, decode = (1, 3) if index % 2 else (2, 2)
        rows.append(
            {
                "requested_replicas": 4,
                "total_queued_requests": 0,
                "active_replicas": 4,
                "active_prefill": prefill,
                "active_decode": decode,
                "provisioned_replicas": 4,
                "provisioned_prefill": prefill,
                "provisioned_decode": decode,
                "provisioned_gpus": 4,
            }
        )

    assert len(_downsample(rows)) <= 1_500


def test_downsample_reserves_budget_for_measured_telemetry() -> None:
    rows = []
    for index in range(5_000):
        decision_only = index % 2 == 1
        rows.append(
            {
                "decision_only": decision_only,
                "requested_replicas": 4 if decision_only else None,
                "total_queued_requests": None if decision_only else index % 17,
                "ttft_ms": None if decision_only else float(index),
                "active_replicas": 4,
                "active_prefill": 1,
                "active_decode": 3,
                "provisioned_replicas": 4,
                "provisioned_prefill": 1,
                "provisioned_decode": 3,
                "provisioned_gpus": 4,
            }
        )

    sampled = _downsample(rows)

    assert len(sampled) <= 1_500
    assert sum(not row["decision_only"] for row in sampled) >= 1_000
    assert any(row["ttft_ms"] == 4_998.0 for row in sampled)


def test_replay_command_guards_saved_dynamo_trace_content(tmp_path: Path) -> None:
    trace = _write_trace(tmp_path, [0, 1_000])
    result = _result("planner", trace)
    digest = "a" * 64
    result["evaluation"]["trace"] = {
        "format": "dynamo",
        "shards": [{"sha256": digest}],
    }

    replay = _scope(build_match_report_data(_report([result])))["results"][0]

    assert replay["replay_trace_guarded"] is True
    assert f"--expect-trace-sha256 {digest}" in replay["replay_command"]


def test_report_without_validated_config_has_no_replay_command(
    tmp_path: Path,
) -> None:
    trace = _write_trace(tmp_path, [0, 1_000])
    report = _report([_result("planner", trace)])
    report["provenance"].pop("replay")

    replay = _scope(build_match_report_data(report))["results"][0]

    assert replay["replay_command"] is None


def test_legacy_config_digest_is_not_used_as_a_replay_guard(
    tmp_path: Path,
) -> None:
    trace = _write_trace(tmp_path, [0, 1_000])
    report = _report([_result("planner", trace)])
    report["provenance"].pop("replay_config_sha256")

    replay = _scope(build_match_report_data(report))["results"][0]

    assert replay["replay_command"] is None


def test_renderer_is_standalone_interactive_scoped_and_script_safe(
    tmp_path: Path,
):
    trace = _write_trace(tmp_path, [0, 1_000])
    attack = '</script><script src="https://evil.invalid/x.js">owned</script>'
    report = _report(
        [
            _result(attack, trace, rank_value=12.0),
            _result("reactive", trace, rank_value=10.0),
        ],
        title=f"Arena {attack}",
        description=f"Description {attack}",
    )

    rendered = render_match_report(report)

    assert rendered.startswith("<!doctype html>")
    assert not re.search(r"<script\b[^>]*\bsrc\s*=", rendered, re.IGNORECASE)
    assert "Plotly.react" in rendered
    assert 'xaxis: "x", yaxis: "y2"' in rendered
    assert 'yaxis2: axis("Requests in queue", [.76, .865])' in rendered
    assert "fixedrange: true" in rendered
    assert 'hoversubplots: "axis"' in rendered
    assert 'id="configuration"' in rendered
    assert 'id="workload"' in rendered
    assert 'id="autoscaler-legend"' in rendered
    assert 'id="leaderboard-body"' in rendered
    assert 'id="arena-chart"' in rendered
    assert 'input.type = "checkbox"' in rendered
    assert "Arriving requests (req/s)" in rendered
    assert "Total requests in queue" in rendered
    assert "TTFT (ms)" in rendered
    assert "TPOT (ms)" in rendered
    assert "KV cache utilization (%)" in rendered
    assert "KV hit / cache reuse (%)" in rendered
    assert "router KV hit rate" in rendered
    assert "scheduler cache reuse" in rendered
    assert "if (!schedulerReuse.length && cachePoints.length)" in rendered
    assert (
        "if (!routerKvHit.length && !schedulerReuse.length && cachePoints.length)"
        not in rendered
    )
    assert "KV reuse (whole run)" in rendered
    assert "Ready replicas" in rendered
    assert "Provisioned GPUs" in rendered
    assert "hollow triangles" in rendered
    assert (
        "const capacity = result.timeline.filter(point => !point.decision_only);"
        in rendered
    )
    assert "requested %{y} ${role.name} replicas" in rendered
    assert 'mode: "lines+markers"' not in rendered
    assert 'mode: "lines"' in rendered
    assert "elapsedTimeTicks" in rendered
    assert "Elapsed time (${units[1]})" in rendered
    assert "thicker lines and upward triangles are prefill" in rendered
    assert 'id="replay-dialog"' in rendered
    assert 'id="replay-copy"' in rendered
    assert "Replay cmd" in rendered
    assert "Cold start</strong>P 30 s · D 45 s" in rendered
    assert "arriving requests" in rendered
    assert "%{y} total requests in queue" in rendered
    assert "preemptions in window" in rendered
    assert "Total requests in queue unavailable; no values inferred." in rendered
    assert "Eight aligned time lanes" in rendered
    assert "metric ranges stay fixed" in rendered
    assert "no time series is inferred" in rendered

    payload_match = re.search(
        r'<script id="arena-report-data" type="application/json">' r"(.*?)</script>",
        rendered,
        re.DOTALL,
    )
    assert payload_match is not None
    payload = payload_match.group(1)
    assert "<" not in payload
    assert "\\u003c/script\\u003e" in payload
    embedded = json.loads(payload)
    assert embedded["title"] == f"Arena {attack}"
    assert embedded["description"] == f"Description {attack}"
    assert embedded["autoscalers"][0]["name"] == attack
    assert embedded["topology"] == "disagg"
    assert embedded["telemetry_sample_interval_s"] == 5.0
    assert embedded["cold_start_label"] == "P 30 s · D 45 s"
    assert embedded["scopes"][0]["results"][0]["replay_command"].endswith(
        "--no-publish"
    )
    assert attack not in rendered
