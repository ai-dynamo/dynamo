# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math
from copy import deepcopy

import pytest
from dynamo_decision_perf.analysis import analyze_run, paired_differences
from dynamo_decision_perf.report import main, write_report

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def encoded(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


@pytest.fixture
def run(tmp_path):
    payload = {
        "model": "decision-test",
        "input": "Classify this message",
        "questions": [
            {
                "type": "choice",
                "name": "department",
                "instructions": "Department?",
                "choices": [{"value": "sales"}, {"value": "support"}],
            }
        ],
    }
    response = {
        "model": "decision-test",
        "answers": [
            {
                "type": "choice",
                "name": "department",
                "choice": "sales",
                "confidence": 0.6,
                "probabilities": [
                    {"value": "sales", "probability": 0.8},
                    {"value": "support", "probability": 0.2},
                ],
            }
        ],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 0,
            "total_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    metadata = {
        "run_id": "run-a",
        "series_id": "series",
        "profile_id": "profile",
        "cache_phase": "warm",
        "dialect": "oai",
        "aiperf_version": "0.13.0",
        "measurement_start_ns": 1_000_000_000,
        "measurement_end_ns": 11_000_000_000,
        "offered_requests": 1,
        "expected_requests": 1,
        "warmup_expected_requests": 0,
        "cost_boundary": "client to Dynamo HTTP, including rendering and orchestration",
    }
    workload = [
        {
            "logical_id": "item-0",
            "expected_questions": 1,
            "payload_sha256": {
                "oai": hashlib.sha256(encoded(payload).encode()).hexdigest()
            },
        }
    ]
    row_metadata = {
        "session_num": 0,
        "x_request_id": "request-0",
        "conversation_id": "item-0",
        "request_start_ns": 2_000_000_000,
        "request_end_ns": 2_010_000_000,
        "worker_id": "worker",
        "record_processor_id": "processor",
        "benchmark_phase": "profiling",
    }
    raw = {
        "metadata": row_metadata,
        "start_perf_ns": 100,
        "payload": payload,
        "status": 200,
        "responses": [{"perf_ns": 10_000_100, "text": encoded(response)}],
    }
    profile = {
        "metadata": row_metadata,
        "metrics": {"request_latency": {"value": 10, "unit": "ms"}},
    }
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    return raw_path, metadata, workload, [raw], [profile]


def audit(run):
    path, metadata, workload, raws, profiles = run
    for filename, rows in (
        ("profile_export_raw.jsonl", raws),
        ("profile_export.jsonl", profiles),
    ):
        (path / filename).write_text("".join(encoded(row) + "\n" for row in rows))
    (path / "profile_export_aiperf.json").write_text(
        encoded({"aiperf_version": "0.13.0", "is_complete": True})
    )
    manifest = path.parent / "manifest.json"
    manifest.write_text(encoded({"audit_metadata": metadata}))
    (path.parent / "benchmark_execution.json").write_text(
        encoded(
            {
                "exit_code": 0,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        )
    )
    return analyze_run(path, metadata, workload)


def test_valid_rates_usage_and_raw_preservation(run):
    result = audit(run)
    before = {p.name: p.read_bytes() for p in run[0].iterdir()}
    assert result["audit"]["status"] == "valid", result["audit"]
    assert result["summary"]["rates"]["valid_http_requests_per_second"] == 0.1
    assert result["summary"]["rates"]["completed_questions_per_second"] == 0.1
    assert result["summary"]["latency_ms"]["all_attempts"] == {
        "count": 1,
        "p50": 10.0,
        "max": 10.0,
        "p95": None,
        "p99": None,
    }
    assert result["summary"]["actual_work"] is None
    assert result["summary"]["api_usage"]["input_tokens"] == {
        "known_sum": 5,
        "unknown_requests": 0,
    }
    assert {p.name: p.read_bytes() for p in run[0].iterdir()} == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile_id", "different-profile"),
        ("series_id", "different-series"),
        ("cache_phase", "cold"),
        ("dialect", "systemone"),
        ("expected_requests", None),
        ("cost_boundary", "different-boundary"),
    ],
)
def test_audit_rejects_swapped_frozen_metadata(run, field, value):
    assert audit(run)["audit"]["status"] == "valid"
    path, metadata, workload, _, _ = run
    result = analyze_run(path, {**metadata, field: value}, workload)
    assert result["audit"]["status"] == "invalid"


def test_failures_remain_in_denominators_and_missing_offers_visible(run):
    _, metadata, _, raws, profiles = run
    metadata.update(offered_requests=5, expected_requests=6)
    for index, (status, error) in enumerate(
        (
            (429, {"message": "full", "code": 429}),
            (None, {"message": "deadline", "type": "TimeoutError"}),
            (500, {"message": "failed", "code": 500}),
        ),
        start=1,
    ):
        raw, profile = deepcopy(raws[0]), deepcopy(profiles[0])
        raw["metadata"]["x_request_id"] = profile["metadata"][
            "x_request_id"
        ] = f"request-{index}"
        raw.update(status=status, error=error, responses=[])
        profile["error"] = error
        raws.append(raw)
        profiles.append(profile)
    result = audit(run)
    counts = result["summary"]["counts"]
    assert counts["rejections"] == counts["timeouts"] == counts["http_errors"] == 1
    assert counts["unattempted_offers"] == counts["unoffered_expected"] == 1
    assert counts["incomplete_requests"] == 2
    assert result["summary"]["rates"]["offered_http_requests_per_second"] == 0.5
    assert result["summary"]["fractions"]["valid_of_offered"] == 0.2
    assert result["audit"]["status"] == "invalid"


@pytest.mark.parametrize(
    "defect",
    [
        "bad_body",
        "wrong_count",
        "wrong_payload",
        "duplicate",
        "missing_profile",
        "window",
        "nan_latency",
    ],
)
def test_integrity_defects_cannot_be_gain_evidence(run, defect):
    _, metadata, workload, raws, profiles = run
    if defect == "bad_body":
        raws[0]["responses"][0]["text"] = "{}"
    elif defect == "wrong_count":
        workload[0]["expected_questions"] = 2
    elif defect == "wrong_payload":
        raws[0]["payload"]["input"] = "changed"
    elif defect == "duplicate":
        raws.append(deepcopy(raws[0]))
    elif defect == "missing_profile":
        profiles.clear()
    elif defect == "window":
        metadata["measurement_end_ns"] = 2_005_000_000
    else:
        profiles[0]["metrics"]["request_latency"]["value"] = float("nan")
    result = audit(run)
    assert result["audit"]["status"] == "invalid"
    assert result["audit"]["gain_claims_allowed"] is False
    assert result["audit"]["blockers"]


def test_warmup_exclusion_and_error_latency_population(run):
    _, metadata, _, raws, profiles = run
    metadata["warmup_expected_requests"] = 1
    warm_raw, warm_profile = deepcopy(raws[0]), deepcopy(profiles[0])
    for row in (warm_raw, warm_profile):
        row["metadata"].update(
            x_request_id="warmup",
            benchmark_phase="warmup",
            request_start_ns=100,
            request_end_ns=200,
        )
    raws.insert(0, warm_raw)
    profiles.insert(0, warm_profile)
    result = audit(run)
    assert result["audit"]["status"] == "valid"
    assert result["summary"]["counts"]["warmup_excluded"] == 1
    assert result["summary"]["latency_ms"]["all_attempts"]["count"] == 1


@pytest.mark.parametrize(
    "count,p95,p99", [(99, False, False), (100, True, False), (1000, True, True)]
)
def test_percentile_minimum_sample_counts(run, count, p95, p99):
    _, metadata, _, raws, profiles = run
    metadata.update(offered_requests=count, expected_requests=count)
    for index in range(1, count):
        raw, profile = deepcopy(raws[0]), deepcopy(profiles[0])
        raw["metadata"]["x_request_id"] = profile["metadata"][
            "x_request_id"
        ] = f"request-{index}"
        raws.append(raw)
        profiles.append(profile)
    stats = audit(run)["summary"]["latency_ms"]["all_attempts"]
    assert (stats["p95"] is not None) is p95
    assert (stats["p99"] is not None) is p99


def test_missing_or_malformed_exports_produce_invalid_artifact(run):
    path, metadata, workload, _, _ = run
    result = analyze_run(path, metadata, workload)
    assert result["audit"]["status"] == "invalid"
    audit(run)
    (path / "profile_export_raw.jsonl").write_text("{broken\n")
    (path / "profile_export_aiperf.json").write_text("[]")
    result = analyze_run(path, metadata, workload)
    assert result["audit"]["status"] == "invalid"
    assert len(result["audit"]["blockers"]) >= 2


def test_controller_window_and_unknown_offered_count(run):
    path, metadata, _, _, _ = run
    metadata.pop("measurement_start_ns")
    metadata.pop("measurement_end_ns")
    metadata["offered_requests"] = None
    (path / "phase_manifest.json").write_text(
        encoded(
            {
                "phases": [
                    {"phase_kind": "warmup", "start_ns": 0, "end_ns": 100},
                    {
                        "phase_kind": "profiling",
                        "start_ns": 1_000_000_000,
                        "end_ns": 11_000_000_000,
                    },
                ]
            }
        )
    )
    result = audit(run)
    assert result["audit"]["status"] == "valid"
    assert result["summary"]["measurement_seconds"] == 10
    assert result["summary"]["rates"]["offered_http_requests_per_second"] is None
    assert result["summary"]["counts"]["unattempted_offers"] is None
    assert (
        result["summary"]["metadata"]["measurement_window_source"]
        == "aiperf.phase_manifest.controller_phase"
    )


def test_duration_run_and_missing_usage_remain_unknown(run):
    run[1].update(
        expected_requests=None, offered_requests=None, warmup_expected_requests=None
    )
    body = json.loads(run[3][0]["responses"][0]["text"])
    body.pop("usage")
    run[3][0]["responses"][0]["text"] = encoded(body)
    result = audit(run)
    assert result["audit"]["status"] == "valid", result["audit"]
    assert result["summary"]["counts"]["incomplete_requests"] is None
    for field in ("input_tokens", "output_tokens", "cached_tokens"):
        assert result["summary"]["api_usage"][field] == {
            "known_sum": None,
            "unknown_requests": 1,
        }


def test_refusal_is_valid_but_not_completed_question(run):
    body = json.loads(run[3][0]["responses"][0]["text"])
    body["answers"] = [{"type": "refusal", "name": "department"}]
    run[3][0]["responses"][0]["text"] = encoded(body)
    result = audit(run)
    assert result["audit"]["status"] == "valid"
    assert result["summary"]["counts"]["valid_http_requests"] == 1
    assert result["summary"]["counts"]["refusal_questions"] == 1
    assert result["summary"]["counts"]["completed_questions"] == 0


def test_error_latency_uses_measured_monotonic_delta_and_not_zero(run):
    raw, profile = run[3][0], run[4][0]
    raw.update(status=429, responses=[], error={"message": "full", "code": 429})
    profile.update(metrics={}, error={"message": "full", "code": 429})
    result = audit(run)
    assert result["audit"]["status"] == "valid"
    assert result["requests"][0]["latency_ms"] == 10
    assert result["requests"][0]["latency_source"] == "metadata.monotonic_delta"
    assert result["summary"]["api_usage"]["input_tokens"]["unknown_requests"] == 1
    raw["metadata"]["request_end_ns"] = profile["metadata"][
        "request_end_ns"
    ] = 2_000_000_000
    assert audit(run)["requests"][0]["latency_ms"] is None


def test_parser_marked_malformed_200_is_schema_error_not_success(run):
    run[3][0]["responses"][0]["text"] = "{}"
    run[4][0]["error"] = {"message": "bad decision", "type": "DecisionContractError"}
    result = audit(run)
    assert result["summary"]["counts"]["schema_errors"] == 1
    assert result["audit"]["status"] == "invalid"


def test_processor_failure_does_not_become_tiny_transport_latency(run):
    raw, profile = run[3][0], run[4][0]
    raw.pop("status")
    raw.pop("payload")
    raw["responses"] = []
    raw["error"] = profile["error"] = {
        "type": "DecisionContractError",
        "message": "schema validation failed",
    }
    profile["metrics"] = {}
    for row in (raw, profile):
        row["metadata"]["request_end_ns"] = row["metadata"]["request_start_ns"] + 100
    result = audit(run)
    assert result["audit"]["status"] == "invalid"
    assert result["summary"]["counts"]["schema_errors"] == 1
    assert result["requests"][0]["latency_ms"] is None


def test_missing_wire_payload_and_missing_execution_fail_closed(run):
    run[3][0]["payload"] = None
    result = audit(run)
    assert result["audit"]["status"] == "invalid"
    assert result["summary"]["counts"]["schema_errors"] == 1
    (run[0].parent / "benchmark_execution.json").unlink()
    result = analyze_run(run[0], run[1], run[2])
    assert any(
        "benchmark_execution.json" in issue for issue in result["audit"]["blockers"]
    )


def test_aggregate_controller_window_requires_explicit_timezone(run):
    path, metadata, workload, _, _ = run
    audit(run)
    metadata.pop("measurement_start_ns")
    metadata.pop("measurement_end_ns")
    (path / "profile_export_aiperf.json").write_text(
        encoded(
            {
                "aiperf_version": "0.13.0",
                "is_complete": True,
                "start_time": "1970-01-01T00:00:01",
                "end_time": "1970-01-01T00:00:11",
            }
        )
    )
    assert analyze_run(path, metadata, workload)["audit"]["status"] == "invalid"
    metadata["export_timezone_offset_seconds"] = 0
    result = analyze_run(path, metadata, workload)
    assert result["audit"]["status"] == "valid", result["audit"]
    assert result["summary"]["measurement_seconds"] == pytest.approx(10.000002)


def test_native_sse_terminal_and_request_salt_correlation(run):
    _, metadata, workload, raws, _ = run
    metadata.update(
        dialect="native_score",
        cost_boundary="native HTTP including serialization; excludes client tokenization",
    )
    payload = {
        "model": "decision-test",
        "input_ids": [1, 2],
        "token_ids_logprob": [3, 4],
        "sampling_params": {"max_new_tokens": 0},
        "stream": True,
        "return_logprob": True,
    }
    digest = hashlib.sha256(encoded(payload).encode()).hexdigest()
    workload[0]["payload_sha256"] = {"native_score": digest}
    raws[0]["payload"] = {
        **payload,
        "cache_salt": "decision-native-" + hashlib.sha256(b"request-0").hexdigest(),
    }
    raws[0]["request_headers"] = {"X-Decision-Payload-SHA256": digest}
    body = {
        "output_ids": [],
        "meta_info": {
            "finish_reason": {"type": "length", "length": 0},
            "prompt_tokens": 2,
            "completion_tokens": 0,
            "output_token_ids_logprobs": [
                [[math.log(0.8), 3, None], [math.log(0.1), 4, None]]
            ],
        },
    }
    raws[0]["responses"] = [
        {"perf_ns": 10_000_100, "packets": [{"name": "data", "value": encoded(body)}]},
        {"perf_ns": 10_000_101, "packets": [{"name": "data", "value": "[DONE]"}]},
    ]
    result = audit(run)
    assert result["audit"]["status"] == "valid", result["audit"]
    assert result["summary"]["api_usage"]["cached_tokens"] == {
        "known_sum": None,
        "unknown_requests": 1,
    }
    raws[0]["responses"].insert(0, deepcopy(raws[0]["responses"][0]))
    assert audit(run)["audit"]["status"] == "invalid"
    raws[0]["responses"].pop(0)
    raws[0]["payload"]["cache_salt"] = "reused"
    assert audit(run)["audit"]["status"] == "invalid"
    raws[0]["payload"]["cache_salt"] = (
        "decision-native-" + hashlib.sha256(b"request-0").hexdigest()
    )
    raws[0]["responses"].insert(
        0,
        {
            "perf_ns": 100,
            "packets": [{"name": "data", "value": encoded({"error": "failed"})}],
        },
    )
    assert audit(run)["audit"]["status"] == "invalid"


def test_paired_differences_require_matching_identity_and_valid_runs(run):
    left = audit(run)
    right = deepcopy(left)
    right["summary"]["metadata"]["run_id"] = "run-b"
    right["requests"][0]["latency_ms"] = 13.0
    result = paired_differences(left, right)
    assert result["status"] == "comparable"
    assert result["latency_difference_ms"]["p50"] == 3.0
    right["summary"]["metadata"]["cache_phase"] = "cold"
    assert paired_differences(left, right)["status"] == "not_comparable"
    right = deepcopy(left)
    right["audit"]["status"] = "invalid"
    assert paired_differences(left, right)["status"] == "not_comparable"
    right = deepcopy(left)
    right["requests"].append(deepcopy(right["requests"][0]))
    assert paired_differences(left, right)["matched_requests"] == 0


def test_paired_quantile_is_not_a_difference_of_quantiles(run):
    left = audit(run)
    right = deepcopy(left)
    left["requests"] = [
        {**left["requests"][0], "logical_id": str(index), "latency_ms": latency}
        for index, latency in enumerate((1, 50, 100))
    ]
    right["requests"] = [
        {**right["requests"][0], "logical_id": str(index), "latency_ms": latency}
        for index, latency in enumerate((30, 40, 200))
    ]
    comparison = paired_differences(left, right)
    assert comparison["matched_requests"] == 3
    assert comparison["latency_difference_ms"]["p50"] == 29
    assert comparison["latency_difference_ms"]["p50"] != 40 - 50
    right["requests"][0]["workload_signature"] = "different workload"
    assert paired_differences(left, right)["matched_requests"] == 2


def test_report_is_reproducible_escaped_and_never_overwrites_raw(run, tmp_path):
    result = audit(run)
    result["summary"]["metadata"]["run_id"] = "<unsafe>"
    output = tmp_path / "report"
    paths = write_report([result], output)
    before = {p.name: p.read_bytes() for p in paths}
    write_report([result], output)
    assert {p.name: p.read_bytes() for p in paths} == before
    assert "&lt;unsafe&gt;" in (output / "latency.svg").read_text()
    assert "not an accuracy" in (output / "report.md").read_text()
    assert "TTFT" not in (output / "report.json").read_text()
    with pytest.raises(ValueError, match="raw"):
        write_report([result], run[0])


def test_report_cli_keeps_invalid_run_visible(run, tmp_path, capsys):
    audit(run)
    metadata_path, workload_path = (
        tmp_path / "metadata.json",
        tmp_path / "workload.json",
    )
    metadata_path.write_text(encoded(run[1]))
    workload_path.write_text(encoded(run[2]))
    args = [
        "--artifacts",
        str(run[0]),
        "--metadata",
        str(metadata_path),
        "--workload",
        str(workload_path),
        "--output",
        str(tmp_path / "report"),
    ]
    assert main(args) == 0
    assert "Wrote" in capsys.readouterr().out
    (run[0] / "profile_export_raw.jsonl").write_text("{}\n")
    assert main(args) == 2
    assert (
        json.loads((tmp_path / "report/report.json").read_text())["runs"][0]["audit"][
            "status"
        ]
        == "invalid"
    )
