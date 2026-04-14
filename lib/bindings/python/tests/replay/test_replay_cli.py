# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

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
