# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def _replay_cli_env() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[6]
    env = os.environ.copy()
    pythonpath_entries = [
        str(repo_root / "lib/bindings/python/src"),
        str(repo_root / "components/src"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    return env


def _assert_basic_report_counts(report, *, num_requests, input_tokens, output_tokens):
    assert report["num_requests"] == num_requests
    assert report["completed_requests"] == num_requests
    assert report["total_input_tokens"] == num_requests * input_tokens
    assert report["total_output_tokens"] == num_requests * output_tokens


def _assert_basic_report_metrics(report):
    assert report["request_throughput_rps"] > 0
    assert report["output_throughput_tok_s"] > 0
    assert report["duration_ms"] > 0


def _assert_replay_cli_outputs(completed, report_path):
    assert "NVIDIA AIPerf | LLM Metrics" in completed.stdout
    assert "Saved full report to:" in completed.stdout
    assert '"completed_requests"' not in completed.stdout
    return json.loads(report_path.read_text(encoding="utf-8"))


def _run_replay_cli(tmp_path, *args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamo.replay",
            *args,
        ],
        capture_output=True,
        check=True,
        cwd=str(tmp_path),
        env=_replay_cli_env(),
        text=True,
    )


def _write_profile_results_dir(tmp_path: Path) -> Path:
    profile_results_dir = tmp_path / "planner_profile_results"
    prefill_dir = profile_results_dir / "selected_prefill_interpolation"
    decode_dir = profile_results_dir / "selected_decode_interpolation"
    prefill_dir.mkdir(parents=True)
    decode_dir.mkdir(parents=True)

    np.savez(
        prefill_dir / "raw_data.npz",
        prefill_isl=np.array([128.0, 256.0, 512.0, 1024.0]),
        prefill_ttft=np.array([4.0, 6.0, 9.0, 14.0]),
        prefill_thpt_per_gpu=np.array([2200.0, 1800.0, 1300.0, 900.0]),
    )

    np.savez(
        decode_dir / "raw_data.npz",
        x_kv_usage=np.array([0.10, 0.10, 0.10, 0.50, 0.50, 0.50, 0.90, 0.90, 0.90]),
        y_context_length=np.array(
            [128.0, 256.0, 384.0, 128.0, 256.0, 384.0, 128.0, 256.0, 384.0]
        ),
        z_itl=np.array([1.0, 1.3, 1.6, 1.2, 1.5, 1.8, 1.4, 1.7, 2.0]),
        z_thpt_per_gpu=np.array(
            [1400.0, 1300.0, 1200.0, 1200.0, 1100.0, 1000.0, 1000.0, 900.0, 800.0]
        ),
        max_kv_tokens=np.array([4096.0]),
    )

    return profile_results_dir


@pytest.mark.timeout(30)
def test_replay_cli_subprocess_synthetic_smoke_accepts_profile_results_dir(tmp_path):
    report_path = tmp_path / "synthetic_report_dir.json"
    planner_profile_data = _write_profile_results_dir(tmp_path)

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
