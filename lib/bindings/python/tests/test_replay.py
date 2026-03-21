# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

import pytest

from dynamo.llm import run_mocker_trace_replay
from dynamo.replay import run_trace_replay

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
]


def _write_trace_and_args(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    args_path = tmp_path / "args.json"
    records = [
        {
            "timestamp": 1000.0,
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [101],
        },
        {
            "timestamp": 1005.0,
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [101],
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    args_path.write_text(
        json.dumps(
            {
                "block_size": 64,
                "speedup_ratio": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    return trace_path, args_path


def test_run_trace_replay_offline_smoke(tmp_path):
    trace_path, args_path = _write_trace_and_args(tmp_path)

    report = run_trace_replay(
        trace_path,
        extra_engine_args=args_path,
        num_workers=1,
        replay_mode="offline",
    )

    assert report["num_requests"] == 2
    assert report["completed_requests"] == 2
    assert report["total_output_tokens"] == 4


def test_run_trace_replay_online_smoke(tmp_path):
    trace_path, args_path = _write_trace_and_args(tmp_path)

    report = run_trace_replay(
        trace_path,
        extra_engine_args=args_path,
        num_workers=1,
        replay_mode="online",
    )

    assert report["num_requests"] == 2
    assert report["completed_requests"] == 2
    assert report["total_output_tokens"] == 4


def test_run_mocker_trace_replay_compatibility_wrapper(tmp_path):
    trace_path, args_path = _write_trace_and_args(tmp_path)

    report = run_mocker_trace_replay(
        trace_path,
        extra_engine_args=args_path,
        num_workers=1,
    )

    assert report["num_requests"] == 2
    assert report["completed_requests"] == 2


def test_dynamo_replay_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "dynamo.replay", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "replay-mode" in result.stdout
