# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for trace command arguments (NVBug 6414352)."""

import sys

import pytest

from benchmarks.router import common

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


@pytest.fixture
def command_args(tmp_path):
    return {
        "model": "test-model",
        "tokenizer": "test-tokenizer",
        "input_dataset": str(tmp_path / "input trace.jsonl"),
        "artifact_dir": str(tmp_path / "aiperf artifacts"),
        "seed": 17,
        "url": "http://router.test",
    }


@pytest.fixture
def agent_command_builder(monkeypatch):
    # The standalone script imports common by its unqualified module name.
    # Import inside the fixture so the temporary alias exists first and is
    # restored at teardown without changing imports for unrelated scripts.
    monkeypatch.setitem(sys.modules, "common", common)
    from benchmarks.router import agent_benchmark

    return agent_benchmark.get_aiperf_cmd


def assert_trace_command(cmd, args):
    assert cmd[:2] == ["aiperf", "profile"]
    flags = {token.split("=", 1)[0] for token in cmd if token.startswith("--")}
    assert "--prompt-input-tokens-block-size" not in flags
    assert "--isl-block-size" not in flags
    assert not any(flag.startswith("--synthetic-input-tokens-") for flag in flags)

    expected_values = {
        "--model": args["model"],
        "--tokenizer": args["tokenizer"],
        "--url": args["url"],
        "--input-file": args["input_dataset"],
        "--custom-dataset-type": "mooncake_trace",
        "--random-seed": str(args["seed"]),
        "--artifact-dir": args["artifact_dir"],
    }
    for flag, value in expected_values.items():
        assert cmd.count(flag) == 1
        index = cmd.index(flag)
        assert cmd[index : index + 2] == [flag, value]

    # The block size is unused by the other controlled inputs.
    assert str(args["block_size"]) not in cmd
    assert "None" not in cmd
    common_flags = common.get_common_aiperf_flags()
    assert cmd[-len(common_flags) :] == common_flags


def test_fixed_schedule_trace_command_omits_block_size(command_args):
    args = {**command_args, "block_size": 512}
    cmd = common.get_aiperf_cmd_for_trace(
        args["model"],
        args["tokenizer"],
        args["input_dataset"],
        args["artifact_dir"],
        args["seed"],
        args["block_size"],
        args["url"],
    )

    assert_trace_command(cmd, args)
    assert cmd.count("--fixed-schedule") == 1
    assert cmd.count("--fixed-schedule-auto-offset") == 1


def test_agent_trace_command_omits_block_size(command_args, agent_command_builder):
    args = {**command_args, "block_size": 64, "concurrency": 3, "request_count": 23}
    cmd = agent_command_builder(
        args["model"],
        args["tokenizer"],
        args["input_dataset"],
        args["artifact_dir"],
        args["seed"],
        args["concurrency"],
        args["block_size"],
        args["request_count"],
        args["url"],
    )

    assert_trace_command(cmd, args)
    for flag, value in (
        ("--concurrency", str(args["concurrency"])),
        ("--request-count", str(args["request_count"])),
    ):
        assert cmd.count(flag) == 1
        index = cmd.index(flag)
        assert cmd[index : index + 2] == [flag, value]
