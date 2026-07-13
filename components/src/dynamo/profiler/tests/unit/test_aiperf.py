# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AIPerf command and failure diagnostics."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from dynamo.profiler.utils.aiperf import (
    _get_common_aiperf_cmd,
    benchmark_decode,
    benchmark_prefill,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_common_command_only_uses_supported_ignore_eos_input():
    command = _get_common_aiperf_cmd("artifacts")
    extra_inputs = [
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--extra-inputs"
    ]

    assert extra_inputs == ["ignore_eos:true"]


def _failed_process(stdout: str, stderr: str) -> MagicMock:
    process = MagicMock(returncode=1)
    process.communicate.return_value = (stdout, stderr)
    return process


def test_benchmark_prefill_logs_stdout_and_stderr(caplog):
    process = _failed_process("request rejected", "profile failed")

    with (
        patch("dynamo.profiler.utils.aiperf.subprocess.Popen", return_value=process),
        caplog.at_level(logging.ERROR, logger="dynamo.profiler.utils.aiperf"),
    ):
        result = benchmark_prefill(
            100,
            "artifacts",
            "test-model",
            "test-tokenizer",
        )

    assert result is None
    assert "stdout: request rejected" in caplog.text
    assert "stderr: profile failed" in caplog.text


def test_benchmark_decode_logs_stdout_and_stderr(caplog):
    warmup_process = MagicMock(returncode=0)
    warmup_process.communicate.return_value = ("", "")
    profile_process = _failed_process("request rejected", "profile failed")

    with (
        patch(
            "dynamo.profiler.utils.aiperf.subprocess.Popen",
            side_effect=[warmup_process, profile_process],
        ),
        caplog.at_level(logging.ERROR, logger="dynamo.profiler.utils.aiperf"),
    ):
        result = benchmark_decode(
            100,
            10,
            1,
            "artifacts",
            "test-model",
            "test-tokenizer",
        )

    assert result is None
    assert "stdout: request rejected" in caplog.text
    assert "stderr: profile failed" in caplog.text
