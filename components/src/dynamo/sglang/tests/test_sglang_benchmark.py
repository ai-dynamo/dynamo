# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang self-benchmark adapter wiring."""

import sys

import pytest

from dynamo.sglang.args import parse_args
from dynamo.sglang.tests.conftest import make_cli_args_fixture

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.filterwarnings("ignore:.*torch.jit.script_method.*:DeprecationWarning"),
]

mock_sglang_cli = make_cli_args_fixture("dynamo.sglang")


@pytest.mark.asyncio
async def test_benchmark_mode_enabled_from_env(monkeypatch, mock_sglang_cli):
    """Dynamo should map benchmark env vars onto native SGLang args."""
    monkeypatch.setenv("DYN_BENCHMARK_MODE", "prefill")
    monkeypatch.setenv("DYN_BENCHMARK_PREFILL_GRANULARITY", "32")
    monkeypatch.setenv("DYN_BENCHMARK_TIMEOUT", "17")
    mock_sglang_cli("--model", "Qwen/Qwen3-0.6B")

    config = await parse_args(sys.argv[1:])
    assert config.server_args.benchmark_mode == "prefill"
    assert config.server_args.enable_forward_pass_metrics is True
    assert config.server_args.benchmark_prefill_granularity == 32
    assert config.server_args.benchmark_timeout == 17


@pytest.mark.asyncio
async def test_benchmark_cli_overrides_env(monkeypatch, mock_sglang_cli):
    """Explicit SGLang benchmark CLI flags should win over Dynamo env defaults."""
    monkeypatch.setenv("DYN_BENCHMARK_MODE", "prefill")
    monkeypatch.setenv("DYN_BENCHMARK_TIMEOUT", "17")
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--benchmark-mode",
        "decode",
        "--benchmark-timeout",
        "23",
    )

    config = await parse_args(sys.argv[1:])
    assert config.server_args.benchmark_mode == "decode"
    assert config.server_args.benchmark_timeout == 23
