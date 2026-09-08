# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tests.gpu_memory_service.common.runtime import (
    _replace_cli_option,
    _sglang_cuda_graph_args,
    _vllm_cuda_graph_args,
)


def test_replace_cli_option_preserves_profile_byte_cap():
    args = [
        "--kv-cache-memory-bytes",
        "67108864",
        "--gpu-memory-utilization",
        "0.01",
    ]

    assert _replace_cli_option(args, "--gpu-memory-utilization", "0.22") == [
        "--kv-cache-memory-bytes",
        "67108864",
        "--gpu-memory-utilization",
        "0.22",
    ]
    assert args[-1] == "0.01"


def test_sglang_cuda_graph_args_default_to_cheap_local_mode(monkeypatch):
    monkeypatch.delenv("GMS_TEST_ENABLE_CUDA_GRAPHS", raising=False)
    assert _sglang_cuda_graph_args() == [
        "--disable-piecewise-cuda-graph",
        "--disable-cuda-graph",
    ]


def test_sglang_cuda_graph_args_match_production_when_enabled(monkeypatch):
    monkeypatch.setenv("GMS_TEST_ENABLE_CUDA_GRAPHS", "1")
    assert _sglang_cuda_graph_args() == ["--disable-piecewise-cuda-graph"]


def test_vllm_cuda_graph_args_are_opt_in(monkeypatch):
    monkeypatch.delenv("GMS_TEST_ENABLE_CUDA_GRAPHS", raising=False)
    assert _vllm_cuda_graph_args() == ["--enforce-eager"]
    monkeypatch.setenv("GMS_TEST_ENABLE_CUDA_GRAPHS", "true")
    assert _vllm_cuda_graph_args() == []
