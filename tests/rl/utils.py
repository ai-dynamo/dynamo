# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process helpers for reinforcement-learning tests."""

from __future__ import annotations

import os

import pytest

from tests.utils.gpu_args import build_vllm_gpu_mem_args
from tests.utils.http_checks import check_model_registered as check_model_registered


def process_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["DYN_LOG"] = "debug"
    env["DYN_NAMESPACE"] = "dynamo"
    env.update(extra)
    return env


def prepare_log_dir(request: pytest.FixtureRequest, suffix: str) -> str:
    # Use pytest's per-test temp dir instead of a repo-relative path so process logs
    # never land in the repo tree and never collide across parallel runs.
    tmp_path = request.getfixturevalue("tmp_path")
    log_dir = tmp_path / suffix
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def vllm_gpu_mem_args(default_utilization: str = "0.4") -> list[str]:
    # Honor the GPU scheduler's per-worker KV-cache budget under bin-packing;
    # fall back to a conservative utilization for serial runs.
    return build_vllm_gpu_mem_args(default_utilization)
