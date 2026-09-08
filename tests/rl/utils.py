# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process helpers for reinforcement-learning tests."""

from __future__ import annotations

import os

import pytest
import requests

from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.payloads import check_models_api


def check_ready(response: requests.Response) -> bool:
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


def check_model_registered(response: requests.Response, *, model: str) -> bool:
    if not check_models_api(response):
        return False
    data = response.json()
    return any(item.get("id") == model for item in data.get("data", []))


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
    return build_gpu_mem_args("build_vllm_gpu_mem_args") or [
        "--gpu-memory-utilization",
        default_utilization,
    ]
