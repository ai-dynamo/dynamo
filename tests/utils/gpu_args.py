# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for tests that launch backend processes directly from Python."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path


def _call_gpu_utils_function(
    function_name: str, env: Mapping[str, str] | None = None
) -> str:
    """Call a backend memory-args function from examples/common/gpu_utils.sh."""
    repo_root = Path(__file__).resolve().parents[2]
    gpu_utils = repo_root / "examples" / "common" / "gpu_utils.sh"
    cmd = f"source {shlex.quote(str(gpu_utils))}; {function_name}"
    result = subprocess.run(
        ["bash", "-lc", cmd],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return result.stdout.strip()


def build_gpu_mem_args(
    function_name: str, env: Mapping[str, str] | None = None
) -> list[str]:
    """Call a backend memory-args function that returns shell-style CLI args."""
    return shlex.split(_call_gpu_utils_function(function_name, env=env))


def build_trtllm_override_args(env: Mapping[str, str] | None = None) -> list[str]:
    """Return TRT-LLM override CLI args from GPU parallel scheduler env vars."""
    override_json = _call_gpu_utils_function("build_trtllm_override_args_with_mem", env)
    if not override_json:
        return []
    return ["--override-engine-args", override_json]
