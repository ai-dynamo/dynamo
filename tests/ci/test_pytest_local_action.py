# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior checks for the local pytest composite action."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_PATH = REPO_ROOT / ".github/actions/pytest-local/action.yml"


def _run_pytest_step() -> dict:
    action = yaml.safe_load(ACTION_PATH.read_text())
    return next(
        step for step in action["runs"]["steps"] if step["name"] == "Run pytest"
    )


def _parallel_cpu_budget_script() -> str:
    script = _run_pytest_step()["run"]
    return script.split("# BEGIN parallel CPU budget", 1)[1].split(
        "# END parallel CPU budget", 1
    )[0]


def _resolved_cpu_budget(*, suite: str, max_vram_gib: str, cpu_limit: str) -> str:
    env = os.environ.copy()
    env.update(
        DYNAMO_TEST_SUITE_NAME=suite,
        PYTEST_MAX_VRAM_GIB=max_vram_gib,
        NUM_CPUS=cpu_limit,
    )
    result = subprocess.run(
        ["bash", "-c", f'{_parallel_cpu_budget_script()}\nprintf "%s" "$NUM_CPUS"'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.rsplit("\n", 1)[-1]


def test_cpu_limit_is_exported_to_gpu_parallel_scheduler() -> None:
    assert _run_pytest_step()["env"]["NUM_CPUS"] == "${{ inputs.cpu_limit }}"


def test_trtllm_gpu_parallel_stage_is_capped_at_four_slots() -> None:
    assert (
        _resolved_cpu_budget(suite="trtllm", max_vram_gib="80", cpu_limit="10") == "4"
    )


def test_non_trtllm_stage_keeps_generic_cpu_limit() -> None:
    assert _resolved_cpu_budget(suite="vllm", max_vram_gib="80", cpu_limit="10") == "10"


def test_trtllm_sequential_stage_keeps_generic_cpu_limit() -> None:
    assert _resolved_cpu_budget(suite="trtllm", max_vram_gib="", cpu_limit="10") == "10"


def test_trtllm_cap_never_raises_a_lower_cpu_limit() -> None:
    assert _resolved_cpu_budget(suite="trtllm", max_vram_gib="80", cpu_limit="2") == "2"
