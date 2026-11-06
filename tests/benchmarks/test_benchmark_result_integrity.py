# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark wrappers must not turn failed or incomplete exports into success."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture
def fake_aiperf(tmp_path):
    executable = tmp_path / "aiperf"
    source = ROOT / "tests/benchmarks/fixtures/aiperf_client.py"
    executable.write_text(f"#!{sys.executable}\n" + source.read_text())
    executable.chmod(0o755)
    return executable


def _shell_region(script: str, name: str) -> str:
    """Extract an explicitly named shell test seam with actionable diagnostics."""
    lines = script.splitlines(keepends=True)
    begins = [i for i, line in enumerate(lines) if line.strip() == f"# BEGIN {name}"]
    ends = [i for i, line in enumerate(lines) if line.strip() == f"# END {name}"]
    assert len(begins) == len(ends) == 1, f"Expected one BEGIN/END region for {name!r}"
    assert begins[0] < ends[0], f"Reversed shell region {name!r}"
    return "".join(lines[begins[0] + 1 : ends[0]])


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "recipe",
    [
        "recipes/gpt-oss-120b/trtllm/agg/perf.yaml",
        "recipes/qwen3-vl-32b-fp8/benchmark/perf.yaml",
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "healthy",
        "mixed-errors",
        "exit",
        "missing",
        "invalid",
        "count",
        "csv-missing",
        "csv-invalid",
        "csv-truncated",
        "csv-empty-count",
    ],
)
@pytest.mark.unit
def test_recipe_rejects_failed_or_incomplete_exports(
    tmp_path, recipe, mode, fake_aiperf
):
    container = yaml.safe_load((ROOT / recipe).read_text())["spec"]["template"]["spec"][
        "containers"
    ][0]
    shell, _, script = container["command"]
    function = _shell_region(script, "benchmark-function")
    prelude = "\n".join(line for line in script.splitlines() if line.startswith("set "))

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "FAKE_AIPERF_MODE": mode,
        "ROOT_ARTIFACT_DIR": str(tmp_path / "results"),
        "EPOCH": "test-model",
        "JOB_NAME": "test-model",
        "TARGET_MODEL": "test-model",
        "TOKENIZER_PATH": "test-model",
        "ENDPOINT": "unused",
        "PYTHONUNBUFFERED": "1",
    }
    result = subprocess.run(
        [
            shell,
            "-c",
            f"{prelude}\n{function}\nrun_perf 1 1 3\necho complete",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == (mode in {"healthy", "mixed-errors"}), (
        result.stdout + result.stderr
    )
