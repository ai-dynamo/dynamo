# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.router.common import add_expected_osl
from benchmarks.router.real_data_priority_benchmark import tag_requests_with_priority

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


@pytest.mark.parametrize("preload_common", [False, True])
@pytest.mark.timeout(30)
def test_benchmark_import_preserves_unrelated_common_package(
    tmp_path, benchmark_env, preload_common
):
    common_dir = tmp_path / "common"
    common_dir.mkdir()
    for filename in ("__init__.py", "backend.py", "utils.py"):
        (common_dir / filename).write_text("")

    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import runpy
import sys
from pathlib import Path

repo_root = Path.cwd()
sys.path.insert(0, sys.argv[1])
if sys.argv[2] == "True":
    importlib.import_module("common")
runpy.run_path(str(repo_root / "benchmarks/router/tests/conftest.py"))
importlib.import_module("benchmarks.router.real_data_priority_benchmark")
for name in ("common.backend", "common.utils"):
    module = importlib.import_module(name)
    assert Path(module.__file__).parent == Path(sys.argv[1]) / "common"
""",
            str(tmp_path),
            str(preload_common),
        ],
        cwd=repo_root,
        env=benchmark_env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("as_module", [False, True])
@pytest.mark.timeout(30)
def test_priority_benchmark_cli_help(benchmark_env, as_module):
    repo_root = Path(__file__).resolve().parents[3]
    command = (
        ["-m", "benchmarks.router.real_data_priority_benchmark"]
        if as_module
        else [str(repo_root / "benchmarks/router/real_data_priority_benchmark.py")]
    )
    result = subprocess.run(
        [sys.executable, *command, "--help"],
        cwd=repo_root if as_module else repo_root / "benchmarks/router",
        env=benchmark_env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--priority-distribution" in result.stdout


def test_expected_osl_uses_mooncake_output_length_and_preserves_hints():
    """Prefer Mooncake output length without replacing existing request hints."""
    request = {
        "output_length": 32,
        "output_tokens": 64,
        "extra": {
            "metadata": "preserved",
            "nvext": {"agent_hints": {"priority": 7}},
        },
    }

    add_expected_osl(request)

    assert request["extra"] == {
        "metadata": "preserved",
        "nvext": {"agent_hints": {"priority": 7, "osl": 32}},
    }
    assert "nvext" not in request


def test_expected_osl_supports_legacy_output_tokens():
    """Use the legacy output token field when output length is absent."""
    request = {"output_tokens": 48}

    add_expected_osl(request)

    assert request["extra"]["nvext"]["agent_hints"]["osl"] == 48


def test_priority_tagging_does_not_mutate_source_request():
    """Add priority to a deep copy while preserving the source request."""
    request = {
        "extra": {
            "metadata": {"request_id": "request-1"},
            "nvext": {"agent_hints": {"osl": 32}},
        }
    }

    tagged_request = tag_requests_with_priority([request], priority=7)[0]

    assert request["extra"]["nvext"]["agent_hints"] == {"osl": 32}
    assert tagged_request["extra"] == {
        "metadata": {"request_id": "request-1"},
        "nvext": {"agent_hints": {"osl": 32, "priority": 7}},
    }
    assert tagged_request["extra"] is not request["extra"]
