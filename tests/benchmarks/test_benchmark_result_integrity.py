# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark wrappers must not turn failed or incomplete exports into success."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.benchmarks.fixtures.aiperf_client import write_artifacts

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmarks/frontend/scripts"
sys.path.insert(0, str(SCRIPTS))

import sweep_runner  # noqa: E402 -- standalone scripts need SCRIPTS on sys.path
from sweep_core.aiperf import (  # noqa: E402 -- standalone scripts need SCRIPTS on sys.path
    parse_aiperf_into_result,
)
from sweep_core.models import (  # noqa: E402 -- standalone scripts need SCRIPTS on sys.path
    AiperfDimension,
    DeployDimension,
    RunResult,
    RunSpec,
)

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


@pytest.mark.timeout(30)
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
@pytest.mark.parametrize("models", [1, 2])
@pytest.mark.unit
def test_frontend_benchmark_rejects_failed_exports(tmp_path, mode, models, fake_aiperf):
    script = (SCRIPTS / "run_perf.sh").read_text()
    loop = _shell_region(script, "benchmark-loop")

    prelude = """
set -euo pipefail
BENCHMARK_STATUS=0
_AIPERF_MODELS=(test-model)
_AIPERF_TOK_ARGS=()
_LOAD_ARGS=(--request-count 3)
_EFFECTIVE_REQUESTS=3
_WARMUP_ARGS=()
MODEL=test-model
ISL=16
OSL=2
CONCURRENCY=1
FRONTEND_PORT=unused
BENCHMARK_DURATION=
"""
    if models == 2:
        prelude = prelude.replace(
            "_AIPERF_MODELS=(test-model)", "_AIPERF_MODELS=(test-model second)"
        )
    result = subprocess.run(
        [
            "bash",
            "-c",
            prelude + loop + '\nexit "$BENCHMARK_STATUS"',
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "FAKE_AIPERF_MODE": mode,
            "OUTPUT_DIR": str(tmp_path / "results"),
            "SCRIPT_DIR": str(SCRIPTS),
            "MODEL_CALLS": str(tmp_path / "model-calls"),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == (mode in {"healthy", "mixed-errors"}), (
        result.stdout + result.stderr
    )
    assert (tmp_path / "model-calls").read_text().splitlines() == (
        ["test-model", "second"] if models == 2 else ["test-model"]
    )


@pytest.mark.parametrize(
    "mode",
    [
        "healthy",
        "mixed-errors",
        "missing",
        "invalid",
        "count",
        "csv-missing",
        "csv-invalid",
        "csv-truncated",
        "csv-count-truncated",
        "csv-empty-count",
    ],
)
@pytest.mark.unit
def test_sweep_validates_required_exports(tmp_path, mode):
    deploy = DeployDimension("mocker", "hf")
    spec = RunSpec(
        deploy, AiperfDimension(1, 16, num_requests=3), deploy.deploy_key, "test"
    )
    directory = tmp_path / "aiperf"
    directory.mkdir()
    write_artifacts(directory, 3, mode)
    result = RunResult(spec, status="ok")
    parse_aiperf_into_result(result, tmp_path, expected=3)
    assert result.status == ("ok" if mode in {"healthy", "mixed-errors"} else "fail")


@pytest.mark.parametrize("status", ["ok", "fail", "skipped"])
@pytest.mark.unit
def test_sweep_cli_propagates_failed_results(monkeypatch, status):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sweep_runner", "--tokenizers", "hf", "--concurrency", "1", "--isl", "16"],
    )
    monkeypatch.setattr(
        sweep_runner,
        "run_sweep",
        lambda plan, executor: [SimpleNamespace(status=status)] * plan.total_runs,
    )
    if status == "fail":
        with pytest.raises(SystemExit) as error:
            sweep_runner.main()
        assert error.value.code == 1
    else:
        sweep_runner.main()


@pytest.mark.unit
def test_multi_model_discovery_ignores_phase_exports(tmp_path):
    deploy = DeployDimension("mocker", "hf", num_models=2)
    spec = RunSpec(
        deploy, AiperfDimension(1, 16, num_requests=3), deploy.deploy_key, "test"
    )
    for model in ["model-0", "model-1"]:
        for suffix, count in [("", 3), ("/phases/warmup", 1)]:
            directory = tmp_path / "aiperf" / (model + suffix)
            directory.mkdir(parents=True)
            write_artifacts(directory, count, "healthy")
    result = RunResult(spec, status="ok")
    parse_aiperf_into_result(result, tmp_path, expected=3)
    assert result.status == "ok"
