# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


@pytest.fixture
def command_stub(tmp_path):
    script = tmp_path / "command_stub.py"
    script.write_text(
        "import json, os, pathlib, sys\n"
        "state = pathlib.Path(os.environ['CALL_STATE'])\n"
        "calls = json.loads(state.read_text()) if state.exists() else []\n"
        "calls.append({\n"
        "    'pip_index_url': os.environ.get('PIP_INDEX_URL'),\n"
        "    'uv_default_index': os.environ.get('UV_DEFAULT_INDEX'),\n"
        "})\n"
        "state.write_text(json.dumps(calls))\n"
        "attempt = len(calls)\n"
        "print(os.environ[f'OUTPUT_{attempt}'])\n"
        "sys.exit(int(os.environ[f'EXIT_{attempt}']))\n"
    )
    return [sys.executable, str(script)]


def run_wrapper(
    tmp_path,
    command_stub,
    *,
    output,
    first_exit="23",
    second_exit="0",
    index_url=(
        "https://artifactory.nvidia.com/artifactory/api/pypi/" "pypi-remote/simple/"
    ),
):
    wrapper = (
        Path(__file__).resolve().parents[1]
        / ".github/scripts/run_with_pypi_fallback.py"
    )
    state = tmp_path / "calls.json"
    log = tmp_path / "build.log"
    env = {
        **os.environ,
        "CALL_STATE": str(state),
        "EXIT_1": first_exit,
        "EXIT_2": second_exit,
        "OUTPUT_1": output,
        "OUTPUT_2": "fallback attempt",
        "PIP_INDEX_URL": index_url,
        "UV_DEFAULT_INDEX": index_url,
    }
    result = subprocess.run(
        [sys.executable, str(wrapper), str(log), *command_stub],
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )
    return result, json.loads(state.read_text()), log.read_text()


def test_cloudfront_403_retries_with_public_pypi(tmp_path, command_stub):
    result, calls, log = run_wrapper(
        tmp_path,
        command_stub,
        output=(
            "https://d1j32scj9xxftt.cloudfront.net/torch.whl: " "403 Request blocked"
        ),
    )

    assert result.returncode == 0
    assert [call["pip_index_url"] for call in calls] == [
        "https://artifactory.nvidia.com/artifactory/api/pypi/pypi-remote/simple/",
        "https://pypi.org/simple/",
    ]
    assert calls[1]["uv_default_index"] == "https://pypi.org/simple/"
    assert "::warning title=PyPI fallback::" in result.stdout
    assert "fallback attempt" in log
    assert (
        "403 Request blocked" in (tmp_path / "build.log.artifactory-failed").read_text()
    )


def test_success_does_not_retry(tmp_path, command_stub):
    result, calls, _ = run_wrapper(
        tmp_path,
        command_stub,
        output="primary attempt",
        first_exit="0",
    )

    assert result.returncode == 0
    assert len(calls) == 1
    assert "PyPI fallback" not in result.stdout


@pytest.mark.parametrize(
    "output",
    [
        pytest.param(
            "https://artifactory.nvidia.com/artifactory/api/pypi/pypi-remote: 403",
            id="artifactory-auth-failure",
        ),
        pytest.param("dependency resolution failed", id="unrelated-failure"),
    ],
)
def test_non_retryable_failure_preserves_exit_code(tmp_path, command_stub, output):
    result, calls, _ = run_wrapper(tmp_path, command_stub, output=output)

    assert result.returncode == 23
    assert len(calls) == 1
    assert "PyPI fallback" not in result.stdout


def test_public_pypi_primary_does_not_retry(tmp_path, command_stub):
    result, calls, _ = run_wrapper(
        tmp_path,
        command_stub,
        output=(
            "https://d1j32scj9xxftt.cloudfront.net/torch.whl: " "403 Request blocked"
        ),
        index_url="https://pypi.org/simple/",
    )

    assert result.returncode == 23
    assert len(calls) == 1
    assert "PyPI fallback" not in result.stdout


def test_fallback_failure_is_propagated(tmp_path, command_stub):
    result, calls, _ = run_wrapper(
        tmp_path,
        command_stub,
        output=(
            "https://d1j32scj9xxftt.cloudfront.net/torch.whl: " "403 Request blocked"
        ),
        second_exit="42",
    )

    assert result.returncode == 42
    assert len(calls) == 2
