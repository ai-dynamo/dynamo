# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@pytest.mark.parametrize(
    "mode", ["backend", "explicit_path", "package", "integration", "mocker"]
)
@pytest.mark.parametrize("missing", ["dynamo_decision_perf", "aiperf"])
def test_optional_package_collection_boundary(tmp_path, mode, missing):
    package = tmp_path / "benchmarks" / "decision_api"
    tests = package / "tests"
    tests.mkdir(parents=True)
    guard = Path(__file__).with_name("conftest.py")
    shutil.copyfile(guard, tests / "conftest.py")
    (tests / "test_optional.py").write_text(
        f'raise ModuleNotFoundError("No module named {missing!r}")\n'
    )
    (tmp_path / "test_backend.py").write_text("def test_backend():\n    pass\n")
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    args = ["-c", str(config), "--confcutdir", str(tmp_path), "-q"]
    cwd = package if mode == "package" else tmp_path
    args += [str(tests)] if mode == "explicit_path" else ["."]
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "DECISION_PERF_INTEGRATION",
            "DECISION_PERF_MOCKER_INTEGRATION",
            "PYTHONPATH",
            "PYTEST_ADDOPTS",
        }
    }
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if mode in {"integration", "mocker"}:
        env[
            "DECISION_PERF_INTEGRATION"
            if mode == "integration"
            else "DECISION_PERF_MOCKER_INTEGRATION"
        ] = "1"
    bootstrap = (
        "import importlib.util, pytest, sys; "
        "original = importlib.util.find_spec; "
        f"importlib.util.find_spec = lambda name, *a, **k: None if name == {missing!r} else original(name, *a, **k); "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    output = result.stdout + result.stderr
    if mode == "backend":
        assert result.returncode == 0, output
        assert "1 passed" in output
    else:
        assert result.returncode in {
            pytest.ExitCode.USAGE_ERROR,
            pytest.ExitCode.INTERRUPTED,
        }, output
        assert "Decision API instrument dependencies missing" in output
        assert missing in output
